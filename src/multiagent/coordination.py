"""
Multi-Agent Coordination Module

Implements coordination mechanisms for multi-agent reinforcement learning:
- Centralized Critic (CTDE - Centralized Training Decentralized Execution)
- Value Decomposition (QMIX, VDN)
- Cooperation Bonuses
- Reward Shaping for Coordination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class CentralizedCritic(nn.Module):
    """
    Centralized Critic for CTDE (Centralized Training, Decentralized Execution).
    
    During training, critic has access to global state (all agents' observations).
    During execution, only local actor is used (decentralized).
    
    Args:
        state_dim: Dimension of individual agent state
        n_agents: Number of agents
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        
        # Global state is concatenation of all agent states
        global_state_dim = state_dim * n_agents
        
        self.fc1 = nn.Linear(global_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, agent_states: torch.Tensor) -> torch.Tensor:
        """
        Compute global value from all agent states.
        
        Args:
            agent_states: Tensor of shape (batch, n_agents, state_dim)
            
        Returns:
            value: Tensor of shape (batch, 1)
        """
        # Flatten agent states to global state
        batch_size = agent_states.shape[0]
        global_state = agent_states.view(batch_size, -1)
        
        x = F.relu(self.fc1(global_state))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        
        return value


class ValueDecompositionNetwork(nn.Module):
    """
    Value Decomposition Network (VDN).
    
    Decomposes global Q-value into sum of individual agent Q-values.
    Q_tot(s, a) = Σ Q_i(s_i, a_i)
    
    Simpler than QMIX but effective for many scenarios.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Individual Q-network (shared across agents)
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, agent_states: torch.Tensor) -> torch.Tensor:
        """
        Compute individual Q-values and sum them.
        
        Args:
            agent_states: Tensor of shape (batch, n_agents, state_dim)
            
        Returns:
            q_values: Tensor of shape (batch, n_agents, action_dim)
        """
        batch_size, n_agents, state_dim = agent_states.shape
        
        # Flatten to process all agents at once
        states_flat = agent_states.view(-1, state_dim)
        q_values_flat = self.q_network(states_flat)
        
        # Reshape back
        action_dim = q_values_flat.shape[-1]
        q_values = q_values_flat.view(batch_size, n_agents, action_dim)
        
        return q_values
    
    def get_total_q(
        self,
        agent_states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Get total Q-value for given states and actions.
        
        Args:
            agent_states: Tensor of shape (batch, n_agents, state_dim)
            actions: Tensor of shape (batch, n_agents)
            
        Returns:
            total_q: Tensor of shape (batch,)
        """
        q_values = self.forward(agent_states)  # (batch, n_agents, action_dim)
        
        # Gather Q-values for selected actions
        batch_size, n_agents = actions.shape
        actions_expanded = actions.unsqueeze(-1)  # (batch, n_agents, 1)
        q_selected = torch.gather(q_values, 2, actions_expanded)  # (batch, n_agents, 1)
        q_selected = q_selected.squeeze(-1)  # (batch, n_agents)
        
        # Sum across agents (VDN decomposition)
        total_q = q_selected.sum(dim=1)  # (batch,)
        
        return total_q


class QMIXMixer(nn.Module):
    """
    QMIX Mixing Network.
    
    Learns to mix individual Q-values into global Q-value using hypernetworks.
    Ensures monotonicity: ∂Q_tot/∂Q_i >= 0
    
    More expressive than VDN while maintaining IGM (Individual-Global-Max) property.
    """
    
    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        hidden_dim: int = 32
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Hypernetworks generate mixing weights from global state
        # Weight for first layer
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * hidden_dim)
        )
        
        # Bias for first layer
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Weight for second layer
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Bias for second layer (scalar)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        agent_q_values: torch.Tensor,
        global_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Mix individual Q-values into global Q-value.
        
        Args:
            agent_q_values: Individual Q-values, shape (batch, n_agents)
            global_state: Global state, shape (batch, state_dim)
            
        Returns:
            q_total: Global Q-value, shape (batch, 1)
        """
        batch_size = agent_q_values.shape[0]
        
        # Generate mixing weights (ensure monotonicity with abs)
        w1 = torch.abs(self.hyper_w1(global_state))  # (batch, n_agents * hidden_dim)
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)  # (batch, n_agents, hidden)
        
        b1 = self.hyper_b1(global_state)  # (batch, hidden_dim)
        
        w2 = torch.abs(self.hyper_w2(global_state))  # (batch, hidden_dim)
        w2 = w2.view(batch_size, self.hidden_dim, 1)  # (batch, hidden, 1)
        
        b2 = self.hyper_b2(global_state)  # (batch, 1)
        
        # First layer mixing
        agent_q_values = agent_q_values.unsqueeze(-1)  # (batch, n_agents, 1)
        hidden = torch.matmul(w1.transpose(1, 2), agent_q_values)  # (batch, hidden, 1)
        hidden = hidden.squeeze(-1) + b1  # (batch, hidden)
        hidden = F.relu(hidden)
        
        # Second layer mixing
        hidden = hidden.unsqueeze(-1)  # (batch, hidden, 1)
        q_total = torch.matmul(w2.transpose(1, 2), hidden)  # (batch, 1, 1)
        q_total = q_total.squeeze(-1) + b2  # (batch, 1)
        
        return q_total


class CooperationReward:
    """
    Reward shaping for encouraging cooperation between agents.
    
    Provides additional rewards for coordinated behaviors:
    - Proximity bonus (agents working together)
    - Resource sharing bonus
    - Joint task completion bonus
    """
    
    def __init__(
        self,
        proximity_bonus: float = 0.1,
        sharing_bonus: float = 0.5,
        joint_bonus: float = 1.0,
        proximity_threshold: float = 3.0
    ):
        self.proximity_bonus = proximity_bonus
        self.sharing_bonus = sharing_bonus
        self.joint_bonus = joint_bonus
        self.proximity_threshold = proximity_threshold
    
    def compute_proximity_reward(
        self,
        positions: List[Tuple[int, int]]
    ) -> float:
        """
        Reward agents for being close to each other.
        
        Args:
            positions: List of (x, y) positions for each agent
            
        Returns:
            reward: Proximity bonus
        """
        if len(positions) < 2:
            return 0.0
        
        total_reward = 0.0
        count = 0
        
        # Check pairwise distances
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                
                if distance <= self.proximity_threshold:
                    total_reward += self.proximity_bonus
                    count += 1
        
        return total_reward
    
    def compute_sharing_reward(
        self,
        agent_resources: List[dict]
    ) -> float:
        """
        Reward agents for balanced resource distribution.
        
        Args:
            agent_resources: List of dicts with 'food', 'water', 'material' counts
            
        Returns:
            reward: Sharing bonus if resources are balanced
        """
        if len(agent_resources) < 2:
            return 0.0
        
        # Check resource balance across agents
        total_food = sum(r.get('food', 0) for r in agent_resources)
        total_water = sum(r.get('water', 0) for r in agent_resources)
        
        if total_food == 0 or total_water == 0:
            return 0.0
        
        # Compute variance in resource distribution
        food_counts = [r.get('food', 0) for r in agent_resources]
        water_counts = [r.get('water', 0) for r in agent_resources]
        
        food_var = np.var(food_counts)
        water_var = np.var(water_counts)
        
        # Reward low variance (balanced distribution)
        # Lower variance = better sharing
        max_var = len(agent_resources)  # Normalize
        balance_score = 1.0 - min((food_var + water_var) / (2 * max_var), 1.0)
        
        return self.sharing_bonus * balance_score
    
    def compute_joint_task_reward(
        self,
        tasks_completed: int,
        agents_involved: int
    ) -> float:
        """
        Bonus for completing tasks with multiple agents.
        
        Args:
            tasks_completed: Number of tasks completed
            agents_involved: Number of agents that contributed
            
        Returns:
            reward: Joint task bonus
        """
        if tasks_completed == 0 or agents_involved < 2:
            return 0.0
        
        # Bonus scales with number of agents involved
        cooperation_factor = agents_involved / 2.0  # Assume 2+ agents needed
        return self.joint_bonus * tasks_completed * cooperation_factor


class TeamReward:
    """
    Team-based reward structure.
    
    Combines individual and team rewards to balance
    individual initiative with team coordination.
    """
    
    def __init__(
        self,
        individual_weight: float = 0.5,
        team_weight: float = 0.5
    ):
        self.individual_weight = individual_weight
        self.team_weight = team_weight
    
    def compute_rewards(
        self,
        individual_rewards: List[float],
        team_reward: float
    ) -> List[float]:
        """
        Combine individual and team rewards.
        
        Args:
            individual_rewards: List of individual agent rewards
            team_reward: Shared team reward
            
        Returns:
            combined_rewards: List of combined rewards for each agent
        """
        combined = []
        for ind_reward in individual_rewards:
            combined_reward = (
                self.individual_weight * ind_reward +
                self.team_weight * team_reward
            )
            combined.append(combined_reward)
        
        return combined


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Testing Coordination Mechanisms...\n")
    
    # Test Centralized Critic
    print("1. Centralized Critic:")
    critic = CentralizedCritic(state_dim=5, n_agents=3, hidden_dim=128)
    agent_states = torch.randn(4, 3, 5)  # (batch=4, n_agents=3, state_dim=5)
    value = critic(agent_states)
    print(f"   Input shape: {agent_states.shape}")
    print(f"   Global value shape: {value.shape}")
    print(f"   ✓ Centralized critic works\n")
    
    # Test VDN
    print("2. Value Decomposition Network (VDN):")
    vdn = ValueDecompositionNetwork(state_dim=5, action_dim=9, hidden_dim=128)
    q_values = vdn(agent_states)
    print(f"   Q-values shape: {q_values.shape}")
    
    actions = torch.randint(0, 9, (4, 3))  # (batch=4, n_agents=3)
    total_q = vdn.get_total_q(agent_states, actions)
    print(f"   Total Q shape: {total_q.shape}")
    print(f"   ✓ VDN works\n")
    
    # Test QMIX
    print("3. QMIX Mixer:")
    qmix = QMIXMixer(n_agents=3, state_dim=15, hidden_dim=32)
    agent_q = torch.randn(4, 3)  # Individual Q-values
    global_state = torch.randn(4, 15)  # Global state
    q_total = qmix(agent_q, global_state)
    print(f"   Individual Q shape: {agent_q.shape}")
    print(f"   Total Q shape: {q_total.shape}")
    print(f"   ✓ QMIX works\n")
    
    # Test Cooperation Rewards
    print("4. Cooperation Rewards:")
    coop_reward = CooperationReward()
    
    positions = [(5, 5), (6, 6), (10, 10)]  # Two close, one far
    proximity_r = coop_reward.compute_proximity_reward(positions)
    print(f"   Proximity reward: {proximity_r:.3f}")
    
    resources = [
        {'food': 3, 'water': 2},
        {'food': 2, 'water': 3},
        {'food': 3, 'water': 2}
    ]
    sharing_r = coop_reward.compute_sharing_reward(resources)
    print(f"   Sharing reward: {sharing_r:.3f}")
    
    joint_r = coop_reward.compute_joint_task_reward(tasks_completed=2, agents_involved=3)
    print(f"   Joint task reward: {joint_r:.3f}")
    print(f"   ✓ Cooperation rewards work\n")
    
    # Test Team Rewards
    print("5. Team Rewards:")
    team_reward_sys = TeamReward(individual_weight=0.6, team_weight=0.4)
    ind_rewards = [1.0, 2.0, 1.5]
    team_r = 3.0
    combined = team_reward_sys.compute_rewards(ind_rewards, team_r)
    print(f"   Individual rewards: {ind_rewards}")
    print(f"   Team reward: {team_r}")
    print(f"   Combined rewards: {[f'{r:.2f}' for r in combined]}")
    print(f"   ✓ Team rewards work\n")
    
    print("✅ All coordination tests passed!")
