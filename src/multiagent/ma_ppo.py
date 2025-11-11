"""
Multi-Agent Proximal Policy Optimization (MAPPO)

Implements multi-agent PPO with:
- Centralized Training, Decentralized Execution (CTDE)
- Parameter sharing across agents
- Communication between agents
- Cooperative reward shaping
- Centralized critic with global state information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np

# Handle both package and standalone imports
try:
    from .communication import CommunicationNetwork
    from .coordination import CentralizedCritic, CooperationReward, TeamReward
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from communication import CommunicationNetwork
    from coordination import CentralizedCritic, CooperationReward, TeamReward


class MultiAgentActorCritic(nn.Module):
    """
    Actor-Critic network for multi-agent coordination.
    
    Features:
    - Individual observations + communication messages → Actor (decentralized)
    - Global state information → Critic (centralized)
    - Parameter sharing across agents (same network for all agents)
    
    Args:
        grid_shape: Shape of grid observation (H, W, C)
        state_dim: Dimension of agent state vector
        action_dim: Number of discrete actions
        n_agents: Number of agents
        message_dim: Dimension of communication messages
        use_communication: Whether to use inter-agent communication
    """
    
    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (7, 7, 5),
        state_dim: int = 5,
        action_dim: int = 9,
        n_agents: int = 2,
        message_dim: int = 16,
        use_communication: bool = True,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.n_agents = n_agents
        self.use_communication = use_communication
        self.grid_shape = grid_shape
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Communication module
        if use_communication:
            self.comm = CommunicationNetwork(
                state_dim=state_dim,
                message_dim=message_dim,
                hidden_dim=32
            )
            extra_dim = message_dim
        else:
            extra_dim = 0
        
        # Actor: processes local observation + messages
        # CNN for grid observation
        self.conv1 = nn.Conv2d(grid_shape[2], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        conv_out_dim = grid_shape[0] * grid_shape[1] * 64
        
        # FC for state vector
        self.state_fc = nn.Linear(state_dim, 64)
        
        # Combine features
        self.feature_fc = nn.Linear(conv_out_dim + 64 + extra_dim, hidden_dim)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Critic: centralized value function
        self.critic = CentralizedCritic(
            state_dim=hidden_dim,
            n_agents=n_agents,
            hidden_dim=256
        )
    
    def forward_actor(
        self,
        grid_obs: torch.Tensor,
        state_obs: torch.Tensor,
        messages: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for actor (decentralized).
        
        Args:
            grid_obs: Grid observation, shape (batch, C, H, W)
            state_obs: State vector, shape (batch, state_dim)
            messages: Communication messages, shape (batch, message_dim)
            
        Returns:
            logits: Action logits, shape (batch, action_dim)
            features: Feature vector for critic, shape (batch, hidden_dim)
        """
        # Process grid through CNN
        x = F.relu(self.conv1(grid_obs))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        
        # Process state vector
        s = F.relu(self.state_fc(state_obs))
        
        # Combine features
        if self.use_communication and messages is not None:
            features = torch.cat([x, s, messages], dim=1)
        else:
            features = torch.cat([x, s], dim=1)
        
        features = F.relu(self.feature_fc(features))
        
        # Actor output
        logits = self.actor(features)
        
        return logits, features
    
    def forward_critic(self, all_agent_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for critic (centralized).
        
        Args:
            all_agent_features: List of feature tensors, each (batch, hidden_dim)
            
        Returns:
            value: State value, shape (batch, 1)
        """
        # Stack features: (batch, n_agents, hidden_dim)
        features_stacked = torch.stack(all_agent_features, dim=1)
        value = self.critic(features_stacked)
        return value


class MultiAgentPPO:
    """
    Multi-Agent Proximal Policy Optimization.
    
    Implements CTDE (Centralized Training, Decentralized Execution):
    - Training: Critic uses global state information
    - Execution: Only local actor is used for action selection
    
    Features:
    - Parameter sharing across agents
    - Communication between agents
    - Cooperative reward shaping
    - GAE for advantage estimation
    - PPO clipped objective
    """
    
    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (7, 7, 5),
        state_dim: int = 5,
        action_dim: int = 9,
        n_agents: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_communication: bool = True,
        cooperation_bonus: float = 0.5,
        device: str = "cpu"
    ):
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        
        # Network
        self.network = MultiAgentActorCritic(
            grid_shape=grid_shape,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            use_communication=use_communication
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        # Cooperation mechanisms
        self.cooperation_reward = CooperationReward(
            proximity_bonus=cooperation_bonus,
            sharing_bonus=cooperation_bonus * 2,
            joint_bonus=cooperation_bonus * 3
        )
        
        self.team_reward = TeamReward(
            individual_weight=0.6,
            team_weight=0.4
        )
        
        # Rollout buffer
        self.rollout_buffer = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
    
    def select_action(
        self,
        observations: List[Dict[str, np.ndarray]],
        training: bool = True
    ) -> Tuple[List[int], List[float], Optional[float]]:
        """
        Select actions for all agents.
        
        Args:
            observations: List of observation dicts for each agent
            training: Whether in training mode
            
        Returns:
            actions: List of actions for each agent
            log_probs: List of log probabilities
            value: Centralized value estimate (only in training)
        """
        # Convert observations to tensors
        grids = []
        states = []
        
        for obs in observations:
            grid = torch.FloatTensor(obs['grid']).unsqueeze(0)  # (1, H, W, C)
            grid = grid.permute(0, 3, 1, 2)  # (1, C, H, W)
            state = torch.FloatTensor(obs['state']).unsqueeze(0)  # (1, state_dim)
            
            grids.append(grid.to(self.device))
            states.append(state.to(self.device))
        
        # Generate messages if using communication
        messages = None
        if self.network.use_communication and training:
            messages = []
            for i, state in enumerate(states):
                # Get messages from other agents
                msg = self.network.comm(states, agent_idx=i)
                messages.append(msg)
        
        # Get actions and values for each agent
        actions = []
        log_probs = []
        features_all = []
        
        with torch.no_grad() if not training else torch.enable_grad():
            for i in range(self.n_agents):
                msg = messages[i] if messages is not None else None
                logits, features = self.network.forward_actor(grids[i], states[i], msg)
                
                # Sample action
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                actions.append(action.item())
                log_probs.append(log_prob.item())
                features_all.append(features)
            
            # Get centralized value
            if training:
                value = self.network.forward_critic(features_all)
                value = value.item()
            else:
                value = None
        
        return actions, log_probs, value
    
    def store_transition(
        self,
        observations: List[Dict[str, np.ndarray]],
        actions: List[int],
        log_probs: List[float],
        rewards: List[float],
        value: float,
        done: bool
    ):
        """Store transition in rollout buffer."""
        self.rollout_buffer['observations'].append(observations)
        self.rollout_buffer['actions'].append(actions)
        self.rollout_buffer['log_probs'].append(log_probs)
        self.rollout_buffer['rewards'].append(rewards)
        self.rollout_buffer['values'].append(value)
        self.rollout_buffer['dones'].append(done)
    
    def compute_advantages(
        self,
        rewards: List[List[float]],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.
        
        Args:
            rewards: List of reward lists (one per timestep, each with n_agents rewards)
            values: List of value estimates
            dones: List of done flags
            
        Returns:
            advantages: Tensor of advantages
            returns: Tensor of returns
        """
        advantages = []
        returns = []
        
        # Convert to team rewards
        team_rewards = [sum(r) / len(r) for r in rewards]  # Average reward
        
        # Compute GAE
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = team_rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, epochs: int = 4) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            epochs: Number of optimization epochs
            
        Returns:
            metrics: Dictionary of training metrics
        """
        if len(self.rollout_buffer['observations']) == 0:
            return {}
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(
            self.rollout_buffer['rewards'],
            self.rollout_buffer['values'],
            self.rollout_buffer['dones']
        )
        
        # Prepare batch data
        observations = self.rollout_buffer['observations']
        actions = self.rollout_buffer['actions']
        old_log_probs = self.rollout_buffer['log_probs']
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0
        
        # PPO update epochs
        for epoch in range(epochs):
            for t in range(len(observations)):
                # Convert observations to tensors
                grids = []
                states = []
                
                for obs in observations[t]:
                    grid = torch.FloatTensor(obs['grid']).unsqueeze(0).permute(0, 3, 1, 2)
                    state = torch.FloatTensor(obs['state']).unsqueeze(0)
                    grids.append(grid.to(self.device))
                    states.append(state.to(self.device))
                
                # Generate messages
                messages = None
                if self.network.use_communication:
                    messages = []
                    for i in range(self.n_agents):
                        msg = self.network.comm(states, agent_idx=i)
                        messages.append(msg)
                
                # Forward pass for all agents
                new_log_probs_list = []
                entropies = []
                features_all = []
                
                for i in range(self.n_agents):
                    msg = messages[i] if messages is not None else None
                    logits, features = self.network.forward_actor(grids[i], states[i], msg)
                    
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    
                    action_tensor = torch.LongTensor([actions[t][i]]).to(self.device)
                    new_log_prob = dist.log_prob(action_tensor)
                    entropy = dist.entropy()
                    
                    new_log_probs_list.append(new_log_prob)
                    entropies.append(entropy)
                    features_all.append(features)
                
                # Sum log probs across agents
                new_log_probs = sum(new_log_probs_list)
                old_log_prob_sum = sum(old_log_probs[t])
                old_log_prob_tensor = torch.FloatTensor([old_log_prob_sum]).to(self.device)
                
                # Ratio for PPO
                ratio = torch.exp(new_log_probs - old_log_prob_tensor)
                
                # Clipped surrogate objective
                adv = advantages[t].unsqueeze(0)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value = self.network.forward_critic(features_all)
                value_loss = F.mse_loss(value, returns[t].unsqueeze(0).unsqueeze(0))
                
                # Entropy bonus
                entropy_loss = -sum(entropies).mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                update_count += 1
        
        # Clear buffer
        for key in self.rollout_buffer:
            self.rollout_buffer[key] = []
        
        # Return metrics
        return {
            'policy_loss': total_policy_loss / update_count if update_count > 0 else 0,
            'value_loss': total_value_loss / update_count if update_count > 0 else 0,
            'entropy': total_entropy / update_count if update_count > 0 else 0
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Testing Multi-Agent PPO...\n")
    
    # Create MAPPO agent
    print("1. Creating MAPPO agent:")
    mappo = MultiAgentPPO(
        grid_shape=(7, 7, 5),
        state_dim=5,
        action_dim=9,
        n_agents=3,
        use_communication=True,
        device="cpu"
    )
    print(f"   ✓ MAPPO agent created")
    print(f"   Agents: {mappo.n_agents}")
    print(f"   Communication: {mappo.network.use_communication}\n")
    
    # Test action selection
    print("2. Testing action selection:")
    observations = [
        {
            'grid': np.random.rand(7, 7, 5),
            'state': np.random.rand(5)
        }
        for _ in range(3)
    ]
    
    actions, log_probs, value = mappo.select_action(observations, training=True)
    print(f"   Actions: {actions}")
    print(f"   Log probs: {[f'{lp:.3f}' for lp in log_probs]}")
    print(f"   Value: {value:.3f}")
    print(f"   ✓ Action selection works\n")
    
    # Test rollout and update
    print("3. Testing rollout and update:")
    for step in range(10):
        actions, log_probs, value = mappo.select_action(observations, training=True)
        rewards = [np.random.rand() for _ in range(3)]
        done = step == 9
        
        mappo.store_transition(observations, actions, log_probs, rewards, value, done)
    
    metrics = mappo.update(epochs=2)
    print(f"   Policy loss: {metrics['policy_loss']:.4f}")
    print(f"   Value loss: {metrics['value_loss']:.4f}")
    print(f"   Entropy: {metrics['entropy']:.4f}")
    print(f"   ✓ Update works\n")
    
    print("✅ All MAPPO tests passed!")
