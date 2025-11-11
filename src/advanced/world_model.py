"""
World Model for Model-Based Reinforcement Learning

Learns environment dynamics for planning and simulation:
- Transition model: predicts next state
- Reward model: predicts rewards
- Termination model: predicts episode end
- Planning: imagine trajectories using learned model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Callable, Optional
import numpy as np


class WorldModel(nn.Module):
    """
    World Model for model-based RL.
    
    Learns to predict environment dynamics:
    - Next state given current state and action
    - Immediate reward
    - Episode termination
    
    Can be used for:
    - Planning (Dyna-Q style)
    - Value prediction
    - Trajectory imagination
    
    Args:
        state_dim: Dimension of state
        action_dim: Number of discrete actions
        hidden_dim: Hidden layer size
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 9,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Transition model: (s, a) → s'
        self.transition_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Reward model: (s, a) → r
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Termination model: (s, a) → done
        self.done_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state, reward, and done flag.
        
        Args:
            state: Current state, shape (batch, state_dim)
            action: Action taken, shape (batch,) - discrete indices
            
        Returns:
            next_state: Predicted next state, shape (batch, state_dim)
            reward: Predicted reward, shape (batch, 1)
            done_prob: Probability of episode end, shape (batch, 1)
        """
        # One-hot encode action
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        
        # Concatenate state and action
        x = torch.cat([state, action_onehot], dim=-1)
        
        # Predict dynamics
        next_state = self.transition_net(x)
        reward = self.reward_net(x)
        done_prob = self.done_net(x)
        
        return next_state, reward, done_prob
    
    def imagine_trajectory(
        self,
        initial_state: torch.Tensor,
        policy_fn: Callable,
        horizon: int = 10,
        stochastic: bool = False
    ) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor]]:
        """
        Imagine a trajectory using the world model.
        
        Useful for planning and value estimation.
        
        Args:
            initial_state: Starting state, shape (1, state_dim)
            policy_fn: Function that takes state and returns action
            horizon: Number of steps to imagine
            stochastic: Whether to sample done based on probability
            
        Returns:
            states: List of imagined states
            actions: List of imagined actions
            rewards: List of imagined rewards
        """
        states = [initial_state]
        actions = []
        rewards = []
        
        state = initial_state
        
        for step in range(horizon):
            # Get action from policy
            action = policy_fn(state)
            
            # Predict next state, reward, done
            next_state, reward, done_prob = self.forward(state, action)
            
            states.append(next_state)
            actions.append(action.item() if isinstance(action, torch.Tensor) else action)
            rewards.append(reward)
            
            # Check termination
            if stochastic:
                done = torch.rand(1) < done_prob
            else:
                done = done_prob > 0.5
            
            if done:
                break
            
            state = next_state
        
        return states, actions, rewards
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute world model training loss.
        
        Args:
            states: States, shape (batch, state_dim)
            actions: Actions, shape (batch,)
            next_states: Next states, shape (batch, state_dim)
            rewards: Rewards, shape (batch, 1)
            dones: Done flags, shape (batch, 1)
            
        Returns:
            loss: Total world model loss
            metrics: Dictionary of loss components
        """
        # Forward pass
        pred_next_states, pred_rewards, pred_dones = self.forward(states, actions)
        
        # Transition loss
        transition_loss = F.mse_loss(pred_next_states, next_states)
        
        # Reward loss
        reward_loss = F.mse_loss(pred_rewards, rewards)
        
        # Termination loss
        done_loss = F.binary_cross_entropy(pred_dones, dones.float())
        
        # Total loss (weighted sum)
        total_loss = transition_loss + reward_loss + 0.1 * done_loss
        
        metrics = {
            'world_model_loss': total_loss.item(),
            'transition_loss': transition_loss.item(),
            'reward_loss': reward_loss.item(),
            'done_loss': done_loss.item()
        }
        
        return total_loss, metrics


class DynaQAgent:
    """
    Dyna-Q Agent: combines model-free RL with model-based planning.
    
    Learns both:
    - Q-values from real experience (model-free)
    - World model from real experience
    - Uses world model to generate simulated experience for planning
    
    Args:
        n_states: Number of discrete states
        n_actions: Number of actions
        world_model: Learned world model
        planning_steps: Number of simulated planning updates per real step
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        world_model: WorldModel,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        planning_steps: int = 5
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.world_model = world_model
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        
        # Q-table
        self.Q = np.zeros((n_states, n_actions))
        
        # Model (for simple tabular case)
        self.model = {}  # (s, a) → (r, s')
    
    def select_action(self, state_idx: int) -> int:
        """Select action using epsilon-greedy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state_idx])
    
    def update(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool
    ):
        """
        Update Q-values and world model, then perform planning.
        
        Args:
            state_idx: Current state
            action: Action taken
            reward: Reward received
            next_state_idx: Next state
            done: Whether episode ended
        """
        # Direct RL update (model-free)
        target = reward + (0 if done else self.gamma * np.max(self.Q[next_state_idx]))
        self.Q[state_idx, action] += self.lr * (target - self.Q[state_idx, action])
        
        # Update world model
        self.model[(state_idx, action)] = (reward, next_state_idx)
        
        # Planning: sample from model and update
        for _ in range(self.planning_steps):
            if len(self.model) == 0:
                break
            
            # Sample random previously experienced state-action
            (s, a), (r, s_next) = list(self.model.items())[
                np.random.randint(len(self.model))
            ]
            
            # Update Q-value using simulated experience
            target = r + self.gamma * np.max(self.Q[s_next])
            self.Q[s, a] += self.lr * (target - self.Q[s, a])


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Testing World Model...\n")
    
    # Test World Model
    print("1. World Model:")
    world_model = WorldModel(state_dim=5, action_dim=9, hidden_dim=128)
    
    batch_size = 32
    states = torch.randn(batch_size, 5)
    actions = torch.randint(0, 9, (batch_size,))
    
    next_states, rewards, dones = world_model(states, actions)
    
    print(f"   Input states shape: {states.shape}")
    print(f"   Predicted next states shape: {next_states.shape}")
    print(f"   Predicted rewards shape: {rewards.shape}")
    print(f"   Predicted dones shape: {dones.shape}")
    print(f"   ✓ World model forward pass works\n")
    
    # Test trajectory imagination
    print("2. Trajectory Imagination:")
    initial_state = torch.randn(1, 5)
    
    def random_policy(state):
        return torch.randint(0, 9, (1,))
    
    imagined_states, imagined_actions, imagined_rewards = world_model.imagine_trajectory(
        initial_state,
        random_policy,
        horizon=10
    )
    
    print(f"   Imagined {len(imagined_states)} states")
    print(f"   Imagined {len(imagined_actions)} actions")
    print(f"   Imagined {len(imagined_rewards)} rewards")
    print(f"   ✓ Trajectory imagination works\n")
    
    # Test world model training
    print("3. World Model Training:")
    true_next_states = torch.randn(batch_size, 5)
    true_rewards = torch.randn(batch_size, 1)
    true_dones = torch.randint(0, 2, (batch_size, 1))
    
    loss, metrics = world_model.compute_loss(
        states, actions, true_next_states, true_rewards, true_dones
    )
    
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   Transition loss: {metrics['transition_loss']:.4f}")
    print(f"   Reward loss: {metrics['reward_loss']:.4f}")
    print(f"   Done loss: {metrics['done_loss']:.4f}")
    print(f"   ✓ World model training works\n")
    
    # Test Dyna-Q
    print("4. Dyna-Q Agent:")
    dyna_agent = DynaQAgent(
        n_states=100,
        n_actions=9,
        world_model=world_model,
        planning_steps=5
    )
    
    # Simulate update
    dyna_agent.update(
        state_idx=10,
        action=3,
        reward=1.0,
        next_state_idx=15,
        done=False
    )
    
    print(f"   Q-table shape: {dyna_agent.Q.shape}")
    print(f"   Model entries: {len(dyna_agent.model)}")
    print(f"   ✓ Dyna-Q works\n")
    
    print("✅ All world model tests passed!")
