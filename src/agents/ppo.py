"""
Proximal Policy Optimization (PPO) Agent
Modern policy gradient method with actor-critic architecture
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional, Tuple
from .base_agent import BaseAgent


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    Actor outputs policy, Critic outputs value function.
    """
    
    def __init__(self, grid_shape: tuple, state_dim: int, action_dim: int):
        super().__init__()
        
        # Shared feature extractor
        self.conv1 = nn.Conv2d(grid_shape[2], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        conv_out_size = grid_shape[0] * grid_shape[1] * 64
        
        self.state_fc = nn.Linear(state_dim, 64)
        self.shared_fc = nn.Linear(conv_out_size + 64, 256)
        
        # Actor head (policy)
        self.actor_fc = nn.Linear(256, 128)
        self.actor_out = nn.Linear(128, action_dim)
        
        # Critic head (value)
        self.critic_fc = nn.Linear(256, 128)
        self.critic_out = nn.Linear(128, 1)
    
    def forward(self, grid_obs, state_obs):
        # Shared features
        x = F.relu(self.conv1(grid_obs))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        s = F.relu(self.state_fc(state_obs))
        features = F.relu(self.shared_fc(torch.cat([x, s], dim=1)))
        
        # Actor (policy logits)
        actor_x = F.relu(self.actor_fc(features))
        logits = self.actor_out(actor_x)
        
        # Critic (value)
        critic_x = F.relu(self.critic_fc(features))
        value = self.critic_out(critic_x)
        
        return logits, value
    
    def get_action_and_value(self, grid_obs, state_obs, action=None):
        logits, value = self(grid_obs, state_obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization - modern policy gradient method.
    
    RL Concepts:
    - Policy gradient theorem
    - Actor-Critic architecture
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Entropy regularization for exploration
    """
    
    def __init__(
        self,
        grid_shape: tuple,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64
    ):
        super().__init__(state_dim, action_dim)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(grid_shape, state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Rollout storage
        self.rollout_buffer = []
        self.loss_history = []
        
        print(f"✓ PPO Agent initialized on {self.device}")
    
    def select_action(self, observation: Dict, training: bool = True) -> Tuple[int, float, float]:
        """
        Sample action from policy.
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        grid = torch.FloatTensor(observation['grid']).permute(2, 0, 1).unsqueeze(0).to(self.device)
        state = torch.FloatTensor(observation['state']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action_and_value(grid, state)
        
        return int(action.item()), log_prob.item(), value.item()
    
    def store_transition(self, state: Dict, action: int, reward: float, 
                        log_prob: float, value: float, done: bool):
        """Store transition for batch update"""
        self.rollout_buffer.append((state, action, reward, log_prob, value, done))
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update(self) -> Optional[float]:
        """PPO update using collected rollouts"""
        if len(self.rollout_buffer) < self.batch_size:
            return None
        
        # Prepare batch
        states, actions, rewards, old_log_probs, values, dones = zip(*self.rollout_buffer)
        
        # Compute advantages
        advantages, returns = self.compute_gae(list(rewards), list(values), list(dones))
        
        # Convert to tensors
        grids = torch.stack([torch.FloatTensor(s['grid']).permute(2, 0, 1) for s in states]).to(self.device)
        state_vecs = torch.stack([torch.FloatTensor(s['state']) for s in states]).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # PPO epochs
        total_loss = 0
        for _ in range(self.n_epochs):
            # Get current policy
            _, new_log_probs, entropy, values_pred = self.network.get_action_and_value(
                grids, state_vecs, actions_t
            )
            
            # Ratio for clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            
            # Policy loss with clipping
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values_pred.squeeze(), returns_t)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / self.n_epochs
        self.loss_history.append(avg_loss)
        
        # Clear buffer
        self.rollout_buffer = []
        
        return avg_loss
    
    def save(self, path: str):
        """Save model weights"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
