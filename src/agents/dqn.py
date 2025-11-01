"""
Deep Q-Network (DQN) Agent
Deep reinforcement learning with experience replay and target network
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from typing import Dict, Tuple
from .base_agent import BaseAgent


class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.
    Handles both grid observations and internal state.
    """
    
    def __init__(self, grid_shape: tuple, state_dim: int, action_dim: int):
        super().__init__()
        
        # CNN for grid observation
        self.conv1 = nn.Conv2d(grid_shape[2], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Calculate conv output size
        conv_out_size = grid_shape[0] * grid_shape[1] * 64
        
        # FC for internal state
        self.state_fc = nn.Linear(state_dim, 64)
        
        # Combined layers
        self.fc1 = nn.Linear(conv_out_size + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, grid_obs, state_obs):
        # Process grid
        x = F.relu(self.conv1(grid_obs))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        
        # Process state
        s = F.relu(self.state_fc(state_obs))
        
        # Combine
        combined = torch.cat([x, s], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        
        return q_values


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network with experience replay and target network.
    
    RL Concepts:
    - Function approximation (neural networks)
    - Experience replay (decorrelation)
    - Target network (stability)
    - Double DQN (optional)
    """
    
    def __init__(
        self,
        grid_shape: tuple,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100
    ):
        super().__init__(state_dim, action_dim)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(grid_shape, state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(grid_shape, state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.update_count = 0
        self.loss_history = []
        
        print(f"✓ DQN Agent initialized on {self.device}")
    
    def _obs_to_tensor(self, obs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert observation dict to tensors"""
        grid = torch.FloatTensor(obs['grid']).permute(2, 0, 1).unsqueeze(0).to(self.device)
        state = torch.FloatTensor(obs['state']).unsqueeze(0).to(self.device)
        return grid, state
    
    def select_action(self, observation: Dict, training: bool = True) -> int:
        """ε-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        grid, state = self._obs_to_tensor(observation)
        with torch.no_grad():
            q_values = self.q_network(grid, state)
        return int(q_values.argmax().item())
    
    def update(self, state: Dict, action: int, reward: float,
               next_state: Dict, done: bool):
        """Store transition and train if buffer is ready"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        grids = torch.stack([torch.FloatTensor(s['grid']).permute(2, 0, 1) for s in states]).to(self.device)
        state_vecs = torch.stack([torch.FloatTensor(s['state']) for s in states]).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_grids = torch.stack([torch.FloatTensor(s['grid']).permute(2, 0, 1) for s in next_states]).to(self.device)
        next_state_vecs = torch.stack([torch.FloatTensor(s['state']) for s in next_states]).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(grids, state_vecs).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_grids, next_state_vecs).max(1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)
        
        # Loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.loss_history.append(loss.item())
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """Save model weights"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
