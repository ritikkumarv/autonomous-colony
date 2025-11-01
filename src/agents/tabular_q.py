"""
Tabular Q-Learning Agent
Classic reinforcement learning with table lookup
"""

import numpy as np
import random
from collections import defaultdict
from typing import Dict
from .base_agent import BaseAgent


class TabularQLearningAgent(BaseAgent):
    """
    Classic Q-Learning with table lookup.
    
    RL Concepts:
    - Value-based learning
    - Temporal Difference (TD) learning
    - ε-greedy exploration
    - Bellman equation: Q(s,a) = Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        super().__init__(state_dim, action_dim)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action values
        self.q_table = defaultdict(lambda: np.zeros(action_dim))
        self.state_visits = defaultdict(int)
        
    def _discretize_state(self, observation: Dict) -> tuple:
        """Convert continuous observation to discrete state"""
        state_vec = observation['state']
        # Discretize to 10 bins
        discrete = tuple((state_vec * 10).astype(int).clip(0, 9))
        
        # Add grid info (simplified)
        grid = observation['grid']
        # Count nearby resources
        center = grid.shape[0] // 2
        local_view = grid[center-1:center+2, center-1:center+2, :]
        food_count = int(local_view[:, :, 1].sum())
        water_count = int(local_view[:, :, 2].sum())
        
        return discrete + (food_count, water_count)
    
    def select_action(self, observation: Dict, training: bool = True) -> int:
        """ε-greedy action selection"""
        state = self._discretize_state(observation)
        
        # Exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Exploitation
        q_values = self.q_table[state]
        return int(np.argmax(q_values))
    
    def update(self, state: Dict, action: int, reward: float, 
               next_state: Dict, done: bool):
        """Q-Learning update"""
        s = self._discretize_state(state)
        s_next = self._discretize_state(next_state)
        
        # Current Q-value
        q_current = self.q_table[s][action]
        
        # Target Q-value
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.q_table[s_next])
        
        # TD error
        td_error = q_target - q_current
        
        # Update
        self.q_table[s][action] += self.lr * td_error
        self.state_visits[s] += 1
        
        return td_error
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
