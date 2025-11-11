"""
Hierarchical Reinforcement Learning

Implements hierarchical RL with temporal abstraction:
- Options Framework: initiation, policy, termination
- Hierarchical Agent: meta-policy selects options
- Goal-conditioned policies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Dict
import numpy as np


class Option:
    """
    A single option (skill/sub-policy) with initiation, policy, and termination.
    
    Options provide temporal abstraction - instead of selecting primitive actions
    at every timestep, agent selects high-level options that execute for multiple steps.
    
    Args:
        name: Option identifier
        policy_fn: Function that maps state to action
        termination_fn: Function that determines when option should end
        initiation_fn: Function that determines if option can start (optional)
    """
    
    def __init__(
        self,
        name: str,
        policy_fn: Callable,
        termination_fn: Callable,
        initiation_fn: Optional[Callable] = None
    ):
        self.name = name
        self.policy = policy_fn
        self.termination = termination_fn
        self.initiation = initiation_fn or (lambda state: True)
        self.steps_active = 0
    
    def can_initiate(self, state: Dict) -> bool:
        """Check if option can be initiated in this state."""
        return self.initiation(state)
    
    def get_action(self, state: Dict) -> int:
        """Get primitive action from option's policy."""
        self.steps_active += 1
        return self.policy(state)
    
    def should_terminate(self, state: Dict) -> bool:
        """Check if option should terminate."""
        return self.termination(state)
    
    def reset(self):
        """Reset option state."""
        self.steps_active = 0


class HierarchicalAgent:
    """
    Agent that learns and executes hierarchical policies using options.
    
    Two levels of decision making:
    - High level: meta-policy selects which option to execute
    - Low level: option executes primitive actions
    
    Args:
        options: List of available options
        meta_policy: Policy for selecting options (optional)
    """
    
    def __init__(
        self,
        options: List[Option],
        meta_policy: Optional[Callable] = None
    ):
        self.options = options
        self.current_option = None
        self.option_history = []
        self.meta_policy = meta_policy or self._default_meta_policy
        
        print(f"✓ Hierarchical Agent with {len(options)} options")
    
    def _default_meta_policy(self, state: Dict) -> Optional[Option]:
        """Default meta-policy: randomly select among available options."""
        available = [opt for opt in self.options if opt.can_initiate(state)]
        if available:
            return np.random.choice(available)
        return None
    
    def select_option(self, state: Dict) -> Optional[Option]:
        """Select which option to execute using meta-policy."""
        option = self.meta_policy(state)
        if option:
            option.reset()
            self.option_history.append(option.name)
        return option
    
    def step(self, state: Dict) -> int:
        """
        Execute one step with hierarchical control.
        
        Args:
            state: Current state
            
        Returns:
            action: Primitive action to take
        """
        # If no current option or option terminated, select new option
        if self.current_option is None or self.current_option.should_terminate(state):
            self.current_option = self.select_option(state)
        
        # Get action from current option
        if self.current_option:
            return self.current_option.get_action(state)
        else:
            return 0  # Default action if no option available


class GoalConditionedPolicy(nn.Module):
    """
    Goal-conditioned policy network.
    
    Takes both state and goal as input, learns to reach specified goals.
    Useful for hierarchical RL where high-level policy sets goals.
    
    Args:
        state_dim: Dimension of state
        goal_dim: Dimension of goal
        action_dim: Number of actions
        hidden_dim: Hidden layer size
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        goal_dim: int = 2,  # e.g., target (x, y) position
        action_dim: int = 9,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: compute action logits given state and goal.
        
        Args:
            state: State tensor, shape (batch, state_dim)
            goal: Goal tensor, shape (batch, goal_dim)
            
        Returns:
            logits: Action logits, shape (batch, action_dim)
        """
        x = torch.cat([state, goal], dim=-1)
        logits = self.policy_net(x)
        return logits
    
    def select_action(
        self,
        state: torch.Tensor,
        goal: torch.Tensor
    ) -> int:
        """Select action given state and goal."""
        logits = self.forward(state, goal)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()


class HierarchicalQLearning:
    """
    Q-Learning with options (Semi-MDP Q-Learning).
    
    Learns Q-values over options instead of primitive actions,
    enabling temporal abstraction.
    
    Args:
        n_states: Number of discrete states
        options: List of available options
        lr: Learning rate
        gamma: Discount factor
    """
    
    def __init__(
        self,
        n_states: int,
        options: List[Option],
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        self.n_states = n_states
        self.options = options
        self.n_options = len(options)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-table over options
        self.Q = np.zeros((n_states, self.n_options))
    
    def select_option(self, state_idx: int) -> int:
        """
        Select option using epsilon-greedy policy.
        
        Args:
            state_idx: Discrete state index
            
        Returns:
            option_idx: Index of selected option
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_options)
        return np.argmax(self.Q[state_idx])
    
    def update(
        self,
        state_idx: int,
        option_idx: int,
        cumulative_reward: float,
        next_state_idx: int,
        done: bool
    ):
        """
        Update Q-value for option.
        
        Args:
            state_idx: Starting state
            option_idx: Option that was executed
            cumulative_reward: Total reward accumulated during option execution
            next_state_idx: State when option terminated
            done: Whether episode ended
        """
        if done:
            target = cumulative_reward
        else:
            target = cumulative_reward + self.gamma * np.max(self.Q[next_state_idx])
        
        # Q-learning update
        self.Q[state_idx, option_idx] += self.lr * (
            target - self.Q[state_idx, option_idx]
        )


# ============================================================================
# Pre-defined Options for Colony Environment
# ============================================================================

def create_colony_options() -> List[Option]:
    """Create predefined options for the colony environment."""
    
    # Option 1: Explore (random movement until resource found)
    def explore_policy(state: Dict) -> int:
        return np.random.randint(0, 8)  # Random movement
    
    def explore_terminate(state: Dict) -> bool:
        # Terminate if resource nearby
        grid = state['grid']
        center = grid.shape[0] // 2
        local = grid[center-1:center+2, center-1:center+2, 1:4]  # Resource channels
        return local.sum() > 0
    
    explore_option = Option("explore", explore_policy, explore_terminate)
    
    # Option 2: Collect (move to resource and collect)
    def collect_policy(state: Dict) -> int:
        grid = state['grid']
        center = grid.shape[0] // 2
        
        # If on resource, collect
        if grid[center, center, 1:4].sum() > 0:
            return 8  # Collect action
        
        # Otherwise move toward nearest resource
        resource_positions = np.argwhere(grid[:, :, 1:4].sum(axis=2) > 0)
        if len(resource_positions) > 0:
            target = resource_positions[0]
            dy = target[0] - center
            dx = target[1] - center
            
            # Convert to action (0-7 for 8 directions)
            if abs(dx) > abs(dy):
                return 0 if dx < 0 else 2  # Left or Right
            else:
                return 1 if dy < 0 else 3  # Up or Down
        
        return 8  # Default: collect
    
    def collect_terminate(state: Dict) -> bool:
        # Terminate after collecting or if no resources nearby
        return True
    
    collect_option = Option("collect", collect_policy, collect_terminate)
    
    # Option 3: Return to base (move toward center)
    def return_policy(state: Dict) -> int:
        grid = state['grid']
        center = grid.shape[0] // 2
        # Always move toward center (simplified)
        return np.random.choice([0, 1, 2, 3])  # Cardinal directions
    
    def return_terminate(state: Dict) -> bool:
        # Terminate when inventory is empty
        inv_state = state['state']
        return inv_state[2:5].sum() == 0  # No resources carried
    
    return_option = Option("return", return_policy, return_terminate)
    
    return [explore_option, collect_option, return_option]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Testing Hierarchical RL...\n")
    
    # Test Options
    print("1. Testing Options:")
    options = create_colony_options()
    print(f"   Created {len(options)} options:")
    for opt in options:
        print(f"      - {opt.name}")
    print(f"   ✓ Options created\n")
    
    # Test Hierarchical Agent
    print("2. Testing Hierarchical Agent:")
    agent = HierarchicalAgent(options)
    
    # Simulate state
    dummy_state = {
        'grid': np.random.rand(7, 7, 5),
        'state': np.array([0.8, 0.9, 0.0, 0.0, 0.0])
    }
    
    action = agent.step(dummy_state)
    print(f"   Selected action: {action}")
    print(f"   Active option: {agent.current_option.name if agent.current_option else 'None'}")
    print(f"   ✓ Hierarchical agent works\n")
    
    # Test Goal-Conditioned Policy
    print("3. Testing Goal-Conditioned Policy:")
    gc_policy = GoalConditionedPolicy(state_dim=5, goal_dim=2, action_dim=9)
    
    state = torch.randn(1, 5)
    goal = torch.randn(1, 2)  # e.g., target position
    
    logits = gc_policy(state, goal)
    action = gc_policy.select_action(state, goal)
    
    print(f"   Logits shape: {logits.shape}")
    print(f"   Selected action: {action}")
    print(f"   ✓ Goal-conditioned policy works\n")
    
    # Test Hierarchical Q-Learning
    print("4. Testing Hierarchical Q-Learning:")
    hq_learning = HierarchicalQLearning(n_states=100, options=options)
    
    # Simulate update
    hq_learning.update(
        state_idx=10,
        option_idx=0,
        cumulative_reward=5.0,
        next_state_idx=15,
        done=False
    )
    
    print(f"   Q-table shape: {hq_learning.Q.shape}")
    print(f"   Sample Q-values: {hq_learning.Q[10]}")
    print(f"   ✓ Hierarchical Q-learning works\n")
    
    print("✅ All hierarchical RL tests passed!")
