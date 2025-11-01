"""
Base Agent Class for The Autonomous Colony
Defines the common interface for all RL agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseAgent(ABC):
    """
    Abstract base class for all RL agents.
    
    All agents must implement:
    - select_action: Choose action given observation
    - update: Learn from experience
    """
    
    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        """
        Initialize base agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            **kwargs: Additional agent-specific parameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    @abstractmethod
    def select_action(self, observation: Dict, training: bool = True) -> int:
        """
        Select an action given an observation.
        
        Args:
            observation: Dictionary with 'grid' and 'state' keys
            training: Whether in training mode (affects exploration)
            
        Returns:
            action: Integer action index
        """
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> Optional[float]:
        """
        Update agent's policy/value function based on experience.
        
        Returns:
            loss: Training loss (if applicable), None otherwise
        """
        pass
    
    def save(self, path: str):
        """Save agent to disk"""
        raise NotImplementedError("Save method not implemented for this agent")
    
    def load(self, path: str):
        """Load agent from disk"""
        raise NotImplementedError("Load method not implemented for this agent")
