"""
Resource Management

Handles spawning and management of resources in the colony environment.
"""

import numpy as np
from typing import Tuple, List

class ResourceSpawner:
    """
    Manages resource spawning in the environment.
    
    Args:
        grid_size: Size of the grid
        food_rate: Food spawn rate per step
        water_rate: Water spawn rate per step
        material_rate: Material spawn rate per step
    """
    
    def __init__(
        self,
        grid_size: int = 20,
        food_rate: float = 0.02,
        water_rate: float = 0.02,
        material_rate: float = 0.01
    ):
        self.grid_size = grid_size
        self.food_rate = food_rate
        self.water_rate = water_rate
        self.material_rate = material_rate
    
    def spawn_resources(self, grid: np.ndarray) -> np.ndarray:
        """
        Spawn new resources on the grid.
        
        Args:
            grid: Current grid state
            
        Returns:
            Updated grid with new resources
        """
        # Food spawning
        if np.random.random() < self.food_rate:
            x, y = np.random.randint(0, self.grid_size, 2)
            if grid[x, y, 1] == 0:  # Empty cell (channel 1 = food)
                grid[x, y, 1] = 1
        
        # Water spawning
        if np.random.random() < self.water_rate:
            x, y = np.random.randint(0, self.grid_size, 2)
            if grid[x, y, 2] == 0:  # Empty cell (channel 2 = water)
                grid[x, y, 2] = 1
        
        # Material spawning
        if np.random.random() < self.material_rate:
            x, y = np.random.randint(0, self.grid_size, 2)
            if grid[x, y, 3] == 0:  # Empty cell (channel 3 = materials)
                grid[x, y, 3] = 1
        
        return grid
