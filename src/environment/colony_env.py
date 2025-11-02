# ============================================================================
# IMPROVED AUTONOMOUS COLONY ENVIRONMENT
# Applying "The Art of Clean Code" Principles
# ============================================================================

"""
Clean Code Improvements Applied:
1. Single Responsibility - Each function does ONE thing
2. Meaningful Names - Clear, descriptive naming
3. Small Functions - Keep functions short and focused
4. DRY Principle - Don't Repeat Yourself
5. Comments - Only where they add value
6. Minimal Complexity - Simplified logic flow
"""

import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Dict


# ============================================================================
# CONSTANTS - Named constants instead of magic numbers (Principle: Clean Code)
# ============================================================================

GRID_SIZE = 20
MAX_AGENTS = 3
MAX_STEPS = 500
ENERGY_DECAY_RATE = 0.1
INITIAL_ENERGY = 100.0
INITIAL_HEALTH = 100.0


class Action(IntEnum):
    """Clear, self-documenting action space"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    COLLECT = 8


class Resource(IntEnum):
    """Resource types - explicit is better than implicit"""
    EMPTY = 0
    FOOD = 1
    WATER = 2
    MATERIAL = 3
    OBSTACLE = 4


# ============================================================================
# DATA CLASSES - Simple data containers (Principle: Small is Beautiful)
# ============================================================================

@dataclass
class Position:
    """Encapsulate position logic - Single Responsibility"""
    x: int
    y: int
    
    def move(self, dx: int, dy: int, max_size: int) -> 'Position':
        """Return new position after move - Immutable pattern"""
        new_x = np.clip(self.x + dx, 0, max_size - 1)
        new_y = np.clip(self.y + dy, 0, max_size - 1)
        return Position(new_x, new_y)
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate distance - Do one thing well"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class AgentStats:
    """Agent internal state - Clear separation of concerns"""
    energy: float = INITIAL_ENERGY
    health: float = INITIAL_HEALTH
    food_count: int = 0
    water_count: int = 0
    material_count: int = 0
    
    def is_alive(self) -> bool:
        """Simple boolean check"""
        return self.energy > 0 and self.health > 0
    
    def consume_energy(self, amount: float = ENERGY_DECAY_RATE):
        """Modify energy - Clear intent"""
        self.energy = max(0, self.energy - amount)


# ============================================================================
# AGENT CLASS - Single Responsibility Principle
# ============================================================================

class Agent:
    """Represents a single agent - Does ONE thing: manage agent state"""
    
    def __init__(self, agent_id: int, position: Position):
        self.id = agent_id
        self.position = position
        self.stats = AgentStats()
        self.alive = True
    
    def update_position(self, new_position: Position):
        """Update position - Clear, focused method"""
        self.position = new_position
    
    def collect_resource(self, resource: Resource) -> float:
        """
        Collect resource and return reward.
        Single method handles ONE resource collection.
        """
        if resource == Resource.FOOD:
            self.stats.food_count += 1
            self.stats.energy = min(self.stats.energy + 30, 100)
            return 5.0
        elif resource == Resource.WATER:
            self.stats.water_count += 1
            self.stats.health = min(self.stats.health + 20, 100)
            return 3.0
        elif resource == Resource.MATERIAL:
            self.stats.material_count += 1
            return 1.0
        return 0.0
    
    def step(self) -> bool:
        """Execute one time step - Returns if agent is still alive"""
        self.stats.consume_energy()
        self.alive = self.stats.is_alive()
        return self.alive


# ============================================================================
# GRID WORLD - Manages environment state
# ============================================================================

class GridWorld:
    """
    Manages the grid and resources.
    Separated from agent logic - Single Responsibility.
    """
    
    def __init__(self, size: int = GRID_SIZE):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int32)
        self._place_obstacles(density=0.1)
    
    def _place_obstacles(self, density: float):
        """Private method for initialization - Clear naming"""
        n_obstacles = int(density * self.grid.size)
        positions = np.random.choice(self.grid.size, size=n_obstacles, replace=False)
        coords = np.unravel_index(positions, self.grid.shape)
        self.grid[coords] = Resource.OBSTACLE
    
    def spawn_resources(self, food_rate: float = 0.02):
        """Spawn resources - Simplified, focused method"""
        empty_cells = np.argwhere(self.grid == Resource.EMPTY)
        if len(empty_cells) == 0:
            return
        
        # Spawn food
        n_food = int(food_rate * len(empty_cells))
        if n_food > 0:
            indices = np.random.choice(len(empty_cells), size=n_food, replace=False)
            for idx in indices:
                y, x = empty_cells[idx]
                self.grid[y, x] = Resource.FOOD
    
    def get_resource(self, position: Position) -> Resource:
        """Get resource at position - Simple getter"""
        return Resource(self.grid[position.y, position.x])
    
    def remove_resource(self, position: Position):
        """Remove resource after collection"""
        self.grid[position.y, position.x] = Resource.EMPTY
    
    def is_valid_position(self, position: Position) -> bool:
        """Check if position is valid - Clear boolean method"""
        if not (0 <= position.x < self.size and 0 <= position.y < self.size):
            return False
        return self.grid[position.y, position.x] != Resource.OBSTACLE


# ============================================================================
# ENVIRONMENT - Orchestrates the simulation (Principle: Big Picture Thinking)
# ============================================================================

class ColonyEnvironment:
    """
    Main environment class for The Autonomous Colony.
    Acts as orchestrator - coordinates agents and world.
    
    Follows clean code principles:
    - Single responsibility
    - Clear separation of concerns
    - Focused methods
    """
    
    def __init__(self, n_agents: int = MAX_AGENTS, grid_size: int = GRID_SIZE):
        self.world = GridWorld(grid_size)
        self.agents: List[Agent] = []
        self.n_agents = n_agents
        self.step_count = 0
        
    def reset(self) -> List[Dict]:
        """Reset environment - Clear, focused reset logic"""
        self.world = GridWorld(self.world.size)
        self.agents = self._create_agents()
        self.step_count = 0
        return self._get_observations()
    
    def _create_agents(self) -> List[Agent]:
        """Private method to create agents - Separation of concerns"""
        agents = []
        for i in range(self.n_agents):
            position = self._find_empty_position()
            agents.append(Agent(i, position))
        return agents
    
    def _find_empty_position(self) -> Position:
        """Find valid starting position - Do one thing well"""
        while True:
            x = np.random.randint(0, self.world.size)
            y = np.random.randint(0, self.world.size)
            pos = Position(x, y)
            if self.world.is_valid_position(pos):
                return pos
    
    def step(self, actions: List[int]) -> Tuple:
        """
        Execute one environment step.
        Clean, readable flow without excessive nesting.
        """
        self.step_count += 1
        rewards = []
        dones = []
        
        for agent, action in zip(self.agents, actions):
            if not agent.alive:
                rewards.append(0.0)
                dones.append(True)
                continue
            
            reward = self._execute_action(agent, action)
            agent.step()  # Update agent state
            
            rewards.append(reward)
            dones.append(not agent.alive)
        
        # Spawn resources periodically
        if self.step_count % 10 == 0:
            self.world.spawn_resources()
        
        observations = self._get_observations()
        truncated = [self.step_count >= MAX_STEPS] * self.n_agents
        info = {'step': self.step_count}
        
        return observations, rewards, dones, truncated, info
    
    def _execute_action(self, agent: Agent, action: int) -> float:
        """
        Execute single action for agent.
        Separated for clarity - Single Responsibility.
        """
        if action == Action.COLLECT:
            return self._handle_collection(agent)
        else:
            return self._handle_movement(agent, action)
    
    def _handle_movement(self, agent: Agent, action: int) -> float:
        """Handle movement action - Focused method"""
        dx, dy = self._action_to_delta(action)
        new_pos = agent.position.move(dx, dy, self.world.size)
        
        if self.world.is_valid_position(new_pos):
            agent.update_position(new_pos)
            return 0.0  # No penalty for valid move
        return -0.5  # Small penalty for invalid move
    
    def _action_to_delta(self, action: int) -> Tuple[int, int]:
        """Convert action to movement delta - Simple mapping"""
        deltas = {
            Action.UP: (0, -1),
            Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0),
            Action.RIGHT: (1, 0),
            Action.UP_LEFT: (-1, -1),
            Action.UP_RIGHT: (1, -1),
            Action.DOWN_LEFT: (-1, 1),
            Action.DOWN_RIGHT: (1, 1)
        }
        return deltas.get(action, (0, 0))
    
    def _handle_collection(self, agent: Agent) -> float:
        """Handle resource collection - Clear logic"""
        resource = self.world.get_resource(agent.position)
        if resource != Resource.EMPTY and resource != Resource.OBSTACLE:
            reward = agent.collect_resource(resource)
            self.world.remove_resource(agent.position)
            return reward
        return -0.1  # Small penalty for collecting nothing
    
    def _get_observations(self) -> List[Dict]:
        """Get observations for all agents - Simple, focused"""
        return [self._get_agent_observation(agent) for agent in self.agents]
    
    def _get_local_grid(self, agent: Agent, radius: int = 3) -> np.ndarray:
        """
        Extract local grid view centered on agent position.
        Returns one-hot encoded grid of shape (2*radius+1, 2*radius+1, 5)
        
        Args:
            agent: Agent to center view on
            radius: View radius (3 = 7x7 grid)
        
        Returns:
            One-hot encoded local grid view with 5 channels
            (empty, food, water, material, obstacle)
        """
        # Extract local region with bounds checking
        y_min = max(0, agent.position.y - radius)
        y_max = min(self.world.size, agent.position.y + radius + 1)
        x_min = max(0, agent.position.x - radius)
        x_max = min(self.world.size, agent.position.x + radius + 1)
        
        local_grid = self.world.grid[y_min:y_max, x_min:x_max]
        
        # Pad if agent is near edges (fill with obstacles)
        pad_top = radius - (agent.position.y - y_min)
        pad_bottom = radius - (y_max - agent.position.y - 1)
        pad_left = radius - (agent.position.x - x_min)
        pad_right = radius - (x_max - agent.position.x - 1)
        
        local_grid = np.pad(
            local_grid, 
            ((pad_top, pad_bottom), (pad_left, pad_right)), 
            constant_values=Resource.OBSTACLE
        )
        
        # One-hot encode: 5 channels (empty, food, water, material, obstacle)
        grid_one_hot = np.eye(5)[local_grid].astype(np.float32)
        
        return grid_one_hot
    
    def _get_agent_observation(self, agent: Agent) -> Dict:
        """
        Get observation for single agent.
        Returns dict with 'grid' (7x7x5) and 'state' (5,) arrays.
        """
        # Get local grid view (7x7x5)
        grid = self._get_local_grid(agent, radius=3)
        
        # Get normalized state vector (5,)
        state = np.array([
            agent.stats.energy / 100.0,
            agent.stats.health / 100.0,
            float(agent.stats.food_count) / 10.0,
            float(agent.stats.water_count) / 10.0,
            float(agent.stats.material_count) / 10.0
        ], dtype=np.float32)
        
        return {
            'grid': grid,
            'state': state
        }


# ============================================================================
# DEMONSTRATION - Clean, simple usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CLEAN CODE PRINCIPLES IN ACTION")
    print("=" * 70)
    
    # Create environment - simple, clear API
    env = ColonyEnvironment(n_agents=3, grid_size=15)
    
    # Reset and get initial observations
    observations = env.reset()
    print(f"✓ Environment created with {len(observations)} agents")
    
    # Run simple episode
    total_rewards = [0.0] * env.n_agents
    
    for step in range(50):
        # Random actions for demo
        actions = [np.random.randint(0, 9) for _ in range(env.n_agents)]
        
        # Step environment - clean interface
        obs, rewards, dones, truncated, info = env.step(actions)
        
        # Accumulate rewards
        for i, reward in enumerate(rewards):
            total_rewards[i] += reward
        
        # Check if episode done
        if all(dones) or all(truncated):
            break
    
    print(f"\n✓ Episode complete after {step + 1} steps")
    print(f"  Total rewards: {[f'{r:.1f}' for r in total_rewards]}")
    print(f"  Agents alive: {sum(1 for agent in env.agents if agent.alive)}/{env.n_agents}")
    
    print("\n" + "=" * 70)
    print("KEY PRINCIPLES DEMONSTRATED:")
    print("=" * 70)
    print("✓ Single Responsibility - Each class has one clear purpose")
    print("✓ Meaningful Names - No ambiguous variable names")
    print("✓ Small Functions - Each method does one thing well")
    print("✓ DRY Principle - No repeated code")
    print("✓ Low Complexity - Minimal nesting, clear flow")
    print("✓ Named Constants - No magic numbers")
    print("✓ Separation of Concerns - Grid, Agent, Environment separated")
    print("=" * 70)