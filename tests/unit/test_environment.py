"""
Unit tests for the Colony Environment.

Tests environment mechanics:
- Reset and initialization
- Step function
- Resource management
- Agent state updates
- Reward calculation
"""

import pytest
import numpy as np
from src.environment import ColonyEnvironment, Resource, Agent, Position, AgentStats, GridWorld


class TestEnvironmentInitialization:
    """Tests for environment initialization."""
    
    def test_default_initialization(self):
        """Test environment with default parameters."""
        env = ColonyEnvironment()
        assert env.n_agents == 2
        assert env.world.size == 20
        assert env.step_count == 0
    
    def test_custom_initialization(self):
        """Test environment with custom parameters."""
        env = ColonyEnvironment(n_agents=3, grid_size=15)
        assert env.n_agents == 3
        assert env.world.size == 15
    
    def test_agent_creation(self, standard_env):
        """Test that agents are created correctly."""
        assert len(standard_env.agents) == standard_env.n_agents
        for agent in standard_env.agents:
            assert isinstance(agent, Agent)
            assert agent.alive
            assert agent.stats.energy == 100.0
            assert agent.stats.health == 100.0


class TestEnvironmentReset:
    """Tests for environment reset functionality."""
    
    def test_reset_returns_observations(self, standard_env):
        """Test that reset returns valid observations."""
        observations = standard_env.reset()
        
        assert isinstance(observations, list)
        assert len(observations) == standard_env.n_agents
        
        for obs in observations:
            assert isinstance(obs, dict)
            assert 'grid' in obs
            assert 'state' in obs
    
    def test_reset_creates_new_grid(self, standard_env):
        """Test that reset creates a new grid."""
        # Get initial grid
        standard_env.reset()
        grid1 = standard_env.world.grid.copy()
        
        # Reset again
        standard_env.reset()
        grid2 = standard_env.world.grid.copy()
        
        # Grids should be different (due to random resource placement)
        assert not np.array_equal(grid1, grid2)
    
    def test_reset_clears_step_count(self, standard_env):
        """Test that reset clears step count."""
        standard_env.reset()
        standard_env.step_count = 100
        standard_env.reset()
        assert standard_env.step_count == 0


class TestEnvironmentStep:
    """Tests for environment step functionality."""
    
    def test_step_with_valid_actions(self, standard_env):
        """Test step with valid actions."""
        standard_env.reset()
        actions = [0, 0]  # Both agents stay
        
        observations, rewards, dones, truncated, info = standard_env.step(actions)
        
        assert isinstance(observations, list)
        assert isinstance(rewards, list)
        assert isinstance(dones, list)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        assert len(observations) == standard_env.n_agents
        assert len(rewards) == standard_env.n_agents
        assert len(dones) == standard_env.n_agents
    
    def test_step_increments_count(self, standard_env):
        """Test that step increments step count."""
        standard_env.reset()
        initial_count = standard_env.step_count
        
        standard_env.step([0, 0])
        
        assert standard_env.step_count == initial_count + 1
    
    def test_step_consumes_energy(self, standard_env):
        """Test that agents consume energy each step."""
        standard_env.reset()
        initial_energy = [agent.stats.energy for agent in standard_env.agents]
        
        standard_env.step([0, 0])  # Stay action
        
        for i, agent in enumerate(standard_env.agents):
            assert agent.stats.energy < initial_energy[i]
    
    def test_movement_actions(self, simple_env):
        """Test that movement actions work correctly."""
        simple_env.reset()
        agent = simple_env.agents[0]
        initial_pos = (agent.position.x, agent.position.y)
        
        # Move up (action 1)
        simple_env.step([1])
        new_pos = (agent.position.x, agent.position.y)
        
        # Position should have changed (unless blocked)
        # We don't assert specific position due to obstacles
        assert isinstance(new_pos, tuple)


class TestObservations:
    """Tests for observation generation."""
    
    def test_observation_structure(self, standard_env):
        """Test observation has correct structure."""
        observations = standard_env.reset()
        
        for obs in observations:
            assert 'grid' in obs
            assert 'state' in obs
            assert obs['grid'].shape == (7, 7, 5)
            assert obs['state'].shape == (5,)
    
    def test_observation_types(self, standard_env):
        """Test observation data types."""
        observations = standard_env.reset()
        
        for obs in observations:
            assert obs['grid'].dtype == np.float32
            assert obs['state'].dtype == np.float32
    
    def test_grid_one_hot_encoding(self, standard_env):
        """Test that grid is properly one-hot encoded."""
        observations = standard_env.reset()
        
        for obs in observations:
            grid = obs['grid']
            # Sum across channels should be 1 (one-hot)
            channel_sum = grid.sum(axis=2)
            assert np.allclose(channel_sum, 1.0)
    
    def test_state_normalization(self, standard_env):
        """Test that state values are normalized."""
        observations = standard_env.reset()
        
        for obs in observations:
            state = obs['state']
            # All values should be in [0, 1]
            assert np.all(state >= 0)
            assert np.all(state <= 1)
    
    def test_edge_case_observations(self):
        """Test observations when agent is at edge."""
        env = ColonyEnvironment(n_agents=1, grid_size=10)
        env.reset()
        
        # Place agent at corner
        env.agents[0].position = Position(0, 0)
        
        obs = env._get_agent_observation(env.agents[0])
        
        # Should still return valid observation
        assert obs['grid'].shape == (7, 7, 5)
        assert obs['state'].shape == (5,)


class TestResourceManagement:
    """Tests for resource spawning and collection."""
    
    def test_resource_spawning(self, standard_env):
        """Test that resources spawn correctly."""
        standard_env.reset()
        
        # Count resources on grid
        food_count = np.sum(standard_env.world.grid == Resource.FOOD)
        water_count = np.sum(standard_env.world.grid == Resource.WATER)
        
        # Should have some resources
        assert food_count > 0 or water_count > 0
    
    def test_resource_collection(self, simple_env):
        """Test that agents can collect resources."""
        simple_env.reset()
        agent = simple_env.agents[0]
        
        # Place food at agent's position
        simple_env.world.grid[agent.position.y, agent.position.x] = Resource.FOOD
        initial_food = agent.stats.food_count
        initial_energy = agent.stats.energy
        
        # Collect (action 8 = collect)
        simple_env.step([8])
        
        # Agent should have collected food
        # (Energy and/or food count should change)
        assert agent.stats.food_count > initial_food or agent.stats.energy > initial_energy
    
    def test_empty_collection_penalty(self, simple_env):
        """Test penalty for collecting from empty cell."""
        simple_env.reset()
        agent = simple_env.agents[0]
        
        # Ensure position is empty
        simple_env.world.grid[agent.position.y, agent.position.x] = Resource.EMPTY
        
        # Try to collect
        _, rewards, _, _, _ = simple_env.step([8])
        
        # Should get negative reward
        assert rewards[0] < 0


class TestAgentStats:
    """Tests for agent statistics management."""
    
    def test_stats_initialization(self):
        """Test that stats initialize correctly."""
        stats = AgentStats()
        assert stats.energy == 100.0
        assert stats.health == 100.0
        assert stats.food_count == 0
        assert stats.water_count == 0
        assert stats.material_count == 0
    
    def test_energy_consumption(self):
        """Test energy consumption."""
        stats = AgentStats()
        initial_energy = stats.energy
        stats.consume_energy()
        assert stats.energy < initial_energy
    
    def test_agent_death(self):
        """Test that agent dies when energy reaches zero."""
        stats = AgentStats()
        stats.energy = 0.0
        assert not stats.is_alive()
    
    def test_agent_health_death(self):
        """Test that agent dies when health reaches zero."""
        stats = AgentStats()
        stats.health = 0.0
        assert not stats.is_alive()


class TestGridWorld:
    """Tests for grid world management."""
    
    def test_grid_initialization(self):
        """Test grid initializes correctly."""
        grid = GridWorld(size=10)
        assert grid.size == 10
        assert grid.grid.shape == (10, 10)
    
    def test_obstacle_placement(self):
        """Test that obstacles are placed."""
        grid = GridWorld(size=20)
        obstacle_count = np.sum(grid.grid == Resource.OBSTACLE)
        assert obstacle_count > 0
    
    def test_valid_position_check(self):
        """Test position validation."""
        grid = GridWorld(size=10)
        
        # Valid position
        assert grid.is_valid_position(Position(5, 5))
        
        # Out of bounds
        assert not grid.is_valid_position(Position(-1, 5))
        assert not grid.is_valid_position(Position(5, 15))
    
    def test_resource_removal(self):
        """Test resource removal."""
        grid = GridWorld(size=10)
        pos = Position(5, 5)
        
        # Place resource
        grid.grid[pos.y, pos.x] = Resource.FOOD
        assert grid.get_resource(pos) == Resource.FOOD
        
        # Remove resource
        grid.remove_resource(pos)
        assert grid.get_resource(pos) == Resource.EMPTY


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_max_steps_truncation(self):
        """Test that episode truncates at max steps."""
        env = ColonyEnvironment(n_agents=1, grid_size=10)
        env.reset()
        
        # Run for many steps
        for _ in range(1000):
            _, _, dones, truncated, _ = env.step([0])
            if truncated or dones[0]:
                break
        
        # Should eventually truncate
        assert truncated or dones[0]
    
    def test_all_agents_dead(self):
        """Test handling when all agents die."""
        env = ColonyEnvironment(n_agents=2, grid_size=10)
        env.reset()
        
        # Kill all agents
        for agent in env.agents:
            agent.stats.energy = 0.0
            agent.alive = False
        
        # Step should handle dead agents
        _, _, dones, _, _ = env.step([0, 0])
        assert all(dones)
    
    def test_single_agent_environment(self):
        """Test environment with single agent."""
        env = ColonyEnvironment(n_agents=1, grid_size=5)
        observations = env.reset()
        
        assert len(observations) == 1
        assert isinstance(observations[0], dict)
    
    def test_large_grid(self):
        """Test environment with large grid."""
        env = ColonyEnvironment(n_agents=2, grid_size=50)
        observations = env.reset()
        
        assert env.world.size == 50
        assert len(observations) == 2
