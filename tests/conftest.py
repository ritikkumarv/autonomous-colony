"""
Pytest configuration and shared fixtures for The Autonomous Colony tests.

This file contains:
- Pytest configuration
- Shared fixtures available to all tests
- Test utilities and helpers
"""

import pytest
import numpy as np
import torch
from src.environment import ColonyEnvironment
from src.agents import TabularQLearningAgent, DQNAgent, PPOAgent


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )


# ============================================================================
# FIXTURES: Environment
# ============================================================================

@pytest.fixture
def simple_env():
    """
    Create a simple environment for quick testing.
    
    Returns:
        ColonyEnvironment: Small environment (10x10, 1 agent)
    """
    return ColonyEnvironment(n_agents=1, grid_size=10)


@pytest.fixture
def multi_agent_env():
    """
    Create a multi-agent environment.
    
    Returns:
        ColonyEnvironment: Medium environment (15x15, 3 agents)
    """
    return ColonyEnvironment(n_agents=3, grid_size=15)


@pytest.fixture
def standard_env():
    """
    Create a standard-sized environment (default size).
    
    Returns:
        ColonyEnvironment: Standard environment (20x20, 3 agents - default)
    """
    return ColonyEnvironment(n_agents=3, grid_size=20)


# ============================================================================
# FIXTURES: Agents
# ============================================================================

@pytest.fixture
def q_learning_agent():
    """Create a Q-learning agent for testing."""
    return TabularQLearningAgent(
        state_dim=5,
        action_dim=9,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )


@pytest.fixture
def dqn_agent():
    """
    Create a DQN agent.
    
    Returns:
        DQNAgent: Configured DQN agent
    """
    return DQNAgent(
        grid_shape=(7, 7, 5),
        state_dim=5,
        action_dim=9,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100
    )


@pytest.fixture
def ppo_agent():
    """
    Create a PPO agent.
    
    Returns:
        PPOAgent: Configured PPO agent
    """
    return PPOAgent(
        grid_shape=(7, 7, 5),
        state_dim=5,
        action_dim=9,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        n_epochs=4,
        batch_size=64
    )


# ============================================================================
# FIXTURES: Test Data
# ============================================================================

@pytest.fixture
def sample_observation():
    """
    Create a sample observation dictionary.
    
    Returns:
        dict: Valid observation with grid and state
    """
    return {
        'grid': np.random.rand(7, 7, 5).astype(np.float32),
        'state': np.array([0.5, 0.8, 0.0, 0.0, 0.0], dtype=np.float32)
    }


@pytest.fixture
def sample_transition():
    """
    Create a sample RL transition for testing.
    
    Returns:
        tuple: (obs, action, reward, next_obs, done)
    """
    obs = {
        'grid': np.random.rand(7, 7, 5).astype(np.float32),
        'state': np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    }
    next_obs = {
        'grid': np.random.rand(7, 7, 5).astype(np.float32),
        'state': np.array([0.99, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    }
    action = 1  # Up
    reward = 0.0
    done = False
    
    return obs, action, reward, next_obs, done


# ============================================================================
# FIXTURES: Random Seeds
# ============================================================================

@pytest.fixture(autouse=True)
def reset_random_seeds():
    """
    Reset random seeds before each test for reproducibility.
    
    This fixture runs automatically before every test.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_episode(env, agent, max_steps=50):
    """
    Helper function to run a single episode.
    
    Args:
        env: Environment instance
        agent: Agent instance
        max_steps: Maximum steps per episode
        
    Returns:
        tuple: (total_reward, episode_length)
    """
    observations = env.reset()
    total_reward = 0.0
    
    for step in range(max_steps):
        # Select actions for all agents
        actions = []
        for obs in observations:
            action = agent.select_action(obs, training=False)
            actions.append(action)
        
        # Execute actions
        observations, rewards, dones, truncated, info = env.step(actions)
        total_reward += sum(rewards) / len(rewards)
        
        if all(dones) or truncated:
            break
    
    return total_reward, step + 1


def assert_valid_observation(obs):
    """
    Assert that an observation has the correct structure and values.
    
    Args:
        obs: Observation dictionary to validate
        
    Raises:
        AssertionError: If observation is invalid
    """
    assert isinstance(obs, dict), "Observation must be a dictionary"
    assert 'grid' in obs, "Observation must contain 'grid' key"
    assert 'state' in obs, "Observation must contain 'state' key"
    
    # Check grid
    assert obs['grid'].shape == (7, 7, 5), f"Grid shape should be (7,7,5), got {obs['grid'].shape}"
    assert obs['grid'].dtype == np.float32, f"Grid dtype should be float32, got {obs['grid'].dtype}"
    
    # Check one-hot encoding
    grid_sum = obs['grid'].sum(axis=2)
    assert np.allclose(grid_sum, 1.0), "Grid should be one-hot encoded"
    
    # Check state
    assert obs['state'].shape == (5,), f"State shape should be (5,), got {obs['state'].shape}"
    assert obs['state'].dtype == np.float32, f"State dtype should be float32, got {obs['state'].dtype}"
    assert np.all(obs['state'] >= 0) and np.all(obs['state'] <= 1), "State should be normalized to [0, 1]"


def assert_valid_action(action, action_space=9):
    """
    Assert that an action is valid.
    
    Args:
        action: Action to validate
        action_space: Number of possible actions
        
    Raises:
        AssertionError: If action is invalid
    """
    assert isinstance(action, (int, np.integer)), f"Action must be an integer, got {type(action)}"
    assert 0 <= action < action_space, f"Action must be in [0, {action_space}), got {action}"
