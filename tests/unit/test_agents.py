"""
Unit tests for RL agents.

Tests individual agent methods and behaviors:
- Action selection
- Learning updates
- State management
- Hyperparameter handling
"""

import pytest
import numpy as np
import torch
from src.agents import TabularQLearningAgent, DQNAgent, PPOAgent


class TestTabularQLearning:
    """Tests for Tabular Q-Learning agent."""
    
    def test_initialization(self, q_learning_agent):
        """Test that Q-Learning agent initializes correctly."""
        assert q_learning_agent.action_space == 9
        assert q_learning_agent.learning_rate == 0.1
        assert q_learning_agent.gamma == 0.99
        assert q_learning_agent.epsilon == 1.0
        assert len(q_learning_agent.q_table) == 0  # Empty at start
    
    def test_action_selection(self, q_learning_agent, sample_observation):
        """Test action selection returns valid actions."""
        action = q_learning_agent.select_action(sample_observation, training=True)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 9
    
    def test_epsilon_decay(self, q_learning_agent, sample_observation):
        """Test that epsilon decays over time."""
        initial_epsilon = q_learning_agent.epsilon
        
        # Run multiple action selections
        for _ in range(100):
            q_learning_agent.select_action(sample_observation, training=True)
        
        # Epsilon should have decayed
        assert q_learning_agent.epsilon < initial_epsilon
        # But not below min_epsilon
        assert q_learning_agent.epsilon >= 0.01
    
    def test_learning_update(self, q_learning_agent, sample_transition):
        """Test that learning updates the Q-table."""
        obs, action, reward, next_obs, done = sample_transition
        
        # Q-table should be empty initially
        assert len(q_learning_agent.q_table) == 0
        
        # Perform learning update
        q_learning_agent.learn(obs, action, reward, next_obs, done)
        
        # Q-table should now have an entry
        assert len(q_learning_agent.q_table) > 0
    
    def test_state_discretization(self, q_learning_agent):
        """Test state discretization is consistent."""
        state1 = np.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        state2 = np.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        
        obs1 = {'grid': np.zeros((7, 7, 5)), 'state': state1}
        obs2 = {'grid': np.zeros((7, 7, 5)), 'state': state2}
        
        # Same state should give same discretization
        key1 = q_learning_agent._discretize_state(obs1)
        key2 = q_learning_agent._discretize_state(obs2)
        assert key1 == key2


class TestDQN:
    """Tests for Deep Q-Network agent."""
    
    def test_initialization(self, dqn_agent):
        """Test that DQN agent initializes correctly."""
        assert dqn_agent.action_space == 9
        assert dqn_agent.learning_rate == 0.001
        assert dqn_agent.gamma == 0.99
        assert dqn_agent.epsilon == 1.0
        assert dqn_agent.q_network is not None
        assert dqn_agent.target_network is not None
        assert len(dqn_agent.replay_buffer) == 0
    
    def test_action_selection(self, dqn_agent, sample_observation):
        """Test action selection returns valid actions."""
        action = dqn_agent.select_action(sample_observation, training=True)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 9
    
    def test_replay_buffer(self, dqn_agent, sample_transition):
        """Test replay buffer stores transitions."""
        obs, action, reward, next_obs, done = sample_transition
        
        # Buffer should be empty initially
        assert len(dqn_agent.replay_buffer) == 0
        
        # Store transition
        dqn_agent.replay_buffer.push(obs, action, reward, next_obs, done)
        
        # Buffer should have one transition
        assert len(dqn_agent.replay_buffer) == 1
        
        # Can sample from buffer
        if len(dqn_agent.replay_buffer) >= dqn_agent.batch_size:
            batch = dqn_agent.replay_buffer.sample(dqn_agent.batch_size)
            assert len(batch) == dqn_agent.batch_size
    
    def test_learning_with_insufficient_data(self, dqn_agent, sample_transition):
        """Test that learning requires sufficient buffer data."""
        obs, action, reward, next_obs, done = sample_transition
        
        # Add single transition (not enough for batch)
        dqn_agent.replay_buffer.push(obs, action, reward, next_obs, done)
        
        # Learning should not crash with insufficient data
        dqn_agent.learn(obs, action, reward, next_obs, done)
    
    def test_target_network_update(self, dqn_agent):
        """Test that target network can be updated."""
        # Get initial target network params
        initial_params = [p.clone() for p in dqn_agent.target_network.parameters()]
        
        # Update target network
        dqn_agent._update_target_network()
        
        # Params should now match q_network
        for target_param, q_param in zip(
            dqn_agent.target_network.parameters(),
            dqn_agent.q_network.parameters()
        ):
            assert torch.allclose(target_param, q_param)
    
    def test_network_forward_pass(self, dqn_agent, sample_observation):
        """Test that neural network forward pass works."""
        grid = torch.FloatTensor(sample_observation['grid']).unsqueeze(0)
        state = torch.FloatTensor(sample_observation['state']).unsqueeze(0)
        
        with torch.no_grad():
            q_values = dqn_agent.q_network(grid, state)
        
        assert q_values.shape == (1, 9)  # Batch size 1, 9 actions


class TestPPO:
    """Tests for Proximal Policy Optimization agent."""
    
    def test_initialization(self, ppo_agent):
        """Test that PPO agent initializes correctly."""
        assert ppo_agent.action_space == 9
        assert ppo_agent.learning_rate == 0.0003
        assert ppo_agent.gamma == 0.99
        assert ppo_agent.actor_critic is not None
        assert len(ppo_agent.memory) == 0
    
    def test_action_selection(self, ppo_agent, sample_observation):
        """Test action selection returns action, log_prob, and value."""
        result = ppo_agent.select_action(sample_observation, training=True)
        
        assert len(result) == 3
        action, log_prob, value = result
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 9
        assert isinstance(log_prob, (float, np.floating, torch.Tensor))
        assert isinstance(value, (float, np.floating, torch.Tensor))
    
    def test_memory_storage(self, ppo_agent, sample_observation):
        """Test that memory stores transitions correctly."""
        action, log_prob, value = ppo_agent.select_action(sample_observation, training=True)
        
        # Store transition
        ppo_agent.memory.append((sample_observation, action, log_prob, 1.0, value))
        
        assert len(ppo_agent.memory) == 1
    
    def test_learning_with_trajectory(self, ppo_agent, sample_observation):
        """Test learning with a short trajectory."""
        # Collect short trajectory
        for _ in range(5):
            action, log_prob, value = ppo_agent.select_action(sample_observation, training=True)
            ppo_agent.memory.append((sample_observation, action, log_prob, 1.0, value))
        
        # Learning should process the trajectory
        ppo_agent.learn(sample_observation, 0, 1.0, sample_observation, False)
        
        # Memory should be cleared after learning
        assert len(ppo_agent.memory) == 0
    
    def test_gae_computation(self, ppo_agent):
        """Test Generalized Advantage Estimation computation."""
        # Create simple rewards and values
        rewards = [1.0, 0.0, 1.0, 0.0]
        values = torch.tensor([0.5, 0.6, 0.7, 0.8])
        next_value = torch.tensor(0.9)
        done = False
        
        # Compute GAE
        advantages = ppo_agent._compute_gae(rewards, values, next_value, done)
        
        assert len(advantages) == len(rewards)
        assert isinstance(advantages, torch.Tensor)
    
    def test_actor_critic_forward_pass(self, ppo_agent, sample_observation):
        """Test that actor-critic network forward pass works."""
        grid = torch.FloatTensor(sample_observation['grid']).unsqueeze(0)
        state = torch.FloatTensor(sample_observation['state']).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, value = ppo_agent.actor_critic(grid, state)
        
        assert action_probs.shape == (1, 9)  # Batch size 1, 9 actions
        assert value.shape == (1, 1)  # Batch size 1, single value
        assert torch.allclose(action_probs.sum(dim=1), torch.ones(1))  # Probs sum to 1


# ============================================================================
# PARAMETRIZED TESTS - Test all agents with same scenarios
# ============================================================================

@pytest.mark.parametrize("agent_fixture", ["q_learning_agent", "dqn_agent", "ppo_agent"])
def test_agent_episode(agent_fixture, sample_observation, request):
    """Test that all agents can run a short episode."""
    agent = request.getfixturevalue(agent_fixture)
    
    for _ in range(10):
        if agent_fixture == "ppo_agent":
            action, log_prob, value = agent.select_action(sample_observation, training=True)
        else:
            action = agent.select_action(sample_observation, training=True)
        
        assert 0 <= action < 9


@pytest.mark.parametrize("agent_fixture", ["q_learning_agent", "dqn_agent", "ppo_agent"])
def test_agent_training_mode(agent_fixture, sample_observation, request):
    """Test that all agents respect training mode."""
    agent = request.getfixturevalue(agent_fixture)
    
    # Training mode should allow exploration
    if agent_fixture == "ppo_agent":
        action1, _, _ = agent.select_action(sample_observation, training=True)
    else:
        action1 = agent.select_action(sample_observation, training=True)
    
    # Eval mode (if agent is Q-Learning or DQN, set epsilon to 0)
    if hasattr(agent, 'epsilon'):
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0
    
    if agent_fixture == "ppo_agent":
        action2, _, _ = agent.select_action(sample_observation, training=False)
    else:
        action2 = agent.select_action(sample_observation, training=False)
    
    # Restore epsilon
    if hasattr(agent, 'epsilon'):
        agent.epsilon = old_epsilon
    
    # Both actions should be valid
    assert 0 <= action1 < 9
    assert 0 <= action2 < 9
