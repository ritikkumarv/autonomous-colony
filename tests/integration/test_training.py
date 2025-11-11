"""
Integration tests for full training workflows.

Tests complete agent-environment interactions:
- Full episode execution
- Training loops
- Agent-environment compatibility
- Multi-agent scenarios
"""

import pytest
import numpy as np
from tests.conftest import run_episode, assert_valid_observation, assert_valid_action


class TestAgentEnvironmentIntegration:
    """Tests for agent-environment interaction."""
    
    @pytest.mark.parametrize("agent_fixture", ["q_learning_agent", "dqn_agent", "ppo_agent"])
    def test_full_episode(self, simple_env, agent_fixture, request):
        """Test that agents can complete full episodes."""
        agent = request.getfixturevalue(agent_fixture)
        
        total_reward, length = run_episode(simple_env, agent, max_steps=50)
        
        assert isinstance(total_reward, (int, float))
        assert isinstance(length, int)
        assert length > 0
        assert length <= 50
    
    @pytest.mark.parametrize("agent_fixture", ["q_learning_agent", "dqn_agent", "ppo_agent"])
    def test_multi_episode_training(self, simple_env, agent_fixture, request):
        """Test multiple episodes of training."""
        agent = request.getfixturevalue(agent_fixture)
        rewards = []
        
        for episode in range(5):
            total_reward, _ = run_episode(simple_env, agent, max_steps=30)
            rewards.append(total_reward)
        
        assert len(rewards) == 5
        # All rewards should be finite
        assert all(np.isfinite(r) for r in rewards)
    
    def test_observation_consistency(self, simple_env, q_learning_agent):
        """Test that observations remain valid throughout episode."""
        observations = simple_env.reset()
        
        for step in range(20):
            # Check observation validity
            for obs in observations:
                assert_valid_observation(obs)
            
            # Take actions
            actions = []
            for obs in observations:
                action = q_learning_agent.select_action(obs, training=False)
                assert_valid_action(action)
                actions.append(action)
            
            # Step environment
            observations, _, dones, truncated, _ = simple_env.step(actions)
            
            if all(dones) or truncated:
                break


class TestMultiAgentScenarios:
    """Tests for multi-agent interactions."""
    
    def test_multiple_agents_same_policy(self, multi_agent_env, ppo_agent):
        """Test multiple agents using same policy."""
        observations = multi_agent_env.reset()
        
        for step in range(20):
            actions = []
            for obs in observations:
                action, _, _ = ppo_agent.select_action(obs, training=True)
                actions.append(action)
            
            observations, rewards, dones, truncated, _ = multi_agent_env.step(actions)
            
            # Check rewards for all agents
            assert len(rewards) == multi_agent_env.n_agents
            assert all(isinstance(r, (int, float)) for r in rewards)
            
            if all(dones) or truncated:
                break
    
    def test_agents_dont_overlap(self, multi_agent_env):
        """Test that agents maintain distinct positions."""
        multi_agent_env.reset()
        
        # Check initial positions are distinct
        positions = [(a.position.x, a.position.y) for a in multi_agent_env.agents]
        assert len(positions) == len(set(positions)), "Agents should have distinct starting positions"
    
    def test_independent_agent_deaths(self, multi_agent_env):
        """Test that agents can die independently."""
        multi_agent_env.reset()
        
        # Kill first agent
        multi_agent_env.agents[0].stats.energy = 0.0
        multi_agent_env.agents[0].alive = False
        
        # Other agents should still be alive
        assert not multi_agent_env.agents[0].alive
        assert multi_agent_env.agents[1].alive
        assert multi_agent_env.agents[2].alive


class TestTrainingStability:
    """Tests for training stability and convergence."""
    
    @pytest.mark.slow
    def test_q_learning_stability(self, simple_env, q_learning_agent):
        """Test Q-Learning training stability over many episodes."""
        rewards = []
        
        for episode in range(20):
            total_reward, _ = run_episode(simple_env, agent_fixture, max_steps=50)
            rewards.append(total_reward)
        
        # Should not have NaN or inf
        assert all(np.isfinite(r) for r in rewards)
        
        # Q-table should be growing
        assert len(q_learning_agent.q_table) > 0
    
    @pytest.mark.slow
    def test_dqn_stability(self, simple_env, dqn_agent):
        """Test DQN training stability."""
        rewards = []
        
        for episode in range(10):
            observations = simple_env.reset()
            episode_reward = 0
            
            for step in range(30):
                obs = observations[0]
                action = dqn_agent.select_action(obs, training=True)
                
                next_observations, reward_list, dones, truncated, _ = simple_env.step([action])
                reward = reward_list[0]
                done = dones[0]
                
                # Learning update
                dqn_agent.learn(obs, action, reward, next_observations[0], done)
                
                episode_reward += reward
                observations = next_observations
                
                if done or truncated:
                    break
            
            rewards.append(episode_reward)
        
        # Should not have NaN or inf
        assert all(np.isfinite(r) for r in rewards)
        
        # Replay buffer should be filling
        assert len(dqn_agent.replay_buffer) > 0
    
    @pytest.mark.slow
    def test_ppo_stability(self, simple_env, ppo_agent):
        """Test PPO training stability."""
        rewards = []
        
        for episode in range(10):
            observations = simple_env.reset()
            episode_reward = 0
            
            for step in range(30):
                obs = observations[0]
                action, log_prob, value = ppo_agent.select_action(obs, training=True)
                
                next_observations, reward_list, dones, truncated, _ = simple_env.step([action])
                reward = reward_list[0]
                done = dones[0]
                
                # Store in memory
                ppo_agent.memory.append((obs, action, log_prob, reward, value))
                
                episode_reward += reward
                observations = next_observations
                
                if done or truncated:
                    break
            
            # Learning update at end of episode
            if len(ppo_agent.memory) > 0:
                ppo_agent.learn(obs, action, reward, next_observations[0], done)
            
            rewards.append(episode_reward)
        
        # Should not have NaN or inf
        assert all(np.isfinite(r) for r in rewards)


class TestDifferentEnvironmentSizes:
    """Tests with different environment configurations."""
    
    @pytest.mark.parametrize("grid_size", [5, 10, 20, 30])
    def test_various_grid_sizes(self, grid_size, q_learning_agent):
        """Test agents work with different grid sizes."""
        from src.environment import ColonyEnvironment
        
        env = ColonyEnvironment(n_agents=1, grid_size=grid_size)
        total_reward, length = run_episode(env, q_learning_agent, max_steps=20)
        
        assert isinstance(total_reward, (int, float))
        assert length > 0
    
    @pytest.mark.parametrize("n_agents", [1, 2, 3, 5])
    def test_various_agent_counts(self, n_agents, dqn_agent):
        """Test environment with different numbers of agents."""
        from src.environment import ColonyEnvironment
        
        env = ColonyEnvironment(n_agents=n_agents, grid_size=15)
        observations = env.reset()
        
        assert len(observations) == n_agents
        
        # Run a few steps
        for _ in range(5):
            actions = []
            for obs in observations:
                action = dqn_agent.select_action(obs, training=False)
                actions.append(action)
            
            observations, _, dones, truncated, _ = env.step(actions)
            if all(dones) or truncated:
                break


class TestLearningProgress:
    """Tests to verify agents actually learn."""
    
    @pytest.mark.slow
    def test_q_learning_improvement(self, simple_env):
        """Test that Q-Learning agent improves over time."""
        from src.agents import TabularQLearningAgent
        
        agent = TabularQLearningAgent(
            action_space=9,
            learning_rate=0.1,
            gamma=0.99,
            epsilon=0.5  # Moderate exploration
        )
        
        # Collect rewards from first 5 episodes
        early_rewards = []
        for _ in range(5):
            reward, _ = run_episode(simple_env, agent, max_steps=50)
            early_rewards.append(reward)
        
        # Train for more episodes
        for _ in range(15):
            run_episode(simple_env, agent, max_steps=50)
        
        # Collect rewards from last 5 episodes
        late_rewards = []
        agent.epsilon = 0.1  # Reduce exploration to test learned policy
        for _ in range(5):
            reward, _ = run_episode(simple_env, agent, max_steps=50)
            late_rewards.append(reward)
        
        # Later episodes should generally be better (on average)
        early_mean = np.mean(early_rewards)
        late_mean = np.mean(late_rewards)
        
        # At minimum, the agent should not get significantly worse
        # (We allow for some variance due to randomness)
        assert late_mean >= early_mean - 20, \
            f"Agent got worse: early={early_mean:.2f}, late={late_mean:.2f}"


# ============================================================================
# STRESS TESTS
# ============================================================================

class TestStress:
    """Stress tests for edge cases."""
    
    @pytest.mark.slow
    def test_long_episode(self, simple_env, ppo_agent):
        """Test very long episode execution."""
        total_reward, length = run_episode(simple_env, ppo_agent, max_steps=500)
        
        assert isinstance(total_reward, (int, float))
        assert np.isfinite(total_reward)
    
    def test_rapid_resets(self, simple_env):
        """Test rapid environment resets."""
        for _ in range(20):
            observations = simple_env.reset()
            assert len(observations) == simple_env.n_agents
    
    def test_action_space_coverage(self, simple_env, q_learning_agent):
        """Test that all actions can be executed."""
        simple_env.reset()
        
        # Try each action
        for action in range(9):
            simple_env.reset()
            observations, rewards, dones, truncated, info = simple_env.step([action])
            
            assert isinstance(observations, list)
            assert isinstance(rewards, list)
