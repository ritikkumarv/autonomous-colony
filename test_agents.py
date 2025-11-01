#!/usr/bin/env python3
"""
Test script to verify agent implementations
Run this to ensure all agents are working correctly
"""

import numpy as np
from src.environment import ColonyEnvironment
from src.agents import TabularQLearningAgent, DQNAgent, PPOAgent


def test_agent(agent_class, agent_name, **agent_kwargs):
    """Test a single agent"""
    print(f"\n{'='*70}")
    print(f"Testing {agent_name}")
    print('='*70)
    
    # Create environment
    env = ColonyEnvironment(n_agents=1, grid_size=15)
    observations = env.reset()
    
    # Create agent
    agent = agent_class(**agent_kwargs)
    
    # Run episode
    episode_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 50:
        obs = observations[0]
        
        # Create observation in expected format
        agent_obs = {
            'grid': np.zeros((7, 7, 5)),  # Simplified grid
            'state': obs['state']
        }
        
        # Select action
        if agent_name == "PPO":
            action, log_prob, value = agent.select_action(agent_obs, training=True)
        else:
            action = agent.select_action(agent_obs, training=True)
        
        # Step environment
        next_observations, rewards, dones, truncated, info = env.step([action])
        
        episode_reward += rewards[0]
        observations = next_observations
        done = dones[0] or truncated[0]
        steps += 1
        
        # Update agent
        if agent_name != "PPO":  # PPO updates in batches
            agent.update(agent_obs, action, rewards[0], 
                        {'grid': np.zeros((7, 7, 5)), 'state': next_observations[0]['state']},
                        done)
    
    print(f"✅ {agent_name} completed {steps} steps")
    print(f"   Total reward: {episode_reward:.2f}")
    
    # Decay epsilon if applicable
    if hasattr(agent, 'decay_epsilon'):
        agent.decay_epsilon()
        print(f"   Epsilon after decay: {agent.epsilon:.4f}")
    
    return True


def main():
    """Run all agent tests"""
    print("\n" + "="*70)
    print("AGENT IMPLEMENTATION TEST SUITE")
    print("="*70)
    
    results = []
    
    # Test Tabular Q-Learning
    try:
        test_agent(
            TabularQLearningAgent,
            "Tabular Q-Learning",
            state_dim=5,
            action_dim=9,
            epsilon=0.3  # Lower epsilon for testing
        )
        results.append(("Tabular Q-Learning", True))
    except Exception as e:
        print(f"❌ Tabular Q-Learning failed: {e}")
        results.append(("Tabular Q-Learning", False))
    
    # Test DQN
    try:
        test_agent(
            DQNAgent,
            "DQN",
            grid_shape=(7, 7, 5),
            state_dim=5,
            action_dim=9,
            epsilon=0.3
        )
        results.append(("DQN", True))
    except Exception as e:
        print(f"❌ DQN failed: {e}")
        results.append(("DQN", False))
    
    # Test PPO
    try:
        test_agent(
            PPOAgent,
            "PPO",
            grid_shape=(7, 7, 5),
            state_dim=5,
            action_dim=9
        )
        results.append(("PPO", True))
    except Exception as e:
        print(f"❌ PPO failed: {e}")
        results.append(("PPO", False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:25s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  Some tests failed")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
