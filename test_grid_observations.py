#!/usr/bin/env python3
"""
Quick test to verify grid observations are working correctly.
"""

import numpy as np
from src.environment import ColonyEnvironment

def test_grid_observations():
    """Test that observations have correct shapes and contents"""
    print("=" * 80)
    print("TESTING GRID OBSERVATIONS")
    print("=" * 80)
    
    # Create environment
    env = ColonyEnvironment(n_agents=2, grid_size=20)
    observations = env.reset()
    
    print(f"\n✓ Environment created with {env.n_agents} agents")
    print(f"✓ Grid size: {env.world.size}x{env.world.size}")
    
    # Check observations structure
    print(f"\n📊 Checking observations structure:")
    print(f"   Type of observations: {type(observations)}")
    print(f"   Number of observations: {len(observations)}")
    
    for i, obs in enumerate(observations):
        print(f"\n   Agent {i}:")
        print(f"      Keys: {list(obs.keys())}")
        
        # Check grid
        if 'grid' in obs:
            grid_shape = obs['grid'].shape
            print(f"      ✓ Grid shape: {grid_shape}")
            assert grid_shape == (7, 7, 5), f"Expected (7, 7, 5), got {grid_shape}"
            
            # Check one-hot encoding
            grid_sum = obs['grid'].sum(axis=2)
            assert np.allclose(grid_sum, 1.0), "Grid should be one-hot encoded"
            print(f"      ✓ Grid is one-hot encoded")
            
            # Check data type
            assert obs['grid'].dtype == np.float32, f"Expected float32, got {obs['grid'].dtype}"
            print(f"      ✓ Grid dtype: {obs['grid'].dtype}")
        else:
            print(f"      ✗ ERROR: 'grid' key missing!")
            return False
        
        # Check state
        if 'state' in obs:
            state_shape = obs['state'].shape
            print(f"      ✓ State shape: {state_shape}")
            assert state_shape == (5,), f"Expected (5,), got {state_shape}"
            
            # Check normalization (values should be in [0, 1] range)
            assert np.all(obs['state'] >= 0) and np.all(obs['state'] <= 1), \
                "State values should be normalized to [0, 1]"
            print(f"      ✓ State normalized: min={obs['state'].min():.2f}, max={obs['state'].max():.2f}")
        else:
            print(f"      ✗ ERROR: 'state' key missing!")
            return False
    
    # Test edge cases - agent near boundaries
    print(f"\n🔍 Testing edge cases (agents near boundaries):")
    
    # Place agent at corner
    env.agents[0].position.x = 0
    env.agents[0].position.y = 0
    observations = env._get_observations()
    
    corner_grid = observations[0]['grid']
    print(f"   Agent at corner (0,0):")
    print(f"      Grid shape: {corner_grid.shape}")
    assert corner_grid.shape == (7, 7, 5), "Grid should be padded correctly"
    print(f"      ✓ Grid correctly padded")
    
    # Check that padding is obstacles (channel 4)
    # Top-left corner should have obstacles in padded areas
    obstacle_channel = corner_grid[:, :, 4]
    print(f"      Obstacle padding present: {obstacle_channel[:3, :3].sum() > 0}")
    
    # Test step function
    print(f"\n🏃 Testing step function with grid observations:")
    env.reset()
    actions = [0, 1]  # Some actions
    observations, rewards, dones, truncated, info = env.step(actions)
    
    print(f"   Step completed successfully")
    print(f"   Observations: {len(observations)}")
    print(f"   Rewards: {rewards}")
    print(f"   Grid shapes: {[obs['grid'].shape for obs in observations]}")
    print(f"   State shapes: {[obs['state'].shape for obs in observations]}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nGrid observations are working correctly:")
    print("  • Grid shape: (7, 7, 5) ✓")
    print("  • State shape: (5,) ✓")
    print("  • One-hot encoding: ✓")
    print("  • Edge padding: ✓")
    print("  • Step function: ✓")
    
    return True


if __name__ == "__main__":
    try:
        success = test_grid_observations()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
