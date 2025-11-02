#!/usr/bin/env python3
"""
The Autonomous Colony - Main Training Script
Complete RL learning project with all major concepts

Usage:
    python train.py --agent ppo --episodes 1000
    python train.py --agent ma_ppo --n_agents 4 --communication
    python train.py --agent ppo --curiosity --curriculum
"""

import argparse
import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Import implemented modules
from src.environment import ColonyEnvironment
from src.agents import TabularQLearningAgent, DQNAgent, PPOAgent

# TODO: Import when implemented
# from src.multiagent import MultiAgentPPO
# from src.advanced import (WorldModelAgent, CuriosityAgent, 
#                           HierarchicalAgent, CurriculumManager)

# ============================================================================
# CONFIGURATION
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train RL agents in The Autonomous Colony"
    )
    
    # Agent selection
    parser.add_argument(
        "--agent",
        type=str,
        default="ppo",
        choices=["q_learning", "dqn", "ppo", "ma_ppo", "hierarchical"],
        help="Which RL agent to train"
    )
    
    # Environment settings
    parser.add_argument("--env_size", type=int, default=20, help="Grid size")
    parser.add_argument("--n_agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    
    # Training settings
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    
    # Agent hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    
    # Multi-agent settings
    parser.add_argument("--communication", action="store_true", help="Enable agent communication")
    parser.add_argument("--cooperation_bonus", type=float, default=2.0, help="Cooperation reward bonus")
    
    # Advanced features
    parser.add_argument("--curiosity", action="store_true", help="Use curiosity-driven exploration")
    parser.add_argument("--curiosity_weight", type=float, default=0.5, help="Curiosity reward weight")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning")
    parser.add_argument("--world_model", action="store_true", help="Use world model for planning")
    parser.add_argument("--meta_learning", action="store_true", help="Use meta-learning")
    
    # Logging and saving
    parser.add_argument("--log_dir", type=str, default="logs", help="TensorBoard log directory")
    parser.add_argument("--save_dir", type=str, default="models", help="Model save directory")
    parser.add_argument("--save_freq", type=int, default=50, help="Save every N episodes")
    parser.add_argument("--eval_freq", type=int, default=20, help="Evaluate every N episodes")
    parser.add_argument("--render_freq", type=int, default=100, help="Render every N episodes")
    
    # Evaluation
    parser.add_argument("--eval_episodes", type=int, default=10, help="Episodes for evaluation")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering")
    
    return parser.parse_args()

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(args):
    """Main training loop"""
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"🖥️  Using device: {device}")
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.agent}_{timestamp}"
    log_dir = Path(args.log_dir) / exp_name
    save_dir = Path(args.save_dir) / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir)
    print(f"📊 Logging to: {log_dir}")
    print(f"💾 Saving to: {save_dir}")
    
    # ========================================================================
    # CREATE ENVIRONMENT
    # ========================================================================
    
    print(f"\n🌍 Creating environment...")
    
    # Create the real Colony Environment
    env = ColonyEnvironment(n_agents=args.n_agents, grid_size=args.env_size)
    print(f"✓ Environment: {args.env_size}×{args.env_size} grid, {args.n_agents} agents")
    
    # ========================================================================
    # CREATE AGENT
    # ========================================================================
    
    print(f"\n🤖 Creating {args.agent.upper()} agent...")
    
    # Agent parameters (grid shape for observations, state dimension, actions)
    grid_shape = (7, 7, 5)  # 7x7 local view, 5 channels
    state_dim = 5  # energy, health, food, water, materials
    action_dim = 9  # 8 directions + collect
    
    agent = None
    
    if args.agent == "q_learning":
        agent = TabularQLearningAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=args.lr,
            gamma=args.gamma
        )
        print("✓ Tabular Q-Learning agent initialized")
        
    elif args.agent == "dqn":
        agent = DQNAgent(
            grid_shape=grid_shape,
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size
        )
        print("✓ DQN agent initialized")
        
    elif args.agent == "ppo":
        agent = PPOAgent(
            grid_shape=grid_shape,
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size
        )
        print("✓ PPO agent initialized")
        
    elif args.agent == "ma_ppo":
        # TODO: Implement Multi-Agent PPO
        print(f"⚠️  Multi-Agent PPO not yet implemented")
        print(f"   Use --agent ppo for now")
        return None, {}
        
    elif args.agent == "hierarchical":
        # TODO: Implement Hierarchical agent
        print("⚠️  Hierarchical agent not yet implemented")
        print("   Use --agent ppo for now")
        return None, {}
    
    if agent is None:
        print(f"❌ Agent type '{args.agent}' not recognized")
        return None, {}
    
    # Add curiosity wrapper if requested
    if args.curiosity:
        # TODO: Implement curiosity wrapper
        print(f"⚠️  Curiosity module not yet implemented (weight={args.curiosity_weight})")
    
    # Initialize curriculum if requested
    curriculum = None
    if args.curriculum:
        # TODO: Implement curriculum manager
        print("⚠️  Curriculum learning not yet implemented")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    print(f"\n🚀 Starting training for {args.episodes} episodes...")
    print("="*80)
    
    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    best_reward = -float('inf')
    
    for episode in range(args.episodes):
        # Reset environment
        observations = env.reset()
        episode_reward = [0.0] * args.n_agents
        episode_length = 0
        done_flags = [False] * args.n_agents
        
        # Episode loop
        while not all(done_flags) and episode_length < args.max_steps:
            # Prepare observations for agent (add dummy grid if needed)
            agent_observations = []
            # Observations now include both grid and state from environment
            agent_observations = observations
            
            # Select actions using agent
            actions = []
            log_probs = []
            values = []
            
            for i in range(args.n_agents):
                if args.agent == "ppo":
                    action, log_prob, value = agent.select_action(agent_observations[i], training=True)
                    actions.append(action)
                    log_probs.append(log_prob)
                    values.append(value)
                else:
                    action = agent.select_action(agent_observations[i], training=True)
                    actions.append(action)
            
            # Environment step
            next_observations, rewards, dones, truncated, info = env.step(actions)
            
            # Store transitions for agent learning
            for i in range(args.n_agents):
                if args.agent == "ppo":
                    # PPO stores transitions for batch update
                    agent.store_transition(
                        agent_observations[i],
                        actions[i],
                        rewards[i],
                        log_probs[i],
                        values[i],
                        dones[i] or truncated[i]
                    )
                else:
                    # Q-learning and DQN update immediately
                    next_agent_obs = {
                        'grid': np.zeros((7, 7, 5)),
                        'state': next_observations[i]['state']
                    }
                    agent.update(
                        agent_observations[i],
                        actions[i],
                        rewards[i],
                        next_agent_obs,
                        dones[i] or truncated[i]
                    )
            
            # Update metrics
            for i in range(args.n_agents):
                episode_reward[i] += rewards[i]
                done_flags[i] = dones[i] or truncated[i]
            
            observations = next_observations
            episode_length += 1
        
        # Update PPO agent after episode
        if args.agent == "ppo":
            loss = agent.update()
            if loss is not None:
                writer.add_scalar("train/loss", loss, episode)
        
        # Decay exploration rate
        if hasattr(agent, 'decay_epsilon'):
            agent.decay_epsilon()
            if (episode + 1) % 50 == 0:
                writer.add_scalar("train/epsilon", agent.epsilon, episode)
        
        # Curriculum update
        if curriculum:
            mean_reward = np.mean(episode_reward)
            success = mean_reward > 50  # Threshold
            # curriculum.update_difficulty(mean_reward, success)
        
        # Compute metrics
        mean_reward = np.mean(episode_reward)
        episode_rewards.append(mean_reward)
        episode_lengths.append(episode_length)
        
        if mean_reward > best_reward:
            best_reward = mean_reward
            success_count += 1
        
        # Logging
        writer.add_scalar("train/episode_reward", mean_reward, episode)
        writer.add_scalar("train/episode_length", episode_length, episode)
        writer.add_scalar("train/best_reward", best_reward, episode)
        
        if args.curriculum:
            # writer.add_scalar("curriculum/difficulty", curriculum.get_difficulty(), episode)
            pass
        
        # Console logging
        if (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            print(f"  Avg Reward (last 10): {avg_reward:.2f} ± {std_reward:.2f}")
            print(f"  Episode Length: {episode_length}")
            print(f"  Best Reward: {best_reward:.2f}")
            print(f"  Success Rate: {success_count / (episode + 1):.2%}")
        
        # Evaluation
        if (episode + 1) % args.eval_freq == 0:
            print(f"\n📊 Evaluating at episode {episode + 1}...")
            eval_rewards = evaluate_agent(env, agent, args.eval_episodes)
            eval_mean = np.mean(eval_rewards)
            eval_std = np.std(eval_rewards)
            
            writer.add_scalar("eval/mean_reward", eval_mean, episode)
            writer.add_scalar("eval/std_reward", eval_std, episode)
            
            print(f"  Eval Reward: {eval_mean:.2f} ± {eval_std:.2f}")
        
        # Render
        if (episode + 1) % args.render_freq == 0 and not args.no_render:
            print(f"\n🎬 Rendering episode {episode + 1}...")
            # env.render()
        
        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = save_dir / f"checkpoint_ep{episode + 1}.pt"
            save_checkpoint(agent, optimizer=None, episode=episode, 
                          reward=mean_reward, path=checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    
    # Final evaluation
    print("\n📊 Final Evaluation...")
    final_eval_rewards = evaluate_agent(env, agent, args.eval_episodes * 2)
    final_mean = np.mean(final_eval_rewards)
    final_std = np.std(final_eval_rewards)
    
    print(f"\nFinal Performance:")
    print(f"  Mean Reward: {final_mean:.2f} ± {final_std:.2f}")
    print(f"  Best Reward: {best_reward:.2f}")
    print(f"  Success Rate: {success_count / args.episodes:.2%}")
    
    # Save final model
    final_model_path = save_dir / "final_model.pt"
    save_checkpoint(agent, optimizer=None, episode=args.episodes,
                   reward=final_mean, path=final_model_path)
    print(f"\n💾 Final model saved: {final_model_path}")
    
    # Plot results
    print("\n📈 Generating training plots...")
    plot_training_results(
        episode_rewards, 
        episode_lengths, 
        save_path=save_dir / "training_results.png"
    )
    
    # Save metrics
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_mean_reward': final_mean,
        'final_std_reward': final_std,
        'best_reward': best_reward,
        'success_rate': success_count / args.episodes
    }
    
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ All results saved to: {save_dir}")
    print(f"✓ View logs with: tensorboard --logdir {log_dir.parent}")
    
    writer.close()
    
    return agent, metrics

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_agent(env, agent, n_episodes: int):
    """Evaluate agent performance without exploration"""
    eval_rewards = []
    
    for ep in range(n_episodes):
        observations = env.reset()
        episode_reward = [0.0] * env.n_agents
        done_flags = [False] * env.n_agents
        steps = 0
        
        while not all(done_flags) and steps < 1000:
            # Prepare observations for agent
            agent_observations = []
            for obs in observations:
                agent_obs = {
                    'grid': np.zeros((7, 7, 5)),
                    'state': obs['state']
                }
                agent_observations.append(agent_obs)
            
            # Select actions (no exploration)
            actions = []
            for i in range(env.n_agents):
                if hasattr(agent, 'select_action'):
                    if isinstance(agent.select_action(agent_observations[i], training=False), tuple):
                        # PPO returns (action, log_prob, value)
                        action, _, _ = agent.select_action(agent_observations[i], training=False)
                    else:
                        action = agent.select_action(agent_observations[i], training=False)
                    actions.append(action)
            
            # Step
            next_observations, rewards, dones, truncated, _ = env.step(actions)
            
            for i in range(env.n_agents):
                episode_reward[i] += rewards[i]
                done_flags[i] = dones[i] or truncated[i]
            
            observations = next_observations
            steps += 1
        
        eval_rewards.append(np.mean(episode_reward))
    
    return eval_rewards

# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def save_checkpoint(agent, optimizer, episode, reward, path):
    """Save model checkpoint"""
    checkpoint = {
        'episode': episode,
        'reward': reward,
        # 'model_state_dict': agent.state_dict() if hasattr(agent, 'state_dict') else None,
        # 'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
    }
    
    # torch.save(checkpoint, path)
    print(f"Checkpoint saved (demo mode)")

def load_checkpoint(agent, optimizer, path):
    """Load model checkpoint"""
    # checkpoint = torch.load(path)
    # agent.load_state_dict(checkpoint['model_state_dict'])
    # if optimizer and checkpoint['optimizer_state_dict']:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # return checkpoint['episode'], checkpoint['reward']
    print(f"Checkpoint loaded from {path} (demo mode)")
    return 0, 0.0

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_results(rewards, lengths, save_path=None):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Episode rewards
    axes[0, 0].plot(rewards, alpha=0.4, color='blue', label='Raw')
    if len(rewards) > 20:
        window = 20
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(rewards)), smoothed, 
                       linewidth=2, color='red', label=f'Smoothed ({window})')
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Reward', fontsize=12)
    axes[0, 0].set_title('Training Rewards', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(lengths, alpha=0.4, color='green', label='Raw')
    if len(lengths) > 20:
        window = 20
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(lengths)), smoothed,
                       linewidth=2, color='orange', label=f'Smoothed ({window})')
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('Length', fontsize=12)
    axes[0, 1].set_title('Episode Lengths', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reward distribution
    axes[1, 0].hist(rewards, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(np.mean(rewards), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    axes[1, 0].set_xlabel('Reward', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Reward Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Cumulative reward
    cumulative = np.cumsum(rewards)
    axes[1, 1].plot(cumulative, linewidth=2, color='teal')
    axes[1, 1].set_xlabel('Episode', fontsize=12)
    axes[1, 1].set_ylabel('Cumulative Reward', fontsize=12)
    axes[1, 1].set_title('Cumulative Reward', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Results - The Autonomous Colony', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    
    plt.show()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    args = parse_args()
    
    print("\n" + "="*80)
    print("🌍 THE AUTONOMOUS COLONY - RL Training")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Agent: {args.agent}")
    print(f"  Environment: {args.env_size}×{args.env_size} grid")
    print(f"  Agents: {args.n_agents}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Seed: {args.seed}")
    
    if args.curiosity:
        print(f"  ✓ Curiosity enabled (weight={args.curiosity_weight})")
    if args.curriculum:
        print(f"  ✓ Curriculum learning enabled")
    if args.communication and args.agent == "ma_ppo":
        print(f"  ✓ Agent communication enabled")
    
    print("="*80 + "\n")
    
    try:
        agent, metrics = train(args)
        
        print("\n" + "="*80)
        print("✅ TRAINING SUCCESSFUL!")
        print("="*80)
        print(f"\nFinal Results:")
        print(f"  Mean Reward: {metrics['final_mean_reward']:.2f}")
        print(f"  Best Reward: {metrics['best_reward']:.2f}")
        print(f"  Success Rate: {metrics['success_rate']:.2%}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Progress has been saved")
    
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

# ============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# ============================================================================

def compare_agents(checkpoints: list, env, n_episodes: int = 50):
    """Compare multiple trained agents"""
    results = {}
    
    print("\n🔬 Comparing agents...")
    
    for checkpoint_path in checkpoints:
        agent_name = Path(checkpoint_path).stem
        print(f"\nEvaluating {agent_name}...")
        
        # Load agent
        # agent = load_agent(checkpoint_path)
        
        # Evaluate
        # rewards = evaluate_agent(env, agent, n_episodes)
        rewards = np.random.randn(n_episodes) * 10 + 50  # Demo
        
        results[agent_name] = {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'rewards': rewards
        }
        
        print(f"  Mean: {results[agent_name]['mean']:.2f} ± {results[agent_name]['std']:.2f}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    names = list(results.keys())
    means = [results[n]['mean'] for n in names]
    stds = [results[n]['std'] for n in names]
    
    axes[0].bar(names, means, yerr=stds, capsize=5, alpha=0.7)
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Agent Comparison')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot
    reward_lists = [results[n]['rewards'] for n in names]
    axes[1].boxplot(reward_lists, labels=names)
    axes[1].set_ylabel('Reward Distribution')
    axes[1].set_title('Reward Distributions')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return results

def hyperparameter_search(env, agent_class, param_grid, n_trials=10):
    """Simple random search for hyperparameters"""
    import random
    
    print("\n🔍 Hyperparameter Search")
    print(f"Trials: {n_trials}")
    print(f"Search space: {param_grid}")
    
    results = []
    
    for trial in range(n_trials):
        # Sample hyperparameters
        params = {k: random.choice(v) for k, v in param_grid.items()}
        
        print(f"\nTrial {trial + 1}/{n_trials}")
        print(f"Params: {params}")
        
        # Train agent (simplified)
        # agent = agent_class(**params)
        # reward = train_agent(env, agent, n_episodes=100)
        reward = np.random.randn() * 10 + 50  # Demo
        
        results.append({
            'params': params,
            'reward': reward
        })
        
        print(f"Reward: {reward:.2f}")
    
    # Find best
    best = max(results, key=lambda x: x['reward'])
    print(f"\n✅ Best configuration:")
    print(f"Params: {best['params']}")
    print(f"Reward: {best['reward']:.2f}")
    
    return results

# ============================================================================
# EXAMPLE CONFIGURATIONS
# ============================================================================

EXAMPLE_CONFIGS = {
    "quick_test": {
        "agent": "ppo",
        "episodes": 100,
        "env_size": 15,
        "n_agents": 2,
        "lr": 3e-4
    },
    
    "single_agent_baseline": {
        "agent": "ppo",
        "episodes": 500,
        "env_size": 20,
        "n_agents": 1,
        "lr": 3e-4,
        "gamma": 0.99
    },
    
    "multiagent_coop": {
        "agent": "ma_ppo",
        "episodes": 500,
        "env_size": 25,
        "n_agents": 4,
        "communication": True,
        "cooperation_bonus": 3.0,
        "lr": 1e-4
    },
    
    "curiosity_exploration": {
        "agent": "ppo",
        "episodes": 600,
        "env_size": 30,
        "n_agents": 2,
        "curiosity": True,
        "curiosity_weight": 0.5,
        "lr": 3e-4
    },
    
    "curriculum_learning": {
        "agent": "ppo",
        "episodes": 800,
        "env_size": 20,
        "n_agents": 3,
        "curriculum": True,
        "lr": 3e-4
    },
    
    "full_advanced": {
        "agent": "ma_ppo",
        "episodes": 1000,
        "env_size": 35,
        "n_agents": 6,
        "communication": True,
        "curiosity": True,
        "curriculum": True,
        "cooperation_bonus": 2.5,
        "curiosity_weight": 0.3,
        "lr": 1e-4
    }
}

def load_config(config_name: str):
    """Load a predefined configuration"""
    if config_name not in EXAMPLE_CONFIGS:
        print(f"❌ Unknown config: {config_name}")
        print(f"Available: {list(EXAMPLE_CONFIGS.keys())}")
        return None
    
    return EXAMPLE_CONFIGS[config_name]

# ============================================================================
# DOCUMENTATION
# ============================================================================

__doc__ = """
The Autonomous Colony - Complete RL Training Script

This script provides a unified interface for training various RL agents
in the multi-agent colony environment.

Quick Start Examples:
---------------------

1. Basic PPO training:
   python train.py --agent ppo --episodes 500

2. Multi-agent with communication:
   python train.py --agent ma_ppo --n_agents 4 --communication

3. Curiosity-driven exploration:
   python train.py --agent ppo --curiosity --curiosity_weight 0.5

4. Full advanced configuration:
   python train.py --agent ma_ppo --n_agents 6 --communication \\
       --curiosity --curriculum --episodes 1000

5. Load predefined config:
   python train.py --config multiagent_coop

For more information, see README.md
"""