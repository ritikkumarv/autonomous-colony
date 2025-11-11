#!/usr/bin/env python3
"""
Visualization Script for The Autonomous Colony

Visualizes trained agents in action with rich analytics:
- Episode playback with grid rendering
- Agent trajectories and heatmaps
- Performance metrics
- Multi-episode analysis

Usage:
    python visualize.py --model models/ppo_final.pt --episodes 5
    python visualize.py --model models/mappo_final.pt --heatmap --trajectory
    python visualize.py --compare models/agent1.pt models/agent2.pt
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
import json

from src.environment import ColonyEnvironment
from src.agents import PPOAgent, DQNAgent, TabularQLearningAgent
from src.environment.rendering import (
    GridRenderer,
    HeatmapVisualizer,
    TrajectoryVisualizer,
    TrainingDashboard,
    CommunicationVisualizer
)


class VisualizationSession:
    """
    Manages a visualization session for trained agents.
    """
    
    def __init__(
        self,
        agent,
        env: ColonyEnvironment,
        output_dir: str = "visualizations"
    ):
        self.agent = agent
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Visualizers
        self.grid_renderer = GridRenderer(grid_size=env.world.size)
        self.heatmap_viz = HeatmapVisualizer(grid_size=env.world.size)
        self.trajectory_viz = TrajectoryVisualizer(
            n_agents=env.n_agents,
            grid_size=env.world.size
        )
        self.dashboard = TrainingDashboard()
        
        # Metrics
        self.episode_data = []
    
    def run_episode(
        self,
        episode_num: int,
        render: bool = True,
        save_frames: bool = False,
        max_steps: int = 500
    ) -> Dict:
        """
        Run a single episode with visualization.
        
        Args:
            episode_num: Episode number
            render: Whether to render in real-time
            save_frames: Whether to save individual frames
            max_steps: Maximum steps per episode
            
        Returns:
            metrics: Episode metrics
        """
        observations = self.env.reset()
        done = False
        step = 0
        total_reward = 0
        agent_rewards = [0.0] * self.env.n_agents
        
        collected_positions = []
        
        while not done and step < max_steps:
            # Select actions
            if hasattr(self.agent, 'select_action'):
                # Single agent controlling all
                actions = []
                for obs in observations:
                    action, _, _ = self.agent.select_action(obs, training=False)
                    actions.append(action)
            else:
                # Simple agent
                actions = [self.agent.act(obs) for obs in observations]
            
            # Step environment
            next_observations, rewards, dones, truncated, info = self.env.step(actions)
            
            # Track metrics
            total_reward += sum(rewards)
            for i, r in enumerate(rewards):
                agent_rewards[i] += r
            
            # Track positions for heatmap
            positions = [(agent.position.x, agent.position.y) for agent in self.env.agents]
            self.heatmap_viz.update(positions)
            self.trajectory_viz.add_positions(positions)
            
            # Track collections
            if info and 'collected' in info:
                collected_positions.extend(info['collected'])
            
            # Render
            if render:
                self.grid_renderer.render(
                    self.env.world.grid,
                    self.env.agents,
                    step=step,
                    save_path=f"{self.output_dir}/ep{episode_num}_step{step:04d}.png" if save_frames else None
                )
            
            observations = next_observations
            done = truncated or all(dones)
            step += 1
        
        # Episode metrics
        metrics = {
            'episode': episode_num,
            'steps': step,
            'total_reward': total_reward,
            'avg_reward': total_reward / self.env.n_agents,
            'agent_rewards': agent_rewards,
            'success': total_reward > 0  # Simple success criterion
        }
        
        self.episode_data.append(metrics)
        self.dashboard.update({
            'episode_reward': total_reward,
            'episode_length': step,
            'success': metrics['success']
        })
        
        return metrics
    
    def run_multiple_episodes(
        self,
        num_episodes: int = 10,
        render: bool = True,
        save_frames: bool = False
    ):
        """Run multiple episodes and collect statistics."""
        print(f"\n{'='*80}")
        print(f"Running {num_episodes} Episodes")
        print(f"{'='*80}\n")
        
        for ep in range(num_episodes):
            print(f"Episode {ep + 1}/{num_episodes}...")
            metrics = self.run_episode(
                episode_num=ep,
                render=render,
                save_frames=save_frames
            )
            
            print(f"  Steps: {metrics['steps']}")
            print(f"  Total Reward: {metrics['total_reward']:.2f}")
            print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
            print()
        
        self.grid_renderer.close()
    
    def save_visualizations(self):
        """Save all visualization outputs."""
        print(f"\n{'='*80}")
        print("Saving Visualizations")
        print(f"{'='*80}\n")
        
        # Save heatmaps
        heatmap_path = self.output_dir / "heatmaps.png"
        self.heatmap_viz.render(save_path=str(heatmap_path))
        print(f"✓ Saved heatmaps to {heatmap_path}")
        
        # Save trajectories
        trajectory_path = self.output_dir / "trajectories.png"
        self.trajectory_viz.render(save_path=str(trajectory_path))
        print(f"✓ Saved trajectories to {trajectory_path}")
        
        # Save dashboard
        dashboard_path = self.output_dir / "dashboard.png"
        self.dashboard.render(save_path=str(dashboard_path))
        print(f"✓ Saved dashboard to {dashboard_path}")
        
        # Save metrics as JSON
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.episode_data, f, indent=2)
        print(f"✓ Saved metrics to {metrics_path}")
        
        # Save summary statistics
        self._save_summary()
    
    def _save_summary(self):
        """Save summary statistics."""
        if not self.episode_data:
            return
        
        summary_path = self.output_dir / "summary.txt"
        
        rewards = [ep['total_reward'] for ep in self.episode_data]
        steps = [ep['steps'] for ep in self.episode_data]
        successes = [ep['success'] for ep in self.episode_data]
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("VISUALIZATION SESSION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Episodes: {len(self.episode_data)}\n\n")
            
            f.write("Rewards:\n")
            f.write(f"  Mean: {np.mean(rewards):.2f}\n")
            f.write(f"  Std:  {np.std(rewards):.2f}\n")
            f.write(f"  Min:  {np.min(rewards):.2f}\n")
            f.write(f"  Max:  {np.max(rewards):.2f}\n\n")
            
            f.write("Episode Lengths:\n")
            f.write(f"  Mean: {np.mean(steps):.0f}\n")
            f.write(f"  Std:  {np.std(steps):.0f}\n")
            f.write(f"  Min:  {np.min(steps):.0f}\n")
            f.write(f"  Max:  {np.max(steps):.0f}\n\n")
            
            f.write(f"Success Rate: {np.mean(successes):.1%}\n\n")
            
            f.write("Per-Episode Details:\n")
            for ep in self.episode_data:
                f.write(f"  Episode {ep['episode']}: "
                       f"Reward={ep['total_reward']:.2f}, "
                       f"Steps={ep['steps']}, "
                       f"Success={'✓' if ep['success'] else '✗'}\n")
        
        print(f"✓ Saved summary to {summary_path}")


def load_agent(model_path: str, env: ColonyEnvironment, device: str = "cpu"):
    """
    Load a trained agent from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        env: Environment instance
        device: Device to load model on
        
    Returns:
        agent: Loaded agent
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Detect agent type from checkpoint or filename
    if 'ppo' in model_path.lower():
        agent = PPOAgent(
            grid_shape=(7, 7, 5),
            state_dim=5,
            action_dim=9
        )
        agent.network.load_state_dict(checkpoint['network_state_dict'])
        print(f"✓ Loaded PPO agent from {model_path}")
    
    elif 'dqn' in model_path.lower():
        agent = DQNAgent(
            grid_shape=(7, 7, 5),
            state_dim=5,
            action_dim=9
        )
        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        print(f"✓ Loaded DQN agent from {model_path}")
    
    elif 'q_learning' in model_path.lower():
        agent = TabularQLearningAgent(
            state_dim=5,
            action_dim=9
        )
        if 'q_table' in checkpoint:
            agent.Q = checkpoint['q_table']
        print(f"✓ Loaded Q-Learning agent from {model_path}")
    
    else:
        raise ValueError(f"Unknown agent type in {model_path}")
    
    # Set to evaluation mode
    if hasattr(agent, 'network'):
        agent.network.eval()
    elif hasattr(agent, 'q_network'):
        agent.q_network.eval()
    
    return agent


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained agents in The Autonomous Colony"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of episodes to visualize (default: 5)'
    )
    
    parser.add_argument(
        '--n-agents',
        type=int,
        default=2,
        help='Number of agents (default: 2)'
    )
    
    parser.add_argument(
        '--grid-size',
        type=int,
        default=20,
        help='Grid size (default: 20)'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=500,
        help='Maximum steps per episode (default: 500)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Output directory for visualizations'
    )
    
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save individual frames for each step'
    )
    
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable real-time rendering'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to run on (cpu/cuda)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("THE AUTONOMOUS COLONY - Agent Visualization")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Grid Size: {args.grid_size}×{args.grid_size}")
    print(f"Agents: {args.n_agents}")
    print(f"Output: {args.output_dir}/")
    print()
    
    # Create environment
    env = ColonyEnvironment(
        n_agents=args.n_agents,
        grid_size=args.grid_size
    )
    
    # Load agent
    agent = load_agent(args.model, env, device=args.device)
    
    # Create visualization session
    session = VisualizationSession(
        agent=agent,
        env=env,
        output_dir=args.output_dir
    )
    
    # Run episodes
    session.run_multiple_episodes(
        num_episodes=args.episodes,
        render=not args.no_render,
        save_frames=args.save_frames
    )
    
    # Save visualizations
    session.save_visualizations()
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {args.output_dir}/")
    print("  - heatmaps.png")
    print("  - trajectories.png")
    print("  - dashboard.png")
    print("  - metrics.json")
    print("  - summary.txt")
    if args.save_frames:
        print("  - ep*_step*.png (individual frames)")
    print()


if __name__ == "__main__":
    main()
