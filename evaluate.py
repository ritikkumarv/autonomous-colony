#!/usr/bin/env python3
"""
The Autonomous Colony - Evaluation Script

Evaluate trained agents and generate detailed performance reports.

Usage:
    python evaluate.py --model models/ppo_final.pt --episodes 100
    python evaluate.py --model models/ma_ppo_checkpoint.pt --n_agents 4 --save_results
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from src.environment import ColonyEnvironment
from src.agents import PPOAgent, DQNAgent, TabularQLearningAgent
from src.multiagent import MultiAgentPPO

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--n_agents", type=int, default=2, help="Number of agents")
    parser.add_argument("--env_size", type=int, default=20, help="Grid size")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--save_results", action="store_true", help="Save results to JSON")
    parser.add_argument("--agent_type", type=str, default="ppo", 
                       choices=["ppo", "dqn", "ma_ppo", "q_learning"],
                       help="Agent type")
    
    return parser.parse_args()

def evaluate(args):
    """Run evaluation"""
    print(f"\n{'='*80}")
    print("🧪 EVALUATION MODE")
    print(f"{'='*80}\n")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Agents: {args.n_agents}")
    print(f"Environment: {args.env_size}×{args.env_size}\n")
    
    # Create environment
    env = ColonyEnvironment(n_agents=args.n_agents, grid_size=args.env_size)
    
    # Load agent
    # TODO: Implement agent loading based on checkpoint
    print("⚠️  Agent loading not yet implemented")
    print("Use visualize.py for now to evaluate trained models")
    
    return {}

def main():
    """Main entry point"""
    args = parse_args()
    results = evaluate(args)
    
    if args.save_results and results:
        output_path = Path("results") / f"eval_{Path(args.model).stem}.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())
