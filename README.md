# 🌍 The Autonomous Colony
## Multi-Agent Reinforcement Learning in a Grid World

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A comprehensive reinforcement learning project covering single-agent, multi-agent, and advanced RL concepts through a simulated colony environment.

---

## 🎯 Overview

**The Autonomous Colony** is a multi-agent RL environment where agents learn to:

- 🏃 **Navigate** a 2D grid world
- 🍎 **Collect resources** (food, water, materials) to survive
- 🧠 **Learn** using various RL algorithms (Q-Learning, DQN, PPO, MA-PPO)
- 🤝 **Cooperate** through communication and coordination
- 🌱 **Explore** using curiosity-driven learning
- 📈 **Adapt** through curriculum and meta-learning

---

## 🧠 RL Concepts Implemented

### Core Algorithms
- **Tabular Q-Learning** - Classic value-based RL
- **Deep Q-Network (DQN)** - Function approximation with experience replay
- **Proximal Policy Optimization (PPO)** - State-of-the-art policy gradient
- **Multi-Agent PPO (MAPPO)** - Centralized training, decentralized execution

### Multi-Agent Features
- **Communication Networks** - Learned message passing between agents
- **Cooperation Rewards** - Proximity, sharing, and joint success bonuses
- **Value Decomposition** - Individual contributions to team success

### Advanced Features
- **Curiosity-Driven Exploration** - Intrinsic Curiosity Module (ICM)
- **Hierarchical RL** - Temporal abstraction with meta-controllers
- **World Models** - Model-based RL with predictive models
- **Meta-Learning** - MAML-style adaptation to new tasks
- **Curriculum Learning** - Progressive difficulty adjustment

---

## 🚀 Quick Start

### Installation

\`\`\`bash
git clone https://github.com/ritikkumarv/autonomous-colony.git
cd autonomous-colony
pip install -r requirements.txt
\`\`\`

### Training

\`\`\`bash
# Single agent PPO
python train.py --agent ppo --episodes 1000

# Multi-agent with communication
python train.py --agent ma_ppo --n_agents 4 --episodes 2000 --communication

# With curiosity and curriculum learning
python train.py --agent ppo --episodes 2000 --curiosity --curriculum

# All features combined
python train.py --agent ma_ppo --n_agents 4 --episodes 3000 \\
    --communication --curiosity --curriculum --world_model
\`\`\`

### Visualization

\`\`\`bash
# Visualize trained agent
python visualize.py --model models/ppo_latest/model.pt --episodes 5

# Create training plots
python visualize.py --model models/ppo_latest/model.pt --plot_training
\`\`\`

### Evaluation

\`\`\`bash
# Evaluate trained agent
python evaluate.py --model models/ppo_latest/model.pt --episodes 100
\`\`\`

---

## 📁 Project Structure

\`\`\`
autonomous-colony/
│
├── train.py                      # Main training script
├── visualize.py                  # Visualization tool
├── evaluate.py                   # Evaluation script
├── download_models.py            # Download pre-trained models
│
├── notebooks/                    # Learning notebooks
│   ├── part1_environment.ipynb   # Environment setup
│   ├── part2_agents.ipynb        # Single-agent RL
│   ├── part3_multiagent.ipynb    # Multi-agent RL
│   └── part4_advanced.ipynb      # Advanced features
│
├── src/
│   ├── environment/              # Grid world environment
│   │   ├── colony_env.py         # Main environment class
│   │   ├── resources.py          # Resource spawning
│   │   └── rendering.py          # Visualization
│   │
│   ├── agents/                   # RL agents
│   │   ├── tabular_q.py          # Q-Learning
│   │   ├── dqn.py                # Deep Q-Network
│   │   ├── ppo.py                # PPO
│   │   └── base_agent.py         # Base agent class
│   │
│   ├── multiagent/               # Multi-agent systems
│   │   ├── ma_ppo.py             # Multi-agent PPO
│   │   ├── communication.py      # Communication networks
│   │   └── coordination.py       # Cooperation rewards
│   │
│   ├── advanced/                 # Advanced RL features
│   │   ├── curiosity.py          # ICM & RND
│   │   ├── hierarchical.py       # Hierarchical RL
│   │   ├── world_model.py        # Model-based RL
│   │   ├── meta_learning.py      # MAML
│   │   └── curriculum.py         # Curriculum learning
│   │
│   └── utils/                    # Utilities
│       ├── training.py           # Training helpers
│       ├── logging.py            # Logging utilities
│       └── checkpointing.py      # Model checkpointing
│
├── models/                       # Saved models
├── logs/                         # Training logs
├── results/                      # Evaluation results
└── visualizations/               # Generated plots
\`\`\`

---

## 🎮 Training Arguments

### Basic Options
\`\`\`
--agent {q_learning,dqn,ppo,ma_ppo,hierarchical}
--episodes N              Number of training episodes
--n_agents N             Number of agents (for multi-agent)
--env_size N             Grid size (default: 16)
--max_steps N            Max steps per episode (default: 200)
\`\`\`

### Multi-Agent Options
\`\`\`
--communication          Enable communication networks
--cooperation            Add cooperation rewards
--value_decomposition    Use value decomposition networks
\`\`\`

### Advanced Options
\`\`\`
--curiosity              Enable curiosity-driven exploration
--curriculum             Use curriculum learning
--world_model            Enable world model learning
--meta_learning          Use meta-learning (MAML)
\`\`\`

### Training Options
\`\`\`
--lr FLOAT               Learning rate (default: 3e-4)
--gamma FLOAT            Discount factor (default: 0.99)
--no_render              Disable live rendering during training
--checkpoint_freq N      Save checkpoint every N episodes
\`\`\`

---

## 📊 Monitoring Training

Training metrics are logged to TensorBoard:

\`\`\`bash
tensorboard --logdir logs/
\`\`\`

Metrics include:
- Episode rewards (mean, min, max)
- Success rate
- Episode length
- Loss values (policy, value, entropy)
- Curiosity rewards (if enabled)
- Communication patterns (if enabled)

---

## 🔬 Experiments

### Baseline Comparisons

\`\`\`bash
# Compare different algorithms
python train.py --agent q_learning --episodes 1000
python train.py --agent dqn --episodes 1000
python train.py --agent ppo --episodes 1000
\`\`\`

### Ablation Studies

\`\`\`bash
# Test impact of curiosity
python train.py --agent ppo --episodes 2000                    # baseline
python train.py --agent ppo --episodes 2000 --curiosity        # with ICM

# Test impact of curriculum
python train.py --agent ppo --episodes 2000                    # baseline
python train.py --agent ppo --episodes 2000 --curriculum       # adaptive
\`\`\`

### Multi-Agent Studies

\`\`\`bash
# Test communication
python train.py --agent ma_ppo --n_agents 4 --episodes 2000                      # baseline
python train.py --agent ma_ppo --n_agents 4 --episodes 2000 --communication      # with comm

# Test cooperation rewards
python train.py --agent ma_ppo --n_agents 4 --episodes 2000                      # baseline
python train.py --agent ma_ppo --n_agents 4 --episodes 2000 --cooperation        # with coop
\`\`\`

---

## 🎓 Learning Notebooks

Explore the concepts step-by-step:

1. **Part 1: Environment** - Build the grid world, understand MDP formulation
2. **Part 2: Agents** - Implement Q-Learning, DQN, and PPO
3. **Part 3: Multi-Agent** - Add communication and coordination
4. **Part 4: Advanced** - Explore curiosity, hierarchical RL, and meta-learning

Each notebook is self-contained with:
- Theory explanations
- Code implementations
- Visualizations
- Exercises

---

## 🛠️ Development

### Running Tests

\`\`\`bash
# Unit tests (coming soon)
pytest tests/unit/

# Integration tests (coming soon)
pytest tests/integration/
\`\`\`

### Code Structure

- **Environment**: Custom Gymnasium environment with partial observability
- **Agents**: Modular agent implementations with common base class
- **Training**: Unified training loop supporting all agent types
- **Visualization**: Multiple rendering modes (grid, trajectories, heatmaps)

---

## 📈 Performance Tips

### For Faster Training

1. **Use smaller environments**: \`--env_size 8\` for quick experiments
2. **Reduce agents**: Start with \`--n_agents 1\` or \`2\`
3. **Disable rendering**: Use \`--no_render\` flag
4. **Adjust episode length**: Use \`--max_steps 100\` for faster iterations

### For Better Results

1. **More episodes**: Train for \`--episodes 3000+\`
2. **Tune learning rate**: Try \`--lr 1e-4\` or \`--lr 5e-4\`
3. **Enable features**: Use \`--curiosity --curriculum\` for sparse rewards
4. **Multiple runs**: Average results over 3-5 random seeds

---

## �� Contributing

Contributions are welcome! Areas for improvement:

- Additional RL algorithms (A3C, SAC, TD3)
- More advanced features (transformer agents, graph networks)
- Better curriculum strategies
- Improved visualizations
- Documentation and tutorials

---

## 📚 Resources

### Reinforcement Learning
- [Sutton & Barto - RL: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Deep RL Course by Hugging Face](https://huggingface.co/learn/deep-rl-course)

### Multi-Agent RL
- [Multi-Agent RL: Foundations and Modern Approaches](https://www.marl-book.com/)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)

### Implementation References
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ✨ Acknowledgments

Built as a comprehensive learning project covering:
- Single-agent RL (Q-Learning, DQN, PPO)
- Multi-agent RL (MAPPO, communication, cooperation)
- Advanced RL (curiosity, hierarchical, world models, meta-learning)

Inspired by research in multi-agent systems, curriculum learning, and intrinsic motivation.

---

**Happy Learning! 🚀**
