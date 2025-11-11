# 🚀 Quick Start Guide - The Autonomous Colony

## Choose Your Path

### 🏃 Just Want To See It Work? (5 minutes)
```bash
# Clone and run basic training
git clone https://github.com/ritikkumarv/autonomous-colony.git
cd autonomous-colony
pip install -r requirements.txt
python train.py --agent ppo --episodes 100 --n_agents 2 --env_size 15
```

### 🎮 Want Fast GPU Training? (30 minutes)
1. Open [colab_training.ipynb](https://colab.research.google.com/github/ritikkumarv/autonomous-colony/blob/main/colab_training.ipynb)
2. Runtime → Change runtime type → **GPU (T4)**
3. Run all cells
4. Download model from Google Drive
5. Visualize locally:
```bash
python visualize.py --model models/ppo_final.pt --episodes 10
```

### 🧪 Want To Experiment? (1-2 hours)
Open the notebooks in order:
1. `part1_environment.ipynb` - Build the world
2. `part2_agents.ipynb` - Train single agents
3. `part3_multiagent.ipynb` - Multi-agent coordination
4. `part4_advanced.ipynb` - Advanced techniques

---

## 📊 Training Commands Cheat Sheet

### Single Agent PPO (Baseline)
```bash
python train.py --agent ppo --episodes 500 --n_agents 2
```

### Multi-Agent with Communication
```bash
python train.py --agent ma_ppo --n_agents 4 --communication --cooperation_bonus 2.0
```

### With Curiosity-Driven Exploration
```bash
python train.py --agent ppo --curiosity --curiosity_weight 0.5 --episodes 600
```

### With Curriculum Learning
```bash
python train.py --agent ppo --curriculum --episodes 800
```

### Everything Combined
```bash
python train.py \
  --agent ma_ppo \
  --n_agents 6 \
  --communication \
  --curiosity \
  --curriculum \
  --world_model \
  --cooperation_bonus 2.5 \
  --curiosity_weight 0.3 \
  --episodes 1000
```

### Hierarchical RL
```bash
python train.py --agent hierarchical --n_agents 2 --episodes 600
```

---

## 🎯 Common Flags

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| `--agent` | Agent type (ppo, dqn, ma_ppo, hierarchical) | ppo | `--agent ma_ppo` |
| `--episodes` | Number of training episodes | 500 | `--episodes 1000` |
| `--n_agents` | Number of agents | 3 | `--n_agents 6` |
| `--env_size` | Grid size | 20 | `--env_size 30` |
| `--max_steps` | Max steps per episode | 500 | `--max_steps 1000` |
| `--lr` | Learning rate | 3e-4 | `--lr 1e-4` |
| `--gamma` | Discount factor | 0.99 | `--gamma 0.95` |
| `--communication` | Enable agent communication | False | `--communication` |
| `--curiosity` | Enable curiosity module | False | `--curiosity` |
| `--curriculum` | Enable curriculum learning | False | `--curriculum` |
| `--world_model` | Enable world model | False | `--world_model` |
| `--cooperation_bonus` | Cooperation reward weight | 2.0 | `--cooperation_bonus 3.0` |
| `--curiosity_weight` | Curiosity reward weight | 0.5 | `--curiosity_weight 0.3` |
| `--save_freq` | Save checkpoint every N episodes | 50 | `--save_freq 25` |
| `--eval_freq` | Evaluate every N episodes | 20 | `--eval_freq 10` |
| `--no_render` | Disable rendering | False | `--no_render` |

---

## 📈 Monitoring Training

### TensorBoard (Real-time Visualization)
```bash
# In a separate terminal
tensorboard --logdir logs

# Open browser to http://localhost:6006
# View: rewards, losses, exploration metrics, etc.
```

### Console Output
```
Episode 100/500
  Avg Reward (last 10): 45.32 ± 8.12
  Episode Length: 324
  Best Reward: 62.19
  Success Rate: 67.00%
  Curiosity Reward: 0.042        # If --curiosity enabled
  Cooperation Reward: 1.234      # If MA-PPO with cooperation
  Curriculum Difficulty: 0.45    # If --curriculum enabled
```

---

## 🎨 Visualizing Results

### After Training
```bash
# Visualize any trained model
python visualize.py \
  --model models/ppo_20251111_081833/final_model.pt \
  --episodes 10 \
  --no-render  # Generate images instead of live display

# Output in visualizations/:
# - heatmaps.png (where agents explored)
# - trajectories.png (agent movement paths)
# - dashboard.png (training metrics)
# - metrics.json (detailed statistics)
# - summary.txt (episode-by-episode results)
```

### During Training (Live)
```bash
# Remove --no_render flag
python train.py --agent ppo --episodes 500 --render_freq 50
```

---

## 🔍 Analyzing Results

### Load Metrics
```python
import json
with open('visualizations/metrics.json') as f:
    data = json.load(f)
    
# Print summary
for ep in data:
    print(f"Ep {ep['episode']}: {ep['total_reward']:.1f} reward, "
          f"{ep['steps']} steps, success={ep['success']}")
```

### Compare Experiments
```bash
# Train with different configs
python train.py --agent ppo --episodes 500 --log_dir logs/baseline
python train.py --agent ppo --episodes 500 --curiosity --log_dir logs/curiosity

# Compare in TensorBoard
tensorboard --logdir logs
```

---

## 🐛 Troubleshooting

### Training Too Slow?
```bash
# Use Google Colab with GPU (10-100x faster!)
# Or reduce episode count for quick tests
python train.py --agent ppo --episodes 50 --n_agents 2
```

### Agent Not Learning?
```bash
# Lower learning rate
python train.py --agent ppo --lr 1e-4

# Increase exploration
python train.py --agent ppo --curiosity --curiosity_weight 0.8
```

### Out of Memory?
```bash
# Reduce grid size
python train.py --agent ppo --env_size 15 --n_agents 2

# Or reduce number of agents
python train.py --agent ppo --n_agents 2
```

### Model Won't Load?
```python
# Verify model file exists
import os
print(os.path.exists('models/my_model.pt'))

# Check PyTorch version (need 2.0+)
import torch
print(torch.__version__)
```

---

## 🎓 Learning Path

### Week 1: Foundation
- [ ] Run basic PPO training
- [ ] Watch agents in visualization
- [ ] Try different grid sizes and agent counts

### Week 2: Multi-Agent
- [ ] Train MA-PPO with communication
- [ ] Compare with/without cooperation bonuses
- [ ] Analyze emergent behaviors

### Week 3: Advanced
- [ ] Add curiosity-driven exploration
- [ ] Try curriculum learning
- [ ] Experiment with hierarchical RL

### Week 4: Research
- [ ] Design custom experiments
- [ ] Compare different configurations
- [ ] Share results with community

---

## 📚 File Locations

```
autonomous-colony/
├── train.py              # Main training script
├── visualize.py          # Visualization tool
├── evaluate.py           # Evaluation script
├── colab_training.ipynb  # Google Colab notebook
│
├── models/               # Saved checkpoints
│   └── ppo_*/
│       ├── final_model.pt
│       ├── checkpoint_ep*.pt
│       ├── config.json
│       └── metrics.json
│
├── logs/                 # TensorBoard logs
│   └── ppo_*/
│
├── visualizations/       # Generated visualizations
│   ├── heatmaps.png
│   ├── trajectories.png
│   ├── dashboard.png
│   ├── metrics.json
│   └── summary.txt
│
└── notebooks/            # Educational notebooks
    ├── part1_environment.ipynb
    ├── part2_agents.ipynb
    ├── part3_multiagent.ipynb
    └── part4_advanced.ipynb
```

---

## 🔗 Resources

- **Main README**: [README.md](README.md) - Full documentation
- **Integration Summary**: [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md) - Technical details
- **Issues**: [GitHub Issues](https://github.com/ritikkumarv/autonomous-colony/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ritikkumarv/autonomous-colony/discussions)

---

## 💡 Pro Tips

1. **Start small**: 1-2 agents, 15×15 grid, 100 episodes
2. **Use Colab**: GPU makes training 10-100x faster
3. **Monitor TensorBoard**: Watch metrics in real-time
4. **Save often**: Use `--save_freq 50` to avoid losing progress
5. **Visualize results**: Pictures tell the story better than numbers
6. **Experiment**: Try different agent types and features

---

## ❓ Need Help?

- 📖 Read the [full README](README.md)
- 🔍 Check [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md) for technical details
- 💬 Ask in [GitHub Discussions](https://github.com/ritikkumarv/autonomous-colony/discussions)
- 🐛 Report bugs in [GitHub Issues](https://github.com/ritikkumarv/autonomous-colony/issues)

---

**Ready? Let's train some agents! 🚀**

```bash
git clone https://github.com/ritikkumarv/autonomous-colony.git
cd autonomous-colony
python train.py --agent ppo --episodes 500
```
