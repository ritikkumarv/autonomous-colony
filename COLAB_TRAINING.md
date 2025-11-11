# 🚀 Google Colab Training Guide

Train your agents on GPU/TPU with Google Colab for **10-100x faster training!**

## Quick Start (3 Steps)

### 1️⃣ Open in Colab

Click this link to open the notebook in Google Colab:

**[Open Colab Training Notebook](https://colab.research.google.com/github/ritikkumarv/autonomous-colony/blob/main/colab_training.ipynb)**

Or manually upload `colab_training.ipynb` to Google Colab.

### 2️⃣ Enable GPU/TPU

1. In Colab: **Runtime → Change runtime type**
2. Select **Hardware accelerator**: GPU
3. Recommended: **T4 GPU** (free tier) or **A100** (Colab Pro)
4. Click **Save**

### 3️⃣ Run All Cells

Click **Runtime → Run all** and let it train!

The notebook will:
- ✅ Check GPU availability
- ✅ Mount Google Drive (for model persistence)
- ✅ Clone this repository
- ✅ Install dependencies
- ✅ Train agent with live visualization
- ✅ Save models to your Google Drive
- ✅ Create downloadable zip of all models

## Training Configuration

Edit the configuration cell to customize:

```python
TRAINING_CONFIG = {
    # Environment
    'n_agents': 4,           # Number of agents
    'grid_size': 30,         # Grid size (30x30)
    
    # Training
    'n_episodes': 1000,      # More episodes with GPU!
    'max_steps': 500,        # Steps per episode
    'save_interval': 100,    # Save every N episodes
    
    # Agent selection
    'agent_type': 'ppo',     # 'ppo', 'dqn', or 'mappo'
    
    # Advanced features
    'use_curiosity': True,   # Add curiosity-driven exploration
    'curiosity_type': 'icm', # 'icm' or 'rnd'
    'use_hierarchical': False,
    'use_world_model': False,
    'use_curriculum': True,  # Adaptive difficulty
}
```

## Available GPU Options

| GPU | Memory | Speed | Availability |
|-----|--------|-------|--------------|
| **T4** | 16 GB | 1x | Free tier |
| **P100** | 16 GB | 2x | Free tier (limited) |
| **V100** | 16 GB | 3x | Colab Pro |
| **A100** | 40 GB | 5x | Colab Pro+ |

## Performance Comparison

| Setup | Episodes/Hour | 1000 Episodes |
|-------|---------------|---------------|
| **Local CPU** | ~20 | ~50 hours |
| **Colab T4 GPU** | ~200 | ~5 hours |
| **Colab A100** | ~500 | ~2 hours |

## After Training

### Download Models

Models are automatically saved to Google Drive at:
```
/MyDrive/autonomous_colony_models/
```

**Option 1: Download Individual Models**
- Browse to the folder in Google Drive
- Right-click → Download

**Option 2: Download Zip (Recommended)**
- The notebook creates a zip file at the end
- Downloads automatically to your computer

### Use Models Locally

1. Download trained models from Google Drive
2. Place in your local `models/` directory:
   ```bash
   # Example structure
   models/
   ├── ppo_final_20251111_123456.pt
   ├── dqn_ep500_20251111_123456.pt
   └── mappo_final_20251111_123456.pt
   ```

3. Visualize:
   ```bash
   python visualize.py --model models/ppo_final_20251111_123456.pt --episodes 10
   ```

## Training Different Agents

### PPO (Recommended for beginners)
```python
'agent_type': 'ppo',
'use_curiosity': True,
'use_curriculum': True,
```
- Fast training
- Stable performance
- Good for continuous tasks

### DQN (Good for discrete tasks)
```python
'agent_type': 'dqn',
'use_curiosity': True,
```
- Sample efficient
- Works well with replay buffer
- Good for exploration

### MAPPO (Advanced multi-agent)
```python
'agent_type': 'mappo',
'use_curiosity': False,  # Built-in communication
```
- Centralized training, decentralized execution
- Agent communication
- Best for cooperative tasks

## Advanced Features

### Curiosity-Driven Exploration

**ICM (Intrinsic Curiosity Module)**
```python
'use_curiosity': True,
'curiosity_type': 'icm',
```
- Learns forward/inverse dynamics
- Rewards novel states
- Good for sparse rewards

**RND (Random Network Distillation)**
```python
'use_curiosity': True,
'curiosity_type': 'rnd',
```
- Predicts random network outputs
- Robust to stochastic dynamics
- State-of-the-art exploration

### Curriculum Learning
```python
'use_curriculum': True,
```
- Starts easy, increases difficulty
- Adapts to agent performance
- Faster convergence

## Monitoring Training

The notebook shows **live plots** updated every 10 episodes:

1. **Episode Rewards** - Raw and moving average
2. **Episode Lengths** - How long episodes last
3. **Success Rate** - 100-episode moving average
4. **Curiosity Bonuses** - Exploration rewards (if enabled)

## Troubleshooting

### GPU Not Available
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size or grid size:
```python
'grid_size': 20,  # Instead of 30
'n_agents': 2,    # Instead of 4
```

### Training Too Slow
```
Episodes taking > 1 minute each
```
**Solution:** 
- Check GPU is enabled (Runtime → Change runtime type)
- Reduce `max_steps` or environment size
- Use fewer agents

### Models Not Saving
```
FileNotFoundError or PermissionError
```
**Solution:**
- Re-run the "Mount Google Drive" cell
- Grant permissions when prompted
- Check Google Drive storage space

## Best Practices

### For Fastest Training
1. Use **T4 GPU** (free) or **A100** (Colab Pro)
2. Start with **500-1000 episodes**
3. Use **save_interval: 100** for checkpoints
4. Enable **curriculum learning**

### For Best Performance
1. Train for **2000-5000 episodes**
2. Use **PPO with curiosity**
3. Larger environment: **grid_size: 40-50**
4. More agents: **n_agents: 6-8**

### For Experimentation
1. Train multiple configurations
2. Compare PPO vs DQN vs MAPPO
3. Test ICM vs RND curiosity
4. Try different hyperparameters

## Example Training Sessions

### Quick Test (30 minutes)
```python
'n_episodes': 500,
'n_agents': 2,
'grid_size': 20,
'agent_type': 'ppo',
```

### Standard Training (2-3 hours)
```python
'n_episodes': 2000,
'n_agents': 4,
'grid_size': 30,
'agent_type': 'ppo',
'use_curiosity': True,
'use_curriculum': True,
```

### Advanced Training (5-6 hours)
```python
'n_episodes': 5000,
'n_agents': 6,
'grid_size': 40,
'agent_type': 'mappo',
'use_curiosity': True,
'use_curriculum': True,
```

## Tips for Success

1. **Start Small**: Test with 100 episodes first
2. **Monitor Progress**: Watch the live plots
3. **Save Checkpoints**: Don't lose progress if Colab disconnects
4. **Multiple Runs**: Train several models, keep the best
5. **Experiment**: Try different configurations

## Next Steps

After training:

1. **Download models** from Google Drive
2. **Visualize locally** with `visualize.py`
3. **Compare agents** with different configurations
4. **Share results** in GitHub issues/discussions
5. **Train more advanced** configurations

## Need Help?

- 📖 Check the [main README](README.md)
- 💬 Open a [GitHub issue](https://github.com/ritikkumarv/autonomous-colony/issues)
- 📚 Read the [documentation](ENHANCEMENTS_SUMMARY.md)

Happy Training! 🚀
