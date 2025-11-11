# 🧹 Cleanup & Standardization Plan

## Overview
This document outlines the step-by-step plan to clean up and standardize The Autonomous Colony project structure following Python best practices.

---

## 🎯 Goals

1. **Organize test suite** - Move tests to proper directory structure
2. **Extract utilities** - Separate reusable code from train.py
3. **Add documentation** - Comprehensive docstrings and type hints
4. **Standardize structure** - Follow Python package conventions
5. **Improve maintainability** - Make codebase easier to understand and extend

---

## 📋 Phase 1: Test Suite Organization

### Current State
```
.
├── test_agents.py              # 154 lines - agent tests
├── test_grid_observations.py   # 111 lines - observation tests
└── (scattered test files)
```

### Target Structure
```
tests/
├── __init__.py
├── conftest.py                 # Pytest fixtures and configuration
├── unit/                       # Unit tests for individual components
│   ├── __init__.py
│   ├── test_environment.py     # Environment class tests
│   ├── test_agents.py          # Agent class tests
│   ├── test_observations.py    # Observation generation tests
│   └── test_utils.py           # Utility function tests
├── integration/                # Integration tests
│   ├── __init__.py
│   ├── test_training.py        # Full training pipeline tests
│   └── test_agent_env.py       # Agent-environment interaction tests
└── benchmarks/                 # Performance benchmarks
    ├── __init__.py
    └── test_performance.py     # Speed and memory benchmarks
```

### Benefits
- ✅ Clear separation of test types
- ✅ Easy to run specific test categories
- ✅ Standard Python testing conventions
- ✅ Pytest fixtures for reusable test code
- ✅ Better CI/CD integration

---

## 📋 Phase 2: Extract Utilities from train.py

### Current State
`train.py` contains 801 lines with mixed responsibilities:
- Training loop logic
- Checkpointing code
- Logging and metrics
- Model saving/loading
- Evaluation logic
- CLI argument parsing

### Extraction Plan

#### 1. **checkpointing.py** - Model persistence
```python
# src/utils/checkpointing.py

class CheckpointManager:
    """Handles model saving and loading"""
    
    def save_checkpoint(model, optimizer, episode, metrics, path)
    def load_checkpoint(path) -> dict
    def save_best_model(model, path)
    def get_latest_checkpoint(directory) -> str
```

**Extracted from:** train.py lines ~500-550

---

#### 2. **logging.py** - Metrics and TensorBoard
```python
# src/utils/logging.py

class MetricsLogger:
    """Tracks and logs training metrics"""
    
    def __init__(log_dir, use_tensorboard=True)
    def log_scalar(tag, value, step)
    def log_episode(episode, reward, length, metrics)
    def log_evaluation(episode, eval_metrics)
    def close()

class MetricsTracker:
    """Tracks episode statistics"""
    
    def add_episode(reward, length, success)
    def get_statistics(window=10) -> dict
    def reset()
```

**Extracted from:** train.py lines ~100-200, ~400-450

---

#### 3. **training.py** - Training loop helpers
```python
# src/utils/training.py

class TrainingManager:
    """Manages the training process"""
    
    def __init__(env, agent, config)
    def run_episode() -> (reward, length, metrics)
    def evaluate(n_episodes) -> dict
    def train(n_episodes) -> history

def set_random_seed(seed):
    """Set random seeds for reproducibility"""

def get_device(prefer_gpu=True) -> torch.device:
    """Get compute device"""

def create_agent(agent_type, **kwargs) -> BaseAgent:
    """Factory function for agent creation"""
```

**Extracted from:** train.py lines ~200-400

---

### Benefits of Extraction
- ✅ **Reusability:** Use in evaluate.py, visualize.py, experiments
- ✅ **Testability:** Each utility can be tested independently
- ✅ **Maintainability:** Changes affect only relevant modules
- ✅ **Clarity:** train.py becomes high-level orchestration

---

## 📋 Phase 3: Documentation Standards

### Docstring Format
We'll use **Google Style** docstrings:

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """Brief one-line description.
    
    Longer description if needed. Explain what the function does,
    when to use it, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something goes wrong
        
    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
    pass
```

### Type Hints
All functions should have type hints:
```python
from typing import Dict, List, Optional, Tuple, Union

def process_observation(
    obs: Dict[str, np.ndarray],
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Process observation dictionary into arrays."""
    pass
```

### Class Documentation
```python
class AgentManager:
    """Manages multiple RL agents in an environment.
    
    This class handles:
    - Agent initialization and configuration
    - Action selection for multiple agents
    - Learning updates across agents
    - Agent state management
    
    Attributes:
        agents: List of agent instances
        n_agents: Number of agents
        config: Configuration dictionary
        
    Example:
        >>> manager = AgentManager(n_agents=3, agent_type='ppo')
        >>> actions = manager.select_actions(observations)
    """
    pass
```

---

## 📋 Phase 4: Package Structure Standardization

### Add setup.py
```python
from setuptools import setup, find_packages

setup(
    name="autonomous-colony",
    version="0.1.0",
    description="A comprehensive multi-agent RL learning project",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0.0",
        "torch>=2.0.0",
        "gymnasium>=1.0.0",
        "matplotlib>=3.5.0",
        "tensorboard>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    python_requires=">=3.8",
)
```

**Benefits:**
- Install with `pip install -e .` for development
- Import as `from autonomous_colony import ...`
- Dependency management
- Version control

---

### Add pyproject.toml
```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "autonomous-colony"
version = "0.1.0"
description = "A comprehensive multi-agent RL learning project"
requires-python = ">=3.8"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"

[tool.black]
line-length = 100
target-version = ['py38']

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
```

---

### Add .gitignore (if not present)
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Distribution / packaging
dist/
build/
*.egg-info/

# Testing
.pytest_cache/
.coverage
htmlcov/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Project specific
logs/
models/
results/
*.pt
*.pth

# Jupyter
.ipynb_checkpoints/
```

---

## 📋 Phase 5: Code Quality Tools

### 1. **pytest** - Testing framework
```bash
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test category
pytest tests/unit/
pytest tests/integration/
```

### 2. **black** - Code formatter
```bash
pip install black

# Format all Python files
black src/ tests/ *.py

# Check formatting
black --check src/
```

### 3. **flake8** - Linter
```bash
pip install flake8

# Lint code
flake8 src/ tests/

# Config in setup.cfg
[flake8]
max-line-length = 100
exclude = .git,__pycache__,venv
```

### 4. **mypy** - Type checker
```bash
pip install mypy

# Check types
mypy src/

# Strict mode
mypy --strict src/
```

---

## 📋 Phase 6: Documentation

### Add docs/ directory
```
docs/
├── index.md                    # Main documentation
├── tutorials/
│   ├── quickstart.md
│   ├── training_guide.md
│   └── custom_agents.md
├── api/
│   ├── agents.md
│   ├── environment.md
│   └── utils.md
└── examples/
    ├── basic_training.md
    ├── multi_agent.md
    └── advanced_features.md
```

---

## 🎯 Implementation Order

### Step 1: Test Organization (20 min)
1. Create `tests/` directory structure
2. Move existing tests
3. Create `conftest.py` with fixtures
4. Add `__init__.py` files
5. Update imports
6. Run tests to verify

### Step 2: Extract Checkpointing (15 min)
1. Create `src/utils/checkpointing.py`
2. Extract save/load functions
3. Add tests
4. Update train.py imports
5. Verify training still works

### Step 3: Extract Logging (20 min)
1. Create `src/utils/logging.py`
2. Extract MetricsLogger class
3. Extract MetricsTracker class
4. Add tests
5. Update train.py imports
6. Verify training still works

### Step 4: Extract Training Utilities (20 min)
1. Create `src/utils/training.py`
2. Extract helper functions
3. Extract TrainingManager (optional)
4. Add tests
5. Update train.py imports
6. Verify training still works

### Step 5: Add Documentation (30 min)
1. Add docstrings to all public functions
2. Add type hints
3. Update README if needed
4. Generate API docs (optional)

### Step 6: Package Setup (15 min)
1. Create `setup.py`
2. Create `pyproject.toml`
3. Test installation: `pip install -e .`
4. Update imports if needed

### Step 7: Code Quality (20 min)
1. Install dev tools
2. Run black formatter
3. Run flake8 linter
4. Fix issues
5. Add pre-commit hooks (optional)

**Total Time: ~2.5 hours**

---

## ✅ Success Criteria

After cleanup, the project should:
- ✅ Have organized test directory with >80% coverage
- ✅ Have reusable utility modules
- ✅ Have comprehensive documentation
- ✅ Follow Python best practices
- ✅ Be installable as a package
- ✅ Pass all tests
- ✅ Have clean code (black + flake8)

---

## 🎓 Learning Outcomes

Through this cleanup, you'll learn:

1. **Python Package Structure**
   - How to organize a real Python project
   - Package vs module organization
   - setup.py and pyproject.toml

2. **Testing Best Practices**
   - Unit vs integration tests
   - Pytest fixtures and configuration
   - Test coverage analysis

3. **Code Quality**
   - Documentation standards (docstrings, type hints)
   - Code formatting (black)
   - Linting (flake8)
   - Type checking (mypy)

4. **Software Engineering**
   - Separation of concerns
   - DRY principle (Don't Repeat Yourself)
   - Code reusability
   - Maintainability

5. **Development Workflow**
   - Version control best practices
   - Testing workflow
   - CI/CD preparation

---

## 🚀 Ready to Start!

Let's begin with **Step 1: Test Organization**. This is the safest first step since we're just moving files and won't break any existing functionality.

Shall we proceed?
