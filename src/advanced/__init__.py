"""
Advanced Reinforcement Learning Techniques

Provides advanced RL methods for improved learning:
- Curiosity-driven exploration
- Hierarchical RL
- World models
- Meta-learning
- Curriculum learning
"""

from .curiosity import (
    IntrinsicCuriosityModule,
    RandomNetworkDistillation
)

# Aliases for convenience
ICM = IntrinsicCuriosityModule
RND = RandomNetworkDistillation

from .hierarchical import (
    Option,
    HierarchicalAgent,
    GoalConditionedPolicy,
    HierarchicalQLearning,
    create_colony_options
)

from .world_model import (
    WorldModel,
    DynaQAgent
)

from .meta_learning import (
    MAMLAgent,
    ReptileAgent
)

from .curriculum import (
    CurriculumScheduler,
    StageBasedCurriculum,
    TaskCurriculum
)

__all__ = [
    # Curiosity
    'IntrinsicCuriosityModule',
    'RandomNetworkDistillation',
    'ICM',  # Alias
    'RND',  # Alias
    # Hierarchical RL
    'Option',
    'HierarchicalAgent',
    'GoalConditionedPolicy',
    'HierarchicalQLearning',
    'create_colony_options',
    # World Model
    'WorldModel',
    'DynaQAgent',
    # Meta-Learning
    'MAMLAgent',
    'ReptileAgent',
    # Curriculum
    'CurriculumScheduler',
    'StageBasedCurriculum',
    'TaskCurriculum',
]
