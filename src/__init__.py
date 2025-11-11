"""The Autonomous Colony - Main Package"""

__version__ = "1.0.0"
__author__ = "Autonomous Colony Contributors"
__description__ = "A comprehensive multi-agent reinforcement learning project"

from . import environment
from . import agents
from . import multiagent
from . import advanced
from . import utils

__all__ = [
    'environment',
    'agents',
    'multiagent',
    'advanced',
    'utils',
]
