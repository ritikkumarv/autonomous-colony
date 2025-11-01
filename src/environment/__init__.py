"""
Colony Environment for The Autonomous Colony

Exports the main environment class and related components
"""

from .colony_env import (
    ColonyEnvironment,
    Action,
    Resource,
    Position,
    Agent,
    AgentStats,
    GridWorld,
    # Constants
    GRID_SIZE,
    MAX_AGENTS,
    MAX_STEPS,
    ENERGY_DECAY_RATE,
    INITIAL_ENERGY,
    INITIAL_HEALTH,
)

__all__ = [
    'ColonyEnvironment',
    'Action',
    'Resource',
    'Position',
    'Agent',
    'AgentStats',
    'GridWorld',
    'GRID_SIZE',
    'MAX_AGENTS',
    'MAX_STEPS',
    'ENERGY_DECAY_RATE',
    'INITIAL_ENERGY',
    'INITIAL_HEALTH',
]
