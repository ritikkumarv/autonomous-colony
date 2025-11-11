"""
Multi-Agent Reinforcement Learning Module

Provides communication, coordination, and multi-agent algorithms for
cooperative multi-agent scenarios.
"""

from .communication import (
    CommunicationNetwork,
    BroadcastCommunication,
    CommChannel
)

from .coordination import (
    CentralizedCritic,
    ValueDecompositionNetwork,
    QMIXMixer,
    CooperationReward,
    TeamReward
)

from .ma_ppo import (
    MultiAgentActorCritic,
    MultiAgentPPO
)

__all__ = [
    # Communication
    'CommunicationNetwork',
    'BroadcastCommunication',
    'CommChannel',
    # Coordination
    'CentralizedCritic',
    'ValueDecompositionNetwork',
    'QMIXMixer',
    'CooperationReward',
    'TeamReward',
    # MAPPO
    'MultiAgentActorCritic',
    'MultiAgentPPO',
]
