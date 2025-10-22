"""
Core module for the Applied Probability and Automation Framework.

This module provides the foundational classes and interfaces for building
probabilistic game simulation frameworks with extensible architecture.

Author: Shihab Belal
License: MIT
Version: 2.0.0
"""

from .base import (
    BaseGameModel,
    BaseStrategy,
    BaseSimulator,
    SimulationConfig,
    SimulationResult
)
from .game_simulator import GameSimulator
from .probability_engine import ProbabilityEngine
from .statistics import StatisticsEngine

__all__ = [
    'BaseGameModel',
    'BaseStrategy', 
    'BaseSimulator',
    'SimulationConfig',
    'SimulationResult',
    'GameSimulator',
    'ProbabilityEngine',
    'StatisticsEngine'
]

__version__ = "2.0.0"