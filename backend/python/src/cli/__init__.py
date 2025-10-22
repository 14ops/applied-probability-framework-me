"""
Command-line interface for the Applied Probability Framework.

This module provides a comprehensive CLI for running simulations,
managing configurations, and analyzing results.

Author: Shihab Belal
License: MIT
Version: 2.0.0
"""

from .main import main
from .commands import (
    run_simulation,
    analyze_results,
    create_config,
    list_strategies,
    benchmark_performance
)

__all__ = [
    'main',
    'run_simulation',
    'analyze_results', 
    'create_config',
    'list_strategies',
    'benchmark_performance'
]

__version__ = "2.0.0"