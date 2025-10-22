"""
Evaluation and benchmarking module for AI progress tracking.

This module provides comprehensive evaluation tools for measuring
and comparing AI agent performance improvements over time.

Author: Shihab Belal
License: MIT
Version: 2.0.0
"""

from .metrics import (
    PerformanceMetrics,
    ImprovementMetrics,
    BenchmarkSuite,
    StatisticalSignificance
)
from .evaluator import (
    AIProgressEvaluator,
    ComparativeAnalysis,
    TrendAnalysis
)
from .benchmarks import (
    StandardBenchmarks,
    CustomBenchmark,
    BenchmarkRunner
)

__all__ = [
    'PerformanceMetrics',
    'ImprovementMetrics', 
    'BenchmarkSuite',
    'StatisticalSignificance',
    'AIProgressEvaluator',
    'ComparativeAnalysis',
    'TrendAnalysis',
    'StandardBenchmarks',
    'CustomBenchmark',
    'BenchmarkRunner'
]

__version__ = "2.0.0"