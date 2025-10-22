"""
Comprehensive metrics for AI progress evaluation.

This module provides detailed metrics for measuring AI agent performance,
improvements, and statistical significance of changes.

Author: Shihab Belal
License: MIT
Version: 2.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import scipy.stats as stats
from scipy.optimize import minimize
import pandas as pd
from datetime import datetime, timedelta


class MetricType(Enum):
    """Enumeration of metric types."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    WIN_RATE = "win_rate"
    AVERAGE_PAYOUT = "average_payout"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    CONVERGENCE_RATE = "convergence_rate"
    STABILITY = "stability"


class ImprovementType(Enum):
    """Enumeration of improvement types."""
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    PERCENTAGE = "percentage"
    EFFECT_SIZE = "effect_size"


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for AI agents.
    
    This class encapsulates all relevant performance metrics with
    statistical measures and confidence intervals.
    """
    # Core performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    win_rate: float = 0.0
    average_payout: float = 0.0
    efficiency: float = 0.0
    robustness: float = 0.0
    
    # Statistical measures
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    standard_error: float = 0.0
    variance: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Sample information
    sample_size: int = 0
    measurement_timestamp: datetime = field(default_factory=datetime.now)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        # Validate probability metrics (0-1 range)
        prob_metrics = [self.accuracy, self.precision, self.recall, 
                       self.f1_score, self.win_rate, self.efficiency, self.robustness]
        for metric in prob_metrics:
            if not 0 <= metric <= 1:
                raise ValueError(f"Probability metric must be in [0, 1], got {metric}")
        
        # Validate sample size
        if self.sample_size < 0:
            raise ValueError(f"Sample size must be non-negative, got {self.sample_size}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            'core_metrics': {
                'accuracy': self.accuracy,
                'precision': self.precision,
                'recall': self.recall,
                'f1_score': self.f1_score,
                'win_rate': self.win_rate,
                'average_payout': self.average_payout,
                'efficiency': self.efficiency,
                'robustness': self.robustness
            },
            'statistical_measures': {
                'confidence_interval': self.confidence_interval,
                'standard_error': self.standard_error,
                'variance': self.variance,
                'skewness': self.skewness,
                'kurtosis': self.kurtosis
            },
            'sample_info': {
                'sample_size': self.sample_size,
                'timestamp': self.measurement_timestamp.isoformat()
            },
            'metadata': self.metadata
        }
    
    def calculate_composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate composite performance score.
        
        Args:
            weights: Optional weights for different metrics
            
        Returns:
            Composite score (0-1)
        """
        if weights is None:
            weights = {
                'accuracy': 0.25,
                'f1_score': 0.25,
                'win_rate': 0.25,
                'efficiency': 0.25
            }
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted average
        composite_score = 0.0
        for metric, weight in normalized_weights.items():
            if hasattr(self, metric):
                composite_score += getattr(self, metric) * weight
        
        return composite_score


@dataclass
class ImprovementMetrics:
    """
    Metrics for measuring AI agent improvements over time.
    
    This class provides comprehensive measures of improvement including
    statistical significance and effect sizes.
    """
    # Improvement measures
    absolute_improvement: float = 0.0
    relative_improvement: float = 0.0
    percentage_improvement: float = 0.0
    effect_size: float = 0.0
    
    # Statistical significance
    p_value: float = 1.0
    is_significant: bool = False
    confidence_level: float = 0.95
    
    # Time-based measures
    improvement_rate: float = 0.0  # per unit time
    convergence_time: Optional[float] = None
    
    # Comparison metrics
    baseline_performance: float = 0.0
    current_performance: float = 0.0
    best_performance: float = 0.0
    
    # Metadata
    measurement_period: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.now(), datetime.now()))
    metric_type: MetricType = MetricType.ACCURACY
    
    def __post_init__(self):
        """Validate improvement metrics."""
        if self.percentage_improvement < -100:
            raise ValueError("Percentage improvement cannot be less than -100%")
        
        if not 0 <= self.p_value <= 1:
            raise ValueError("P-value must be in [0, 1]")
        
        if not 0 <= self.confidence_level <= 1:
            raise ValueError("Confidence level must be in [0, 1]")
    
    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of improvement metrics."""
        return {
            'improvement_measures': {
                'absolute': self.absolute_improvement,
                'relative': self.relative_improvement,
                'percentage': self.percentage_improvement,
                'effect_size': self.effect_size
            },
            'statistical_significance': {
                'p_value': self.p_value,
                'is_significant': self.is_significant,
                'confidence_level': self.confidence_level
            },
            'performance_comparison': {
                'baseline': self.baseline_performance,
                'current': self.current_performance,
                'best': self.best_performance
            },
            'temporal_measures': {
                'improvement_rate': self.improvement_rate,
                'convergence_time': self.convergence_time,
                'measurement_period': [self.measurement_period[0].isoformat(), 
                                     self.measurement_period[1].isoformat()]
            },
            'metric_type': self.metric_type.value
        }


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for AI agent evaluation.
    
    This class provides standardized benchmarks for evaluating
    AI agent performance across different scenarios and metrics.
    """
    
    def __init__(self, name: str = "Standard Benchmark Suite"):
        """
        Initialize benchmark suite.
        
        Args:
            name: Name of the benchmark suite
        """
        self.name = name
        self.benchmarks: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, PerformanceMetrics]] = {}
        self._initialize_standard_benchmarks()
    
    def _initialize_standard_benchmarks(self) -> None:
        """Initialize standard benchmark scenarios."""
        # Basic performance benchmarks
        self.benchmarks['basic_accuracy'] = {
            'description': 'Basic accuracy measurement',
            'metric_type': MetricType.ACCURACY,
            'expected_range': (0.0, 1.0),
            'weight': 1.0
        }
        
        self.benchmarks['win_rate'] = {
            'description': 'Win rate measurement',
            'metric_type': MetricType.WIN_RATE,
            'expected_range': (0.0, 1.0),
            'weight': 1.0
        }
        
        self.benchmarks['efficiency'] = {
            'description': 'Computational efficiency',
            'metric_type': MetricType.EFFICIENCY,
            'expected_range': (0.0, 1.0),
            'weight': 0.8
        }
        
        self.benchmarks['robustness'] = {
            'description': 'Robustness to noise',
            'metric_type': MetricType.ROBUSTNESS,
            'expected_range': (0.0, 1.0),
            'weight': 0.9
        }
        
        # Advanced benchmarks
        self.benchmarks['convergence_speed'] = {
            'description': 'Speed of convergence',
            'metric_type': MetricType.CONVERGENCE_RATE,
            'expected_range': (0.0, float('inf')),
            'weight': 0.7
        }
        
        self.benchmarks['stability'] = {
            'description': 'Performance stability',
            'metric_type': MetricType.STABILITY,
            'expected_range': (0.0, 1.0),
            'weight': 0.8
        }
    
    def add_custom_benchmark(self, name: str, description: str, 
                           metric_type: MetricType, expected_range: Tuple[float, float],
                           weight: float = 1.0) -> None:
        """
        Add custom benchmark to the suite.
        
        Args:
            name: Benchmark name
            description: Benchmark description
            metric_type: Type of metric to measure
            expected_range: Expected range of values
            weight: Weight for composite scoring
        """
        self.benchmarks[name] = {
            'description': description,
            'metric_type': metric_type,
            'expected_range': expected_range,
            'weight': weight
        }
    
    def run_benchmark(self, agent_name: str, agent_data: Dict[str, Any]) -> Dict[str, PerformanceMetrics]:
        """
        Run all benchmarks for an agent.
        
        Args:
            agent_name: Name of the agent
            agent_data: Agent performance data
            
        Returns:
            Dictionary of benchmark results
        """
        results = {}
        
        for benchmark_name, benchmark_config in self.benchmarks.items():
            metric_type = benchmark_config['metric_type']
            
            # Extract relevant data based on metric type
            if metric_type == MetricType.ACCURACY:
                metric_value = agent_data.get('accuracy', 0.0)
            elif metric_type == MetricType.WIN_RATE:
                metric_value = agent_data.get('win_rate', 0.0)
            elif metric_type == MetricType.EFFICIENCY:
                metric_value = agent_data.get('efficiency', 0.0)
            elif metric_type == MetricType.ROBUSTNESS:
                metric_value = agent_data.get('robustness', 0.0)
            elif metric_type == MetricType.CONVERGENCE_RATE:
                metric_value = agent_data.get('convergence_rate', 0.0)
            elif metric_type == MetricType.STABILITY:
                metric_value = agent_data.get('stability', 0.0)
            else:
                metric_value = 0.0
            
            # Create performance metrics
            performance = PerformanceMetrics(
                accuracy=agent_data.get('accuracy', 0.0),
                precision=agent_data.get('precision', 0.0),
                recall=agent_data.get('recall', 0.0),
                f1_score=agent_data.get('f1_score', 0.0),
                win_rate=agent_data.get('win_rate', 0.0),
                average_payout=agent_data.get('average_payout', 0.0),
                efficiency=agent_data.get('efficiency', 0.0),
                robustness=agent_data.get('robustness', 0.0),
                sample_size=agent_data.get('sample_size', 0),
                metadata={'benchmark': benchmark_name}
            )
            
            results[benchmark_name] = performance
        
        self.results[agent_name] = results
        return results
    
    def calculate_composite_score(self, agent_name: str) -> float:
        """
        Calculate composite benchmark score for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Composite score (0-1)
        """
        if agent_name not in self.results:
            raise ValueError(f"No results found for agent: {agent_name}")
        
        total_score = 0.0
        total_weight = 0.0
        
        for benchmark_name, performance in self.results[agent_name].items():
            benchmark_config = self.benchmarks[benchmark_name]
            weight = benchmark_config['weight']
            
            # Get the relevant metric value
            metric_type = benchmark_config['metric_type']
            if metric_type == MetricType.ACCURACY:
                score = performance.accuracy
            elif metric_type == MetricType.WIN_RATE:
                score = performance.win_rate
            elif metric_type == MetricType.EFFICIENCY:
                score = performance.efficiency
            elif metric_type == MetricType.ROBUSTNESS:
                score = performance.robustness
            else:
                score = 0.0
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def compare_agents(self, agent_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple agents across benchmarks.
        
        Args:
            agent_names: List of agent names to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            'agents': agent_names,
            'benchmark_scores': {},
            'composite_scores': {},
            'rankings': {}
        }
        
        # Calculate scores for each agent
        for agent_name in agent_names:
            if agent_name in self.results:
                comparison['composite_scores'][agent_name] = self.calculate_composite_score(agent_name)
                
                # Individual benchmark scores
                benchmark_scores = {}
                for benchmark_name, performance in self.results[agent_name].items():
                    benchmark_config = self.benchmarks[benchmark_name]
                    metric_type = benchmark_config['metric_type']
                    
                    if metric_type == MetricType.ACCURACY:
                        score = performance.accuracy
                    elif metric_type == MetricType.WIN_RATE:
                        score = performance.win_rate
                    elif metric_type == MetricType.EFFICIENCY:
                        score = performance.efficiency
                    elif metric_type == MetricType.ROBUSTNESS:
                        score = performance.robustness
                    else:
                        score = 0.0
                    
                    benchmark_scores[benchmark_name] = score
                
                comparison['benchmark_scores'][agent_name] = benchmark_scores
        
        # Calculate rankings
        composite_scores = comparison['composite_scores']
        sorted_agents = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['rankings'] = {agent: rank + 1 for rank, (agent, _) in enumerate(sorted_agents)}
        
        return comparison


class StatisticalSignificance:
    """
    Statistical significance testing for AI improvements.
    
    This class provides comprehensive statistical tests for determining
    the significance of AI agent improvements.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical significance tester.
        
        Args:
            confidence_level: Confidence level for tests (0-1)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def t_test_improvement(self, baseline_data: np.ndarray, 
                          improved_data: np.ndarray) -> Dict[str, Any]:
        """
        Perform t-test for improvement significance.
        
        Args:
            baseline_data: Baseline performance data
            improved_data: Improved performance data
            
        Returns:
            Test results dictionary
        """
        # Perform two-sample t-test
        statistic, p_value = stats.ttest_ind(improved_data, baseline_data)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(baseline_data) - 1) * np.var(baseline_data, ddof=1) + 
                             (len(improved_data) - 1) * np.var(improved_data, ddof=1)) / 
                            (len(baseline_data) + len(improved_data) - 2))
        cohens_d = (np.mean(improved_data) - np.mean(baseline_data)) / pooled_std
        
        # Determine significance
        is_significant = p_value < self.alpha
        
        return {
            'test_type': 'two_sample_t_test',
            'statistic': statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'effect_size': cohens_d,
            'confidence_level': self.confidence_level,
            'baseline_mean': np.mean(baseline_data),
            'improved_mean': np.mean(improved_data),
            'baseline_std': np.std(baseline_data, ddof=1),
            'improved_std': np.std(improved_data, ddof=1)
        }
    
    def mann_whitney_test(self, baseline_data: np.ndarray, 
                         improved_data: np.ndarray) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test for improvement significance.
        
        Args:
            baseline_data: Baseline performance data
            improved_data: Improved performance data
            
        Returns:
            Test results dictionary
        """
        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(improved_data, baseline_data, 
                                              alternative='two-sided')
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(baseline_data), len(improved_data)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        # Determine significance
        is_significant = p_value < self.alpha
        
        return {
            'test_type': 'mann_whitney_u',
            'statistic': statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'effect_size': effect_size,
            'confidence_level': self.confidence_level,
            'baseline_median': np.median(baseline_data),
            'improved_median': np.median(improved_data)
        }
    
    def bootstrap_confidence_interval(self, data: np.ndarray, 
                                    statistic_func: callable,
                                    n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence interval for a statistic.
        
        Args:
            data: Input data
            statistic_func: Function to calculate statistic
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Bootstrap results dictionary
        """
        # Generate bootstrap samples
        bootstrap_statistics = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_statistics.append(statistic_func(bootstrap_sample))
        
        bootstrap_statistics = np.array(bootstrap_statistics)
        
        # Calculate confidence interval
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_statistics, lower_percentile)
        ci_upper = np.percentile(bootstrap_statistics, upper_percentile)
        
        return {
            'test_type': 'bootstrap_confidence_interval',
            'confidence_interval': (ci_lower, ci_upper),
            'confidence_level': self.confidence_level,
            'bootstrap_mean': np.mean(bootstrap_statistics),
            'bootstrap_std': np.std(bootstrap_statistics, ddof=1),
            'n_bootstrap': n_bootstrap
        }
    
    def anova_test(self, groups_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Perform ANOVA test for multiple groups.
        
        Args:
            groups_data: Dictionary of group names to data arrays
            
        Returns:
            ANOVA test results
        """
        if len(groups_data) < 2:
            raise ValueError("ANOVA requires at least 2 groups")
        
        # Prepare data for ANOVA
        group_names = list(groups_data.keys())
        group_arrays = list(groups_data.values())
        
        # Perform ANOVA
        f_statistic, p_value = stats.f_oneway(*group_arrays)
        
        # Calculate effect size (eta squared)
        all_data = np.concatenate(group_arrays)
        grand_mean = np.mean(all_data)
        
        ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in group_arrays)
        ss_total = sum((all_data - grand_mean) ** 2)
        eta_squared = ss_between / ss_total
        
        # Determine significance
        is_significant = p_value < self.alpha
        
        return {
            'test_type': 'one_way_anova',
            'f_statistic': f_statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'effect_size': eta_squared,
            'confidence_level': self.confidence_level,
            'group_means': {name: np.mean(data) for name, data in groups_data.items()},
            'group_stds': {name: np.std(data, ddof=1) for name, data in groups_data.items()}
        }