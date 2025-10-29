"""
Performance Profiling and Monitoring Module

This module provides comprehensive performance profiling capabilities for the
Applied Probability Framework, including:
- Function-level timing
- Memory usage monitoring
- Strategy performance analysis
- Bottleneck identification
- Performance regression detection
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import json
from datetime import datetime


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    function_name: str
    duration: float
    memory_usage: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    function_name: str
    call_count: int
    total_duration: float
    avg_duration: float
    min_duration: float
    max_duration: float
    std_duration: float
    total_memory: float
    avg_memory: float
    max_memory: float


class PerformanceProfiler:
    """
    High-performance profiler with minimal overhead.
    
    Features:
    - Decorator-based function profiling
    - Memory usage tracking
    - Statistical analysis
    - Performance regression detection
    - Export capabilities
    """
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.metrics: List[PerformanceMetric] = []
        self._lock = threading.Lock()
        self._enabled = True
        
        # Performance tracking
        self._function_stats: Dict[str, List[float]] = defaultdict(list)
        self._memory_stats: Dict[str, List[float]] = defaultdict(list)
        
    def profile(self, function_name: Optional[str] = None, 
                track_memory: bool = True,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Decorator for profiling function performance.
        
        Args:
            function_name: Custom name for the function (defaults to function.__name__)
            track_memory: Whether to track memory usage
            metadata: Additional metadata to store
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self._enabled:
                    return func(*args, **kwargs)
                
                # Get function name
                name = function_name or func.__name__
                
                # Start timing and memory tracking
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # Record metrics
                    end_time = time.perf_counter()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    
                    duration = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    metric = PerformanceMetric(
                        function_name=name,
                        duration=duration,
                        memory_usage=memory_delta if track_memory else 0.0,
                        timestamp=time.time(),
                        metadata=metadata or {}
                    )
                    
                    self._record_metric(metric)
            
            return wrapper
        return decorator
    
    def _record_metric(self, metric: PerformanceMetric):
        """Record a performance metric (thread-safe)."""
        with self._lock:
            self.metrics.append(metric)
            
            # Keep only recent metrics to prevent memory bloat
            if len(self.metrics) > self.max_samples:
                self.metrics = self.metrics[-self.max_samples:]
            
            # Update function statistics
            self._function_stats[metric.function_name].append(metric.duration)
            if metric.memory_usage > 0:
                self._memory_stats[metric.function_name].append(metric.memory_usage)
    
    def get_stats(self, function_name: Optional[str] = None) -> Dict[str, PerformanceStats]:
        """
        Get performance statistics.
        
        Args:
            function_name: Specific function to analyze (None for all)
            
        Returns:
            Dictionary mapping function names to performance stats
        """
        with self._lock:
            if function_name:
                functions = [function_name] if function_name in self._function_stats else []
            else:
                functions = list(self._function_stats.keys())
            
            stats = {}
            for func_name in functions:
                durations = self._function_stats[func_name]
                memories = self._memory_stats[func_name]
                
                if durations:
                    stats[func_name] = PerformanceStats(
                        function_name=func_name,
                        call_count=len(durations),
                        total_duration=sum(durations),
                        avg_duration=np.mean(durations),
                        min_duration=min(durations),
                        max_duration=max(durations),
                        std_duration=np.std(durations),
                        total_memory=sum(memories),
                        avg_memory=np.mean(memories) if memories else 0.0,
                        max_memory=max(memories) if memories else 0.0
                    )
            
            return stats
    
    def get_bottlenecks(self, top_n: int = 10) -> List[PerformanceStats]:
        """Get the top N performance bottlenecks by average duration."""
        stats = self.get_stats()
        return sorted(stats.values(), key=lambda x: x.avg_duration, reverse=True)[:top_n]
    
    def get_memory_hogs(self, top_n: int = 10) -> List[PerformanceStats]:
        """Get the top N memory consumers."""
        stats = self.get_stats()
        return sorted(stats.values(), key=lambda x: x.avg_memory, reverse=True)[:top_n]
    
    def detect_regressions(self, baseline_stats: Dict[str, PerformanceStats], 
                          threshold: float = 0.2) -> List[str]:
        """
        Detect performance regressions compared to baseline.
        
        Args:
            baseline_stats: Baseline performance statistics
            threshold: Performance degradation threshold (0.2 = 20% slower)
            
        Returns:
            List of function names with detected regressions
        """
        current_stats = self.get_stats()
        regressions = []
        
        for func_name, current in current_stats.items():
            if func_name in baseline_stats:
                baseline = baseline_stats[func_name]
                avg_duration_increase = (current.avg_duration - baseline.avg_duration) / baseline.avg_duration
                
                if avg_duration_increase > threshold:
                    regressions.append(f"{func_name}: {avg_duration_increase:.1%} slower")
        
        return regressions
    
    def export_report(self, filepath: str, format: str = 'json'):
        """Export performance report to file."""
        stats = self.get_stats()
        
        if format == 'json':
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_functions': len(stats),
                'total_calls': sum(s.call_count for s in stats.values()),
                'total_duration': sum(s.total_duration for s in stats.values()),
                'functions': {
                    name: {
                        'call_count': s.call_count,
                        'avg_duration': s.avg_duration,
                        'max_duration': s.max_duration,
                        'avg_memory': s.avg_memory,
                        'max_memory': s.max_memory
                    }
                    for name, s in stats.items()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
        
        elif format == 'csv':
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Function', 'Calls', 'Avg Duration', 'Max Duration', 
                               'Avg Memory', 'Max Memory'])
                for name, s in stats.items():
                    writer.writerow([name, s.call_count, s.avg_duration, s.max_duration,
                                   s.avg_memory, s.max_memory])
    
    def clear(self):
        """Clear all performance data."""
        with self._lock:
            self.metrics.clear()
            self._function_stats.clear()
            self._memory_stats.clear()
    
    def enable(self):
        """Enable profiling."""
        self._enabled = True
    
    def disable(self):
        """Disable profiling."""
        self._enabled = False


class StrategyProfiler:
    """
    Specialized profiler for strategy performance analysis.
    
    Tracks strategy-specific metrics like:
    - Decision time per action
    - Memory usage per strategy
    - EV calculation performance
    - Cache hit rates
    """
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.strategy_metrics: Dict[str, List[float]] = defaultdict(list)
        self.cache_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'hits': 0, 'misses': 0})
    
    def profile_strategy(self, strategy_name: str):
        """Decorator for profiling strategy methods."""
        return self.profiler.profile(
            function_name=f"strategy_{strategy_name}",
            track_memory=True,
            metadata={'type': 'strategy'}
        )
    
    def profile_ev_calculation(self, strategy_name: str):
        """Decorator for profiling EV calculations."""
        return self.profiler.profile(
            function_name=f"ev_calc_{strategy_name}",
            track_memory=False,
            metadata={'type': 'ev_calculation'}
        )
    
    def record_cache_hit(self, strategy_name: str, cache_type: str):
        """Record a cache hit."""
        self.cache_stats[strategy_name][cache_type]['hits'] += 1
    
    def record_cache_miss(self, strategy_name: str, cache_type: str):
        """Record a cache miss."""
        self.cache_stats[strategy_name][cache_type]['misses'] += 1
    
    def get_cache_hit_rate(self, strategy_name: str, cache_type: str) -> float:
        """Get cache hit rate for a strategy and cache type."""
        stats = self.cache_stats[strategy_name][cache_type]
        total = stats['hits'] + stats['misses']
        return stats['hits'] / total if total > 0 else 0.0
    
    def get_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """Get comprehensive performance metrics for a strategy."""
        stats = self.profiler.get_stats()
        strategy_stats = {}
        
        for func_name, stat in stats.items():
            if func_name.startswith(f"strategy_{strategy_name}"):
                strategy_stats[func_name] = stat
        
        return strategy_stats


# Global profiler instance
_global_profiler = PerformanceProfiler()
_strategy_profiler = StrategyProfiler(_global_profiler)


def profile_function(function_name: Optional[str] = None, track_memory: bool = True):
    """Convenience decorator for function profiling."""
    return _global_profiler.profile(function_name, track_memory)


def profile_strategy(strategy_name: str):
    """Convenience decorator for strategy profiling."""
    return _strategy_profiler.profile_strategy(strategy_name)


def get_performance_report() -> Dict[str, Any]:
    """Get current performance report."""
    return _global_profiler.get_stats()


def export_performance_report(filepath: str, format: str = 'json'):
    """Export performance report."""
    _global_profiler.export_report(filepath, format)


def clear_performance_data():
    """Clear all performance data."""
    _global_profiler.clear()
