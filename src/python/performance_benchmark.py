"""
Performance Benchmark Suite

Comprehensive benchmarking tools for the Applied Probability Framework.
Includes:
- Strategy performance comparison
- Memory usage analysis
- Scalability testing
- Performance regression detection
- Automated benchmark reporting
"""

import time
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from core.parallel_engine import ParallelSimulationEngine, SimulationConfig
from core.base_strategy import BaseStrategy
from performance_profiler import PerformanceProfiler, StrategyProfiler


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    duration: float
    memory_peak: float
    memory_avg: float
    cpu_usage: float
    iterations: int
    success_rate: float
    metadata: Dict[str, Any]


@dataclass
class StrategyBenchmark:
    """Benchmark results for a specific strategy."""
    strategy_name: str
    avg_decision_time: float
    memory_per_decision: float
    cache_hit_rate: float
    ev_calculation_time: float
    total_simulations: int
    win_rate: float
    avg_reward: float


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite.
    
    Features:
    - Strategy performance comparison
    - Memory usage profiling
    - Scalability testing
    - Regression detection
    - Automated reporting
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.strategy_results: Dict[str, StrategyBenchmark] = {}
        self.profiler = PerformanceProfiler()
        self.strategy_profiler = StrategyProfiler(self.profiler)
    
    def benchmark_strategy(self, strategy: BaseStrategy, 
                          simulator_factory: callable,
                          num_simulations: int = 1000,
                          warmup_runs: int = 100) -> StrategyBenchmark:
        """
        Benchmark a single strategy.
        
        Args:
            strategy: Strategy to benchmark
            simulator_factory: Factory function for creating simulators
            num_simulations: Number of simulations to run
            warmup_runs: Number of warmup runs to perform
            
        Returns:
            StrategyBenchmark with performance metrics
        """
        print(f"Benchmarking strategy: {strategy.name}")
        
        # Warmup runs
        print(f"  Running {warmup_runs} warmup simulations...")
        for _ in range(warmup_runs):
            sim = simulator_factory()
            sim.reset()
            strategy.reset()
            
            done = False
            while not done:
                valid_actions = sim.get_valid_actions()
                if not valid_actions:
                    break
                action = strategy.select_action(sim.get_state(), valid_actions)
                if action is None:
                    break
                _, _, done, _ = sim.step(action)
        
        # Clear profiler data after warmup
        self.profiler.clear()
        
        # Actual benchmark
        print(f"  Running {num_simulations} benchmark simulations...")
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        decision_times = []
        ev_times = []
        rewards = []
        wins = 0
        
        for i in range(num_simulations):
            sim = simulator_factory()
            sim.reset()
            strategy.reset()
            
            sim_start = time.perf_counter()
            done = False
            steps = 0
            
            while not done and steps < 1000:  # Prevent infinite loops
                valid_actions = sim.get_valid_actions()
                if not valid_actions:
                    break
                
                # Time decision making
                decision_start = time.perf_counter()
                action = strategy.select_action(sim.get_state(), valid_actions)
                decision_time = time.perf_counter() - decision_start
                decision_times.append(decision_time)
                
                if action is None:
                    break
                
                _, reward, done, _ = sim.step(action)
                rewards.append(reward)
                steps += 1
            
            if reward > 0:
                wins += 1
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        total_duration = end_time - start_time
        memory_used = end_memory - start_memory
        win_rate = wins / num_simulations
        avg_reward = np.mean(rewards) if rewards else 0.0
        
        # Get profiler stats
        profiler_stats = self.profiler.get_stats()
        ev_calc_time = 0.0
        cache_hit_rate = 0.0
        
        for func_name, stats in profiler_stats.items():
            if 'ev_calc' in func_name:
                ev_calc_time = stats.avg_duration
            if 'cache' in func_name.lower():
                # Calculate cache hit rate from profiler data
                pass
        
        result = StrategyBenchmark(
            strategy_name=strategy.name,
            avg_decision_time=np.mean(decision_times) if decision_times else 0.0,
            memory_per_decision=memory_used / num_simulations,
            cache_hit_rate=cache_hit_rate,
            ev_calculation_time=ev_calc_time,
            total_simulations=num_simulations,
            win_rate=win_rate,
            avg_reward=avg_reward
        )
        
        self.strategy_results[strategy.name] = result
        return result
    
    def benchmark_parallel_engine(self, 
                                 simulator_factory: callable,
                                 strategy_factory: callable,
                                 num_simulations: int = 1000,
                                 worker_counts: List[int] = [1, 2, 4, 8]) -> Dict[int, BenchmarkResult]:
        """
        Benchmark parallel engine performance with different worker counts.
        
        Args:
            simulator_factory: Factory for creating simulators
            strategy_factory: Factory for creating strategies
            num_simulations: Number of simulations per worker count
            worker_counts: List of worker counts to test
            
        Returns:
            Dictionary mapping worker counts to benchmark results
        """
        print("Benchmarking parallel engine performance...")
        results = {}
        
        for n_workers in worker_counts:
            print(f"  Testing with {n_workers} workers...")
            
            # Create parallel engine
            engine = ParallelSimulationEngine(self.config, n_jobs=n_workers)
            
            # Benchmark
            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Run simulations
            aggregated = engine.run_batch(
                simulator=simulator_factory(),
                strategy=strategy_factory(),
                num_runs=num_simulations,
                show_progress=False
            )
            
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Calculate metrics
            duration = end_time - start_time
            memory_peak = end_memory - start_memory
            success_rate = aggregated.success_rate
            
            result = BenchmarkResult(
                name=f"parallel_{n_workers}_workers",
                duration=duration,
                memory_peak=memory_peak,
                memory_avg=memory_peak / num_simulations,
                cpu_usage=psutil.cpu_percent(),
                iterations=num_simulations,
                success_rate=success_rate,
                metadata={
                    'n_workers': n_workers,
                    'mean_reward': aggregated.mean_reward,
                    'std_reward': aggregated.std_reward
                }
            )
            
            results[n_workers] = result
            self.results.append(result)
        
        return results
    
    def benchmark_memory_usage(self, 
                              strategy: BaseStrategy,
                              simulator_factory: callable,
                              max_simulations: int = 10000,
                              step_size: int = 1000) -> List[BenchmarkResult]:
        """
        Benchmark memory usage scaling with simulation count.
        
        Args:
            strategy: Strategy to test
            simulator_factory: Factory for creating simulators
            max_simulations: Maximum number of simulations
            step_size: Step size for testing different simulation counts
            
        Returns:
            List of benchmark results for different simulation counts
        """
        print("Benchmarking memory usage scaling...")
        results = []
        
        for n_sims in range(step_size, max_simulations + 1, step_size):
            print(f"  Testing with {n_sims} simulations...")
            
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            start_time = time.perf_counter()
            
            # Run simulations
            for _ in range(n_sims):
                sim = simulator_factory()
                sim.reset()
                strategy.reset()
                
                done = False
                while not done:
                    valid_actions = sim.get_valid_actions()
                    if not valid_actions:
                        break
                    action = strategy.select_action(sim.get_state(), valid_actions)
                    if action is None:
                        break
                    _, _, done, _ = sim.step(action)
            
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            end_time = time.perf_counter()
            
            result = BenchmarkResult(
                name=f"memory_{n_sims}_sims",
                duration=end_time - start_time,
                memory_peak=end_memory - start_memory,
                memory_avg=(end_memory - start_memory) / n_sims,
                cpu_usage=psutil.cpu_percent(),
                iterations=n_sims,
                success_rate=1.0,  # Assume all succeed for memory test
                metadata={'simulation_count': n_sims}
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def compare_strategies(self, 
                          strategies: List[BaseStrategy],
                          simulator_factory: callable,
                          num_simulations: int = 1000) -> Dict[str, StrategyBenchmark]:
        """
        Compare performance of multiple strategies.
        
        Args:
            strategies: List of strategies to compare
            simulator_factory: Factory for creating simulators
            num_simulations: Number of simulations per strategy
            
        Returns:
            Dictionary mapping strategy names to benchmark results
        """
        print("Comparing strategy performance...")
        results = {}
        
        for strategy in strategies:
            result = self.benchmark_strategy(strategy, simulator_factory, num_simulations)
            results[strategy.name] = result
        
        return results
    
    def generate_report(self, output_file: str = "performance_report.json"):
        """Generate comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': mp.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}"
            },
            'strategy_benchmarks': {
                name: {
                    'avg_decision_time': result.avg_decision_time,
                    'memory_per_decision': result.memory_per_decision,
                    'cache_hit_rate': result.cache_hit_rate,
                    'ev_calculation_time': result.ev_calculation_time,
                    'win_rate': result.win_rate,
                    'avg_reward': result.avg_reward
                }
                for name, result in self.strategy_results.items()
            },
            'general_benchmarks': [
                {
                    'name': result.name,
                    'duration': result.duration,
                    'memory_peak': result.memory_peak,
                    'memory_avg': result.memory_avg,
                    'cpu_usage': result.cpu_usage,
                    'iterations': result.iterations,
                    'success_rate': result.success_rate,
                    'metadata': result.metadata
                }
                for result in self.results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Performance report saved to {output_file}")
        return report
    
    def plot_performance_comparison(self, output_file: str = "performance_comparison.png"):
        """Generate performance comparison plots."""
        if not self.strategy_results:
            print("No strategy results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        names = list(self.strategy_results.keys())
        decision_times = [r.avg_decision_time for r in self.strategy_results.values()]
        memory_usage = [r.memory_per_decision for r in self.strategy_results.values()]
        win_rates = [r.win_rate for r in self.strategy_results.values()]
        rewards = [r.avg_reward for r in self.strategy_results.values()]
        
        # Decision time comparison
        axes[0, 0].bar(names, decision_times)
        axes[0, 0].set_title('Average Decision Time by Strategy')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        axes[0, 1].bar(names, memory_usage)
        axes[0, 1].set_title('Memory Usage per Decision')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Win rate comparison
        axes[1, 0].bar(names, win_rates)
        axes[1, 0].set_title('Win Rate by Strategy')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Average reward comparison
        axes[1, 1].bar(names, rewards)
        axes[1, 1].set_title('Average Reward by Strategy')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance comparison plot saved to {output_file}")


def run_comprehensive_benchmark(config: SimulationConfig,
                               strategies: List[BaseStrategy],
                               simulator_factory: callable,
                               output_dir: str = "benchmark_results") -> Dict[str, Any]:
    """
    Run comprehensive performance benchmark suite.
    
    Args:
        config: Simulation configuration
        strategies: List of strategies to benchmark
        simulator_factory: Factory for creating simulators
        output_dir: Directory to save results
        
    Returns:
        Comprehensive benchmark results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    benchmark = PerformanceBenchmark(config)
    
    # Strategy comparison
    print("=== Strategy Performance Comparison ===")
    strategy_results = benchmark.compare_strategies(strategies, simulator_factory)
    
    # Parallel engine benchmark
    print("\n=== Parallel Engine Performance ===")
    parallel_results = benchmark.benchmark_parallel_engine(
        simulator_factory, 
        lambda: strategies[0],  # Use first strategy for parallel test
        num_simulations=1000
    )
    
    # Memory usage benchmark
    print("\n=== Memory Usage Scaling ===")
    memory_results = benchmark.benchmark_memory_usage(
        strategies[0],  # Use first strategy for memory test
        simulator_factory,
        max_simulations=5000
    )
    
    # Generate reports
    report_file = os.path.join(output_dir, "performance_report.json")
    plot_file = os.path.join(output_dir, "performance_comparison.png")
    
    benchmark.generate_report(report_file)
    benchmark.plot_performance_comparison(plot_file)
    
    return {
        'strategy_results': strategy_results,
        'parallel_results': parallel_results,
        'memory_results': memory_results,
        'report_file': report_file,
        'plot_file': plot_file
    }
