"""
Parallel simulation engine for Monte Carlo experiments.

This module provides efficient parallel execution of simulations using
multiprocessing or joblib, with proper seed management for reproducibility
and convergence tracking for adaptive termination.
"""

from typing import List, Callable, Any, Dict, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from dataclasses import dataclass
import time

from .base_simulator import BaseSimulator
from .base_strategy import BaseStrategy
from .config import SimulationConfig


@dataclass
class SimulationResult:
    """
    Results from a single simulation run.
    
    Attributes:
        run_id: Unique identifier for this run
        seed: Random seed used
        reward: Total reward obtained
        steps: Number of steps taken
        success: Whether simulation was successful
        metadata: Additional run-specific information
        duration: Execution time in seconds
    """
    run_id: int
    seed: int
    reward: float
    steps: int
    success: bool
    metadata: Dict[str, Any]
    duration: float


@dataclass
class AggregatedResults:
    """
    Aggregated results from multiple simulation runs.
    
    Attributes:
        num_runs: Number of completed runs
        mean_reward: Average reward across runs
        std_reward: Standard deviation of rewards
        confidence_interval: (lower, upper) bounds at specified confidence
        success_rate: Proportion of successful runs
        total_duration: Total execution time
        convergence_achieved: Whether convergence was detected
        all_results: List of individual SimulationResult objects
    """
    num_runs: int
    mean_reward: float
    std_reward: float
    confidence_interval: Tuple[float, float]
    success_rate: float
    total_duration: float
    convergence_achieved: bool
    all_results: List[SimulationResult]


def _run_single_simulation(args: Tuple[int, int, Any, Any, Dict]) -> SimulationResult:
    """
    Worker function for running a single simulation.
    
    This function is designed to be called by multiprocessing workers.
    It must be picklable, so it's a module-level function.
    
    Args:
        args: Tuple of (run_id, seed, simulator_class, strategy_class, config)
        
    Returns:
        SimulationResult object
    """
    run_id, seed, simulator, strategy, config = args
    
    start_time = time.time()
    
    # Initialize simulator and strategy with unique seed
    if hasattr(simulator, '__call__'):
        # It's a factory function
        sim = simulator(seed=seed)
    else:
        # It's an instance (make a copy with new seed)
        sim = simulator.__class__(seed=seed, config=simulator.config)
    
    if hasattr(strategy, '__call__'):
        strat = strategy()
    else:
        strat = strategy.__class__(name=strategy.name, config=strategy.config)
    
    # Run the episode
    total_reward = 0.0
    steps = 0
    success = False
    
    try:
        state = sim.reset()
        strat.reset()
        done = False
        
        max_steps = config.get('max_steps', 1000)
        
        while not done and steps < max_steps:
            valid_actions = sim.get_valid_actions()
            if not valid_actions:
                break
                
            action = strat.select_action(state, valid_actions)
            next_state, reward, done, info = sim.step(action)
            
            strat.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            steps += 1
            
        success = total_reward > 0 or info.get('success', False)
        
    except Exception as e:
        # Log error but don't crash entire batch
        success = False
        metadata = {'error': str(e)}
    else:
        metadata = sim.get_metadata()
    
    duration = time.time() - start_time
    
    return SimulationResult(
        run_id=run_id,
        seed=seed,
        reward=total_reward,
        steps=steps,
        success=success,
        metadata=metadata,
        duration=duration
    )


class ParallelSimulationEngine:
    """
    Engine for running Monte Carlo simulations in parallel.
    
    This class manages parallel execution of simulations with:
    - Proper seed management for reproducibility
    - Progress tracking and reporting
    - Convergence detection for adaptive termination
    - Result aggregation and statistical analysis
    
    Attributes:
        config: Simulation configuration
        n_jobs: Number of parallel workers
    """
    
    def __init__(self, config: SimulationConfig, n_jobs: Optional[int] = None):
        """
        Initialize the parallel engine.
        
        Args:
            config: Simulation configuration
            n_jobs: Number of parallel jobs. If None, uses config.n_jobs.
                   -1 means use all CPU cores.
        """
        self.config = config
        if n_jobs is None:
            n_jobs = config.n_jobs
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        self.n_jobs = max(1, n_jobs)
        
    def run_batch(self, 
                  simulator: BaseSimulator,
                  strategy: BaseStrategy,
                  num_runs: Optional[int] = None,
                  show_progress: bool = True) -> AggregatedResults:
        """
        Run a batch of simulations in parallel.
        
        Args:
            simulator: Simulator instance or factory function
            strategy: Strategy instance or factory function
            num_runs: Number of runs (uses config if None)
            show_progress: Whether to print progress updates
            
        Returns:
            AggregatedResults with statistics
        """
        if num_runs is None:
            num_runs = self.config.num_simulations
        
        # Generate unique seeds for reproducibility
        base_seed = self.config.seed if self.config.seed is not None else 42
        seeds = self._generate_seeds(base_seed, num_runs)
        
        # Prepare arguments for workers
        args_list = [
            (i, seed, simulator, strategy, {'max_steps': 1000})
            for i, seed in enumerate(seeds)
        ]
        
        start_time = time.time()
        results: List[SimulationResult] = []
        
        if self.config.parallel and self.n_jobs > 1:
            # Parallel execution
            results = self._run_parallel(args_list, show_progress)
        else:
            # Sequential execution
            results = self._run_sequential(args_list, show_progress)
        
        total_duration = time.time() - start_time
        
        # Check convergence if enabled
        convergence_achieved = False
        if self.config.convergence_check:
            convergence_achieved = self._check_convergence(results)
        
        # Aggregate results
        return self._aggregate_results(results, total_duration, convergence_achieved)
    
    def _run_parallel(self, args_list: List[Tuple], show_progress: bool) -> List[SimulationResult]:
        """Run simulations in parallel using ProcessPoolExecutor."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all jobs
            futures = {executor.submit(_run_single_simulation, args): args[0] 
                      for args in args_list}
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                
                if show_progress and completed % max(1, len(args_list) // 20) == 0:
                    pct = 100 * completed / len(args_list)
                    print(f"Progress: {completed}/{len(args_list)} ({pct:.1f}%)")
        
        # Sort by run_id to maintain order
        results.sort(key=lambda r: r.run_id)
        return results
    
    def _run_sequential(self, args_list: List[Tuple], show_progress: bool) -> List[SimulationResult]:
        """Run simulations sequentially."""
        results = []
        
        for i, args in enumerate(args_list):
            result = _run_single_simulation(args)
            results.append(result)
            
            if show_progress and (i + 1) % max(1, len(args_list) // 20) == 0:
                pct = 100 * (i + 1) / len(args_list)
                print(f"Progress: {i+1}/{len(args_list)} ({pct:.1f}%)")
        
        return results
    
    def _generate_seeds(self, base_seed: int, num_runs: int) -> List[int]:
        """
        Generate unique seeds for each run.
        
        Uses a deterministic method to ensure reproducibility while
        maintaining statistical independence between runs.
        """
        rng = np.random.default_rng(base_seed)
        return rng.integers(0, 2**31 - 1, size=num_runs).tolist()
    
    def _check_convergence(self, results: List[SimulationResult]) -> bool:
        """
        Check if simulation has converged.
        
        Uses a sliding window to compare recent means and determine if
        additional runs would significantly change the estimate.
        """
        if len(results) < 2 * self.config.convergence_window:
            return False
        
        # Extract recent rewards
        rewards = [r.reward for r in results[-2 * self.config.convergence_window:]]
        
        # Compare last two windows
        window_size = self.config.convergence_window
        mean1 = np.mean(rewards[:window_size])
        mean2 = np.mean(rewards[window_size:])
        
        relative_diff = abs(mean2 - mean1) / (abs(mean1) + 1e-10)
        
        return relative_diff < self.config.convergence_tolerance
    
    def _aggregate_results(self, 
                          results: List[SimulationResult],
                          total_duration: float,
                          convergence_achieved: bool) -> AggregatedResults:
        """Aggregate individual results into summary statistics."""
        rewards = np.array([r.reward for r in results])
        successes = np.array([r.success for r in results])
        
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards, ddof=1) if len(rewards) > 1 else 0.0)
        success_rate = float(np.mean(successes))
        
        # Calculate confidence interval
        if len(rewards) > 1:
            from scipy import stats
            ci = stats.t.interval(
                self.config.simulation.confidence_level if hasattr(self.config, 'simulation') else 0.95,
                len(rewards) - 1,
                loc=mean_reward,
                scale=stats.sem(rewards)
            )
        else:
            ci = (mean_reward, mean_reward)
        
        return AggregatedResults(
            num_runs=len(results),
            mean_reward=mean_reward,
            std_reward=std_reward,
            confidence_interval=ci,
            success_rate=success_rate,
            total_duration=total_duration,
            convergence_achieved=convergence_achieved,
            all_results=results
        )

