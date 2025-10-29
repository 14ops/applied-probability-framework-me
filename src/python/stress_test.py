"""
Comprehensive Stress Testing Suite for Applied Probability Framework

This module provides extensive stress testing capabilities to validate
the framework's performance, stability, and reliability under various
conditions and loads.

Features:
- 50,000+ automated rounds across all strategies
- DRL agent integration and testing
- Performance monitoring and profiling
- Memory usage tracking
- Error detection and reporting
- Comprehensive result analysis
"""

import time
import psutil
import gc
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue

# Import framework components
from core.parallel_engine import ParallelSimulationEngine, SimulationConfig
from core.game_simulator import MinesGameSimulator
from character_strategies.takeshi_kovacs import TakeshiKovacsStrategy
from character_strategies.lelouch_vi_britannia import LelouchViBritanniaStrategy
from character_strategies.kazuya_kinoshita import KazuyaKinoshitaStrategy
from character_strategies.senku_ishigami import SenkuIshigamiStrategy
from character_strategies.hybrid_strategy import HybridStrategy
from character_strategies.rintaro_okabe import RintaroOkabeStrategy
from drl_agent import DRLAgent, train_drl_agent
from bayesian_mines import BayesianMinesInference
from performance_profiler import PerformanceProfiler
from performance_benchmark import PerformanceBenchmark


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    total_rounds: int = 50000
    rounds_per_batch: int = 1000
    max_parallel_workers: int = 8
    memory_limit_mb: int = 2048
    timeout_seconds: int = 3600
    enable_drl: bool = True
    enable_bayesian: bool = True
    enable_profiling: bool = True
    save_results: bool = True
    results_dir: str = "stress_test_results"


@dataclass
class StressTestResult:
    """Result of a stress test run."""
    strategy_name: str
    total_rounds: int
    successful_rounds: int
    failed_rounds: int
    avg_reward: float
    win_rate: float
    max_reward: float
    min_reward: float
    avg_duration: float
    max_duration: float
    min_duration: float
    memory_usage_mb: float
    cpu_usage_percent: float
    errors: List[str]
    timestamp: str


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]


class StressTestMonitor:
    """Monitor system performance during stress testing."""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.metrics: List[SystemMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start system monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                # Disk usage
                disk = psutil.disk_usage('/')
                
                # Network I/O
                network = psutil.net_io_counters()
                
                # Process count
                process_count = len(psutil.pids())
                
                # Load average (Unix only)
                try:
                    load_avg = list(psutil.getloadavg())
                except AttributeError:
                    load_avg = [0.0, 0.0, 0.0]
                
                # Create metrics
                metrics = SystemMetrics(
                    timestamp=datetime.now().isoformat(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_available_mb=memory.available / (1024 * 1024),
                    disk_usage_percent=disk.percent,
                    network_io={
                        'bytes_sent': network.bytes_sent,
                        'bytes_recv': network.bytes_recv,
                        'packets_sent': network.packets_sent,
                        'packets_recv': network.packets_recv
                    },
                    process_count=process_count,
                    load_average=load_avg
                )
                
                self.metrics.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.metrics) > 1000:
                    self.metrics = self.metrics[-1000:]
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.interval)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average system metrics."""
        if not self.metrics:
            return {}
        
        return {
            'avg_cpu_percent': np.mean([m.cpu_percent for m in self.metrics]),
            'max_cpu_percent': np.max([m.cpu_percent for m in self.metrics]),
            'avg_memory_percent': np.mean([m.memory_percent for m in self.metrics]),
            'max_memory_percent': np.max([m.memory_percent for m in self.metrics]),
            'avg_memory_available_mb': np.mean([m.memory_available_mb for m in self.metrics]),
            'min_memory_available_mb': np.min([m.memory_available_mb for m in self.metrics]),
            'avg_disk_usage_percent': np.mean([m.disk_usage_percent for m in self.metrics]),
            'max_disk_usage_percent': np.max([m.disk_usage_percent for m in self.metrics])
        }


class StressTestRunner:
    """Main stress testing runner."""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.results: List[StressTestResult] = []
        self.monitor = StressTestMonitor()
        self.profiler = PerformanceProfiler() if config.enable_profiling else None
        
        # Create results directory
        os.makedirs(config.results_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for stress testing."""
        log_file = os.path.join(self.config.results_dir, f"stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_stress_test(self) -> Dict[str, Any]:
        """Run comprehensive stress test."""
        logging.info("Starting comprehensive stress test...")
        logging.info(f"Configuration: {asdict(self.config)}")
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        try:
            # Test all strategies
            strategy_results = self._test_all_strategies()
            
            # Test DRL agent if enabled
            drl_results = None
            if self.config.enable_drl:
                drl_results = self._test_drl_agent()
            
            # Test Bayesian inference if enabled
            bayesian_results = None
            if self.config.enable_bayesian:
                bayesian_results = self._test_bayesian_inference()
            
            # Test system performance
            system_results = self._test_system_performance()
            
            # Compile results
            final_results = {
                'config': asdict(self.config),
                'strategy_results': strategy_results,
                'drl_results': drl_results,
                'bayesian_results': bayesian_results,
                'system_results': system_results,
                'monitor_metrics': self.monitor.get_average_metrics(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save results
            if self.config.save_results:
                self._save_results(final_results)
            
            # Generate report
            self._generate_report(final_results)
            
            return final_results
            
        except Exception as e:
            logging.error(f"Stress test failed: {e}")
            raise
        finally:
            # Stop monitoring
            self.monitor.stop_monitoring()
    
    def _test_all_strategies(self) -> Dict[str, StressTestResult]:
        """Test all character strategies."""
        logging.info("Testing all character strategies...")
        
        strategies = {
            'Takeshi Kovacs': TakeshiKovacsStrategy(),
            'Lelouch vi Britannia': LelouchViBritanniaStrategy(),
            'Kazuya Kinoshita': KazuyaKinoshitaStrategy(),
            'Senku Ishigami': SenkuIshigamiStrategy(),
            'Hybrid Strategy': HybridStrategy(),
            'Rintaro Okabe': RintaroOkabeStrategy()
        }
        
        results = {}
        
        for name, strategy in strategies.items():
            logging.info(f"Testing strategy: {name}")
            result = self._test_single_strategy(name, strategy)
            results[name] = result
            self.results.append(result)
        
        return results
    
    def _test_single_strategy(self, name: str, strategy) -> StressTestResult:
        """Test a single strategy."""
        start_time = time.time()
        errors = []
        
        try:
            # Create simulator
            simulator = MinesGameSimulator(board_size=5, mine_count=3)
            
            # Create simulation config
            config = SimulationConfig(
                simulation={'num_simulations': self.config.rounds_per_batch, 'parallel': True},
                system={'n_jobs': min(self.config.max_parallel_workers, 4)}
            )
            
            # Create parallel engine
            engine = ParallelSimulationEngine(config)
            
            # Run simulations in batches
            total_rewards = []
            total_wins = 0
            total_durations = []
            successful_rounds = 0
            failed_rounds = 0
            
            for batch in range(self.config.total_rounds // self.config.rounds_per_batch):
                try:
                    # Run batch
                    batch_start = time.time()
                    batch_results = engine.run_simulation(
                        simulator=simulator,
                        strategy=strategy,
                        num_simulations=self.config.rounds_per_batch
                    )
                    batch_duration = time.time() - batch_start
                    
                    # Collect results
                    total_rewards.extend([r.total_reward for r in batch_results.results])
                    total_wins += sum(1 for r in batch_results.results if r.win)
                    total_durations.append(batch_duration)
                    successful_rounds += len(batch_results.results)
                    
                    # Check memory usage
                    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
                    if memory_usage > self.config.memory_limit_mb:
                        logging.warning(f"Memory usage exceeded limit: {memory_usage:.2f} MB")
                        break
                    
                    # Log progress
                    if batch % 10 == 0:
                        logging.info(f"Strategy {name}: Batch {batch + 1}, "
                                   f"Win Rate: {total_wins / successful_rounds:.2%}, "
                                   f"Avg Reward: {np.mean(total_rewards):.2f}")
                
                except Exception as e:
                    error_msg = f"Batch {batch} failed: {str(e)}"
                    errors.append(error_msg)
                    failed_rounds += self.config.rounds_per_batch
                    logging.error(error_msg)
            
            # Calculate final metrics
            if total_rewards:
                avg_reward = np.mean(total_rewards)
                max_reward = np.max(total_rewards)
                min_reward = np.min(total_rewards)
                win_rate = total_wins / successful_rounds if successful_rounds > 0 else 0.0
            else:
                avg_reward = max_reward = min_reward = 0.0
                win_rate = 0.0
            
            avg_duration = np.mean(total_durations) if total_durations else 0.0
            max_duration = np.max(total_durations) if total_durations else 0.0
            min_duration = np.min(total_durations) if total_durations else 0.0
            
            # Get system metrics
            memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
            cpu_usage = psutil.cpu_percent()
            
            # Create result
            result = StressTestResult(
                strategy_name=name,
                total_rounds=successful_rounds + failed_rounds,
                successful_rounds=successful_rounds,
                failed_rounds=failed_rounds,
                avg_reward=avg_reward,
                win_rate=win_rate,
                max_reward=max_reward,
                min_reward=min_reward,
                avg_duration=avg_duration,
                max_duration=max_duration,
                min_duration=min_duration,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                errors=errors,
                timestamp=datetime.now().isoformat()
            )
            
            logging.info(f"Strategy {name} completed: "
                        f"Win Rate: {win_rate:.2%}, "
                        f"Avg Reward: {avg_reward:.2f}, "
                        f"Memory: {memory_usage:.2f} MB")
            
            return result
            
        except Exception as e:
            error_msg = f"Strategy {name} failed completely: {str(e)}"
            errors.append(error_msg)
            logging.error(error_msg)
            
            return StressTestResult(
                strategy_name=name,
                total_rounds=0,
                successful_rounds=0,
                failed_rounds=0,
                avg_reward=0.0,
                win_rate=0.0,
                max_reward=0.0,
                min_reward=0.0,
                avg_duration=0.0,
                max_duration=0.0,
                min_duration=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                errors=errors,
                timestamp=datetime.now().isoformat()
            )
    
    def _test_drl_agent(self) -> Dict[str, Any]:
        """Test DRL agent performance."""
        logging.info("Testing DRL agent...")
        
        try:
            # Train DRL agent
            logging.info("Training DRL agent...")
            agent = train_drl_agent(
                num_episodes=10000,  # Reduced for stress testing
                board_size=5,
                mine_count=3,
                strategy_type='analytical'
            )
            
            # Test trained agent
            logging.info("Testing trained DRL agent...")
            test_results = agent.evaluate(
                env=create_rl_environment(board_size=5, mine_count=3),
                num_episodes=1000
            )
            
            # Get performance summary
            performance = agent.get_performance_summary()
            
            return {
                'test_results': test_results,
                'performance_summary': performance,
                'training_successful': True
            }
            
        except Exception as e:
            logging.error(f"DRL agent testing failed: {e}")
            return {
                'test_results': None,
                'performance_summary': None,
                'training_successful': False,
                'error': str(e)
            }
    
    def _test_bayesian_inference(self) -> Dict[str, Any]:
        """Test Bayesian inference performance."""
        logging.info("Testing Bayesian inference...")
        
        try:
            # Create inference system
            inference = BayesianMinesInference(board_size=5, mine_count=3)
            
            # Test with various game states
            test_cases = [
                {
                    'revealed_cells': {(0, 0), (0, 1), (1, 0)},
                    'mine_positions': {(2, 2)},
                    'constraints': [
                        MineConstraint(
                            cell_position=(0, 0),
                            number=1,
                            adjacent_cells=[(0, 1), (1, 0), (1, 1)]
                        )
                    ]
                },
                {
                    'revealed_cells': {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2)},
                    'mine_positions': {(1, 1)},
                    'constraints': [
                        MineConstraint(
                            cell_position=(0, 1),
                            number=2,
                            adjacent_cells=[(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]
                        )
                    ]
                }
            ]
            
            results = []
            for i, test_case in enumerate(test_cases):
                # Update game state
                inference.update_game_state(
                    test_case['revealed_cells'],
                    test_case['mine_positions'],
                    test_case['constraints']
                )
                
                # Get performance stats
                stats = inference.get_performance_stats()
                results.append({
                    'test_case': i,
                    'stats': stats,
                    'safest_cells': inference.get_safest_cells(3),
                    'riskiest_cells': inference.get_riskiest_cells(3)
                })
            
            return {
                'test_results': results,
                'performance_stats': inference.get_performance_stats(),
                'inference_successful': True
            }
            
        except Exception as e:
            logging.error(f"Bayesian inference testing failed: {e}")
            return {
                'test_results': None,
                'performance_stats': None,
                'inference_successful': False,
                'error': str(e)
            }
    
    def _test_system_performance(self) -> Dict[str, Any]:
        """Test overall system performance."""
        logging.info("Testing system performance...")
        
        try:
            # Run performance benchmark
            benchmark = PerformanceBenchmark(num_runs=1000)
            benchmark_results = benchmark.run_all_benchmarks()
            
            # Test memory usage
            memory_test = self._test_memory_usage()
            
            # Test CPU usage
            cpu_test = self._test_cpu_usage()
            
            return {
                'benchmark_results': benchmark_results,
                'memory_test': memory_test,
                'cpu_test': cpu_test,
                'system_performance_successful': True
            }
            
        except Exception as e:
            logging.error(f"System performance testing failed: {e}")
            return {
                'benchmark_results': None,
                'memory_test': None,
                'cpu_test': None,
                'system_performance_successful': False,
                'error': str(e)
            }
    
    def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage patterns."""
        logging.info("Testing memory usage...")
        
        memory_usage = []
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Create and destroy objects to test memory management
        for i in range(100):
            # Create large objects
            large_list = [np.random.rand(1000, 1000) for _ in range(10)]
            
            # Record memory usage
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_usage.append(current_memory)
            
            # Force garbage collection
            del large_list
            gc.collect()
        
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        return {
            'start_memory_mb': start_memory,
            'end_memory_mb': end_memory,
            'max_memory_mb': np.max(memory_usage),
            'min_memory_mb': np.min(memory_usage),
            'avg_memory_mb': np.mean(memory_usage),
            'memory_growth_mb': end_memory - start_memory
        }
    
    def _test_cpu_usage(self) -> Dict[str, Any]:
        """Test CPU usage patterns."""
        logging.info("Testing CPU usage...")
        
        cpu_usage = []
        
        # Run CPU-intensive operations
        for i in range(100):
            # Matrix operations
            a = np.random.rand(1000, 1000)
            b = np.random.rand(1000, 1000)
            c = np.dot(a, b)
            
            # Record CPU usage
            cpu_percent = psutil.cpu_percent()
            cpu_usage.append(cpu_percent)
        
        return {
            'avg_cpu_percent': np.mean(cpu_usage),
            'max_cpu_percent': np.max(cpu_usage),
            'min_cpu_percent': np.min(cpu_usage),
            'std_cpu_percent': np.std(cpu_usage)
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save stress test results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        json_file = os.path.join(self.config.results_dir, f"stress_test_results_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save CSV results for strategies
        if 'strategy_results' in results:
            strategy_data = []
            for name, result in results['strategy_results'].items():
                strategy_data.append(asdict(result))
            
            df = pd.DataFrame(strategy_data)
            csv_file = os.path.join(self.config.results_dir, f"strategy_results_{timestamp}.csv")
            df.to_csv(csv_file, index=False)
        
        logging.info(f"Results saved to {self.config.results_dir}")
    
    def _generate_report(self, results: Dict[str, Any]):
        """Generate comprehensive stress test report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.config.results_dir, f"stress_test_report_{timestamp}.md")
        
        with open(report_file, 'w') as f:
            f.write("# Stress Test Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            
            # Configuration
            f.write("## Configuration\n\n")
            f.write(f"- Total Rounds: {self.config.total_rounds:,}\n")
            f.write(f"- Rounds per Batch: {self.config.rounds_per_batch:,}\n")
            f.write(f"- Max Parallel Workers: {self.config.max_parallel_workers}\n")
            f.write(f"- Memory Limit: {self.config.memory_limit_mb} MB\n")
            f.write(f"- DRL Enabled: {self.config.enable_drl}\n")
            f.write(f"- Bayesian Enabled: {self.config.enable_bayesian}\n\n")
            
            # Strategy Results
            if 'strategy_results' in results:
                f.write("## Strategy Results\n\n")
                f.write("| Strategy | Win Rate | Avg Reward | Max Reward | Memory (MB) | Errors |\n")
                f.write("|----------|----------|------------|------------|-------------|--------|\n")
                
                for name, result in results['strategy_results'].items():
                    f.write(f"| {name} | {result.win_rate:.2%} | {result.avg_reward:.2f} | "
                           f"{result.max_reward:.2f} | {result.memory_usage_mb:.1f} | {len(result.errors)} |\n")
                f.write("\n")
            
            # DRL Results
            if 'drl_results' in results and results['drl_results']:
                f.write("## DRL Agent Results\n\n")
                if results['drl_results']['training_successful']:
                    test_results = results['drl_results']['test_results']
                    f.write(f"- Win Rate: {test_results['win_rate']:.2%}\n")
                    f.write(f"- Average Reward: {test_results['avg_reward']:.2f}\n")
                    f.write(f"- Max Reward: {test_results['max_reward']:.2f}\n")
                else:
                    f.write(f"- Training Failed: {results['drl_results'].get('error', 'Unknown error')}\n")
                f.write("\n")
            
            # System Performance
            if 'system_results' in results and results['system_results']:
                f.write("## System Performance\n\n")
                if results['system_results']['system_performance_successful']:
                    memory_test = results['system_results']['memory_test']
                    cpu_test = results['system_results']['cpu_test']
                    f.write(f"- Memory Growth: {memory_test['memory_growth_mb']:.1f} MB\n")
                    f.write(f"- Max Memory: {memory_test['max_memory_mb']:.1f} MB\n")
                    f.write(f"- Average CPU: {cpu_test['avg_cpu_percent']:.1f}%\n")
                    f.write(f"- Max CPU: {cpu_test['max_cpu_percent']:.1f}%\n")
                else:
                    f.write(f"- System Performance Test Failed: {results['system_results'].get('error', 'Unknown error')}\n")
                f.write("\n")
            
            # Monitor Metrics
            if 'monitor_metrics' in results:
                f.write("## System Monitor Metrics\n\n")
                metrics = results['monitor_metrics']
                f.write(f"- Average CPU: {metrics.get('avg_cpu_percent', 0):.1f}%\n")
                f.write(f"- Max CPU: {metrics.get('max_cpu_percent', 0):.1f}%\n")
                f.write(f"- Average Memory: {metrics.get('avg_memory_percent', 0):.1f}%\n")
                f.write(f"- Max Memory: {metrics.get('max_memory_percent', 0):.1f}%\n")
                f.write(f"- Min Available Memory: {metrics.get('min_memory_available_mb', 0):.1f} MB\n")
                f.write("\n")
            
            # Summary
            f.write("## Summary\n\n")
            total_rounds = sum(r.total_rounds for r in self.results)
            successful_rounds = sum(r.successful_rounds for r in self.results)
            total_errors = sum(len(r.errors) for r in self.results)
            
            f.write(f"- Total Rounds Executed: {total_rounds:,}\n")
            f.write(f"- Successful Rounds: {successful_rounds:,}\n")
            f.write(f"- Success Rate: {successful_rounds/total_rounds:.2%}\n")
            f.write(f"- Total Errors: {total_errors}\n")
            f.write(f"- Error Rate: {total_errors/total_rounds:.4%}\n")
        
        logging.info(f"Report generated: {report_file}")


def main():
    """Main function for running stress tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run stress tests for Applied Probability Framework')
    parser.add_argument('--rounds', type=int, default=50000, help='Total number of rounds to run')
    parser.add_argument('--batch-size', type=int, default=1000, help='Rounds per batch')
    parser.add_argument('--workers', type=int, default=8, help='Maximum parallel workers')
    parser.add_argument('--memory-limit', type=int, default=2048, help='Memory limit in MB')
    parser.add_argument('--enable-drl', action='store_true', help='Enable DRL testing')
    parser.add_argument('--enable-bayesian', action='store_true', help='Enable Bayesian testing')
    parser.add_argument('--results-dir', type=str, default='stress_test_results', help='Results directory')
    
    args = parser.parse_args()
    
    # Create configuration
    config = StressTestConfig(
        total_rounds=args.rounds,
        rounds_per_batch=args.batch_size,
        max_parallel_workers=args.workers,
        memory_limit_mb=args.memory_limit,
        enable_drl=args.enable_drl,
        enable_bayesian=args.enable_bayesian,
        results_dir=args.results_dir
    )
    
    # Run stress test
    runner = StressTestRunner(config)
    results = runner.run_stress_test()
    
    print("\n" + "="*60)
    print("STRESS TEST COMPLETE")
    print("="*60)
    print(f"Total Rounds: {args.rounds:,}")
    print(f"Results Directory: {args.results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
