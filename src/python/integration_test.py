"""
Comprehensive Integration Test for Applied Probability Framework

This module provides end-to-end integration testing for all framework
components, ensuring they work together seamlessly and meet performance
requirements.

Features:
- End-to-end workflow testing
- Component integration validation
- Performance benchmarking
- Error detection and reporting
- Comprehensive result analysis
"""

import time
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

# Import all framework components
from core.parallel_engine import ParallelSimulationEngine, SimulationConfig
from core.game_simulator import MinesGameSimulator
from character_strategies.takeshi_kovacs import TakeshiKovacsStrategy
from character_strategies.lelouch_vi_britannia import LelouchViBritanniaStrategy
from character_strategies.kazuya_kinoshita import KazuyaKinoshitaStrategy
from character_strategies.senku_ishigami import SenkuIshigamiStrategy
from character_strategies.hybrid_strategy import HybridStrategy
from character_strategies.rintaro_okabe import RintaroOkabeStrategy
from drl_agent import DRLAgent, train_drl_agent
from bayesian_mines import BayesianMinesInference, MineConstraint
from gan_human_simulation import create_wgan_human_simulation, create_human_pattern_dataset
from performance_profiler import PerformanceProfiler
from performance_benchmark import PerformanceBenchmark
from stress_test import StressTestRunner, StressTestConfig


@dataclass
class IntegrationTestResult:
    """Result of an integration test."""
    test_name: str
    success: bool
    duration: float
    error_message: Optional[str]
    metrics: Dict[str, Any]
    timestamp: str


class IntegrationTestSuite:
    """Comprehensive integration test suite."""
    
    def __init__(self, results_dir: str = "integration_test_results"):
        self.results_dir = results_dir
        self.results: List[IntegrationTestResult] = []
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for integration testing."""
        log_file = os.path.join(self.results_dir, f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logging.info("Starting comprehensive integration test suite...")
        
        test_results = {}
        
        try:
            # Test 1: Basic Framework Integration
            test_results['basic_integration'] = self._test_basic_integration()
            
            # Test 2: Character Strategy Integration
            test_results['strategy_integration'] = self._test_strategy_integration()
            
            # Test 3: DRL Agent Integration
            test_results['drl_integration'] = self._test_drl_integration()
            
            # Test 4: Bayesian Inference Integration
            test_results['bayesian_integration'] = self._test_bayesian_integration()
            
            # Test 5: GAN Human Simulation Integration
            test_results['gan_integration'] = self._test_gan_integration()
            
            # Test 6: Performance Integration
            test_results['performance_integration'] = self._test_performance_integration()
            
            # Test 7: End-to-End Workflow
            test_results['end_to_end_workflow'] = self._test_end_to_end_workflow()
            
            # Test 8: Stress Test Integration
            test_results['stress_test_integration'] = self._test_stress_integration()
            
            # Compile final results
            final_results = {
                'test_results': test_results,
                'summary': self._generate_summary(test_results),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save results
            self._save_results(final_results)
            
            # Generate report
            self._generate_report(final_results)
            
            return final_results
            
        except Exception as e:
            logging.error(f"Integration test suite failed: {e}")
            raise
    
    def _test_basic_integration(self) -> IntegrationTestResult:
        """Test basic framework integration."""
        test_name = "Basic Framework Integration"
        start_time = time.time()
        
        try:
            logging.info(f"Running {test_name}...")
            
            # Create basic components
            simulator = MinesGameSimulator(board_size=5, mine_count=3)
            strategy = SenkuIshigamiStrategy()
            config = SimulationConfig(
                simulation={'num_simulations': 100, 'parallel': True},
                system={'n_jobs': 2}
            )
            engine = ParallelSimulationEngine(config)
            
            # Run basic simulation
            results = engine.run_simulation(
                simulator=simulator,
                strategy=strategy,
                num_simulations=100
            )
            
            # Validate results
            assert len(results.results) == 100, "Expected 100 simulation results"
            assert results.win_rate >= 0.0 and results.win_rate <= 1.0, "Invalid win rate"
            assert results.avg_reward >= 0.0, "Invalid average reward"
            
            duration = time.time() - start_time
            
            metrics = {
                'win_rate': results.win_rate,
                'avg_reward': results.avg_reward,
                'max_reward': results.max_reward,
                'min_reward': results.min_reward,
                'simulation_count': len(results.results)
            }
            
            logging.info(f"{test_name} completed successfully in {duration:.2f}s")
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                duration=duration,
                error_message=None,
                metrics=metrics,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{test_name} failed: {str(e)}"
            logging.error(error_msg)
            
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error_message=error_msg,
                metrics={},
                timestamp=datetime.now().isoformat()
            )
    
    def _test_strategy_integration(self) -> IntegrationTestResult:
        """Test character strategy integration."""
        test_name = "Character Strategy Integration"
        start_time = time.time()
        
        try:
            logging.info(f"Running {test_name}...")
            
            # Test all strategies
            strategies = {
                'Takeshi Kovacs': TakeshiKovacsStrategy(),
                'Lelouch vi Britannia': LelouchViBritanniaStrategy(),
                'Kazuya Kinoshita': KazuyaKinoshitaStrategy(),
                'Senku Ishigami': SenkuIshigamiStrategy(),
                'Hybrid Strategy': HybridStrategy(),
                'Rintaro Okabe': RintaroOkabeStrategy()
            }
            
            simulator = MinesGameSimulator(board_size=5, mine_count=3)
            config = SimulationConfig(
                simulation={'num_simulations': 50, 'parallel': True},
                system={'n_jobs': 2}
            )
            engine = ParallelSimulationEngine(config)
            
            strategy_results = {}
            for name, strategy in strategies.items():
                results = engine.run_simulation(
                    simulator=simulator,
                    strategy=strategy,
                    num_simulations=50
                )
                strategy_results[name] = {
                    'win_rate': results.win_rate,
                    'avg_reward': results.avg_reward,
                    'max_reward': results.max_reward
                }
            
            # Validate all strategies worked
            assert len(strategy_results) == 6, "Expected 6 strategy results"
            for name, results in strategy_results.items():
                assert results['win_rate'] >= 0.0 and results['win_rate'] <= 1.0, f"Invalid win rate for {name}"
                assert results['avg_reward'] >= 0.0, f"Invalid avg reward for {name}"
            
            duration = time.time() - start_time
            
            metrics = {
                'strategy_count': len(strategy_results),
                'strategy_results': strategy_results
            }
            
            logging.info(f"{test_name} completed successfully in {duration:.2f}s")
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                duration=duration,
                error_message=None,
                metrics=metrics,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{test_name} failed: {str(e)}"
            logging.error(error_msg)
            
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error_message=error_msg,
                metrics={},
                timestamp=datetime.now().isoformat()
            )
    
    def _test_drl_integration(self) -> IntegrationTestResult:
        """Test DRL agent integration."""
        test_name = "DRL Agent Integration"
        start_time = time.time()
        
        try:
            logging.info(f"Running {test_name}...")
            
            # Create DRL environment
            from drl_environment import create_rl_environment
            env = create_rl_environment(board_size=5, mine_count=3)
            
            # Create DRL agent
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            
            agent = DRLAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=0.001,
                gamma=0.99,
                epsilon_start=0.9,
                epsilon_end=0.01,
                epsilon_decay=0.995,
                batch_size=32,
                memory_size=10000,
                target_update_freq=100
            )
            
            # Train agent for a few episodes
            training_episodes = 100
            for episode in range(training_episodes):
                episode_stats = agent.train_episode(env)
                if episode % 20 == 0:
                    logging.info(f"DRL Training Episode {episode}: Reward = {episode_stats['total_reward']:.2f}")
            
            # Evaluate agent
            eval_results = agent.evaluate(env, num_episodes=50)
            
            # Validate results
            assert eval_results['avg_reward'] >= 0.0, "Invalid average reward"
            assert eval_results['win_rate'] >= 0.0 and eval_results['win_rate'] <= 1.0, "Invalid win rate"
            
            duration = time.time() - start_time
            
            metrics = {
                'training_episodes': training_episodes,
                'eval_episodes': 50,
                'avg_reward': eval_results['avg_reward'],
                'win_rate': eval_results['win_rate'],
                'max_reward': eval_results['max_reward'],
                'min_reward': eval_results['min_reward']
            }
            
            logging.info(f"{test_name} completed successfully in {duration:.2f}s")
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                duration=duration,
                error_message=None,
                metrics=metrics,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{test_name} failed: {str(e)}"
            logging.error(error_msg)
            
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error_message=error_msg,
                metrics={},
                timestamp=datetime.now().isoformat()
            )
    
    def _test_bayesian_integration(self) -> IntegrationTestResult:
        """Test Bayesian inference integration."""
        test_name = "Bayesian Inference Integration"
        start_time = time.time()
        
        try:
            logging.info(f"Running {test_name}...")
            
            # Create Bayesian inference system
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
            
            test_results = []
            for i, test_case in enumerate(test_cases):
                # Update game state
                inference.update_game_state(
                    test_case['revealed_cells'],
                    test_case['mine_positions'],
                    test_case['constraints']
                )
                
                # Get probability estimates
                safest_cells = inference.get_safest_cells(3)
                riskiest_cells = inference.get_riskiest_cells(3)
                prob_map = inference.get_probability_map()
                uncertainty_map = inference.get_uncertainty_map()
                
                test_results.append({
                    'test_case': i,
                    'safest_cells': safest_cells,
                    'riskiest_cells': riskiest_cells,
                    'prob_map_shape': prob_map.shape,
                    'uncertainty_map_shape': uncertainty_map.shape
                })
            
            # Validate results
            assert len(test_results) == 2, "Expected 2 test case results"
            for result in test_results:
                assert len(result['safest_cells']) <= 3, "Too many safest cells"
                assert len(result['riskiest_cells']) <= 3, "Too many riskiest cells"
                assert result['prob_map_shape'] == (5, 5), "Invalid probability map shape"
                assert result['uncertainty_map_shape'] == (5, 5), "Invalid uncertainty map shape"
            
            duration = time.time() - start_time
            
            metrics = {
                'test_cases': len(test_results),
                'test_results': test_results,
                'performance_stats': inference.get_performance_stats()
            }
            
            logging.info(f"{test_name} completed successfully in {duration:.2f}s")
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                duration=duration,
                error_message=None,
                metrics=metrics,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{test_name} failed: {str(e)}"
            logging.error(error_msg)
            
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error_message=error_msg,
                metrics={},
                timestamp=datetime.now().isoformat()
            )
    
    def _test_gan_integration(self) -> IntegrationTestResult:
        """Test GAN human simulation integration."""
        test_name = "GAN Human Simulation Integration"
        start_time = time.time()
        
        try:
            logging.info(f"Running {test_name}...")
            
            # Create WGAN
            wgan = create_wgan_human_simulation()
            
            # Create training data
            human_patterns = create_human_pattern_dataset(100)
            
            # Train WGAN for a few epochs
            training_history = wgan.train(human_patterns, num_epochs=10, batch_size=16)
            
            # Generate test patterns
            test_patterns = wgan.generate_pattern(
                behavior_type=2,  # analytical
                experience_level=1,  # intermediate
                emotional_state=0,  # calm
                batch_size=5
            )
            
            # Evaluate authenticity
            authenticity_scores = wgan.evaluate_authenticity(test_patterns)
            
            # Validate results
            assert len(test_patterns) == 5, "Expected 5 generated patterns"
            assert len(authenticity_scores) == 5, "Expected 5 authenticity scores"
            for score in authenticity_scores:
                assert 0.0 <= score <= 1.0, "Invalid authenticity score"
            
            duration = time.time() - start_time
            
            metrics = {
                'training_epochs': 10,
                'training_patterns': len(human_patterns),
                'generated_patterns': len(test_patterns),
                'avg_authenticity': np.mean(authenticity_scores),
                'training_history': {
                    'generator_losses': len(training_history['generator_losses']),
                    'discriminator_losses': len(training_history['discriminator_losses']),
                    'wasserstein_distances': len(training_history['wasserstein_distances'])
                }
            }
            
            logging.info(f"{test_name} completed successfully in {duration:.2f}s")
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                duration=duration,
                error_message=None,
                metrics=metrics,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{test_name} failed: {str(e)}"
            logging.error(error_msg)
            
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error_message=error_msg,
                metrics={},
                timestamp=datetime.now().isoformat()
            )
    
    def _test_performance_integration(self) -> IntegrationTestResult:
        """Test performance monitoring integration."""
        test_name = "Performance Integration"
        start_time = time.time()
        
        try:
            logging.info(f"Running {test_name}...")
            
            # Create performance profiler
            profiler = PerformanceProfiler()
            
            # Create performance benchmark
            benchmark = PerformanceBenchmark(num_runs=100)
            
            # Run benchmarks
            benchmark_results = benchmark.run_all_benchmarks()
            
            # Profile a simple operation
            profiler.start()
            
            # Simulate some work
            simulator = MinesGameSimulator(board_size=5, mine_count=3)
            strategy = SenkuIshigamiStrategy()
            config = SimulationConfig(
                simulation={'num_simulations': 50, 'parallel': True},
                system={'n_jobs': 2}
            )
            engine = ParallelSimulationEngine(config)
            
            results = engine.run_simulation(
                simulator=simulator,
                strategy=strategy,
                num_simulations=50
            )
            
            profiler.stop()
            
            # Get performance report
            performance_report = profiler.report()
            memory_usage = profiler.monitor_memory()
            
            # Validate results
            assert len(benchmark_results) > 0, "No benchmark results"
            assert performance_report is not None, "No performance report"
            assert memory_usage is not None, "No memory usage data"
            
            duration = time.time() - start_time
            
            metrics = {
                'benchmark_results': benchmark_results,
                'memory_usage_mb': memory_usage['rss_mb'],
                'performance_report_length': len(performance_report),
                'simulation_results': {
                    'win_rate': results.win_rate,
                    'avg_reward': results.avg_reward
                }
            }
            
            logging.info(f"{test_name} completed successfully in {duration:.2f}s")
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                duration=duration,
                error_message=None,
                metrics=metrics,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{test_name} failed: {str(e)}"
            logging.error(error_msg)
            
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error_message=error_msg,
                metrics={},
                timestamp=datetime.now().isoformat()
            )
    
    def _test_end_to_end_workflow(self) -> IntegrationTestResult:
        """Test complete end-to-end workflow."""
        test_name = "End-to-End Workflow"
        start_time = time.time()
        
        try:
            logging.info(f"Running {test_name}...")
            
            # Step 1: Create game environment
            simulator = MinesGameSimulator(board_size=5, mine_count=3)
            
            # Step 2: Create strategy
            strategy = SenkuIshigamiStrategy()
            
            # Step 3: Create Bayesian inference
            inference = BayesianMinesInference(board_size=5, mine_count=3)
            
            # Step 4: Run simulation with Bayesian integration
            config = SimulationConfig(
                simulation={'num_simulations': 100, 'parallel': True},
                system={'n_jobs': 2}
            )
            engine = ParallelSimulationEngine(config)
            
            results = engine.run_simulation(
                simulator=simulator,
                strategy=strategy,
                num_simulations=100
            )
            
            # Step 5: Analyze results with Bayesian inference
            # Simulate some revealed cells for analysis
            revealed_cells = {(0, 0), (0, 1), (1, 0)}
            mine_positions = {(2, 2)}
            constraints = [
                MineConstraint(
                    cell_position=(0, 0),
                    number=1,
                    adjacent_cells=[(0, 1), (1, 0), (1, 1)]
                )
            ]
            
            inference.update_game_state(revealed_cells, mine_positions, constraints)
            safest_cells = inference.get_safest_cells(3)
            riskiest_cells = inference.get_riskiest_cells(3)
            
            # Step 6: Generate human-like patterns
            wgan = create_wgan_human_simulation()
            human_patterns = create_human_pattern_dataset(10)
            wgan.train(human_patterns, num_epochs=5, batch_size=8)
            generated_patterns = wgan.generate_pattern(batch_size=3)
            
            # Validate end-to-end workflow
            assert results.win_rate >= 0.0 and results.win_rate <= 1.0, "Invalid win rate"
            assert len(safest_cells) <= 3, "Too many safest cells"
            assert len(riskiest_cells) <= 3, "Too many riskiest cells"
            assert len(generated_patterns) == 3, "Expected 3 generated patterns"
            
            duration = time.time() - start_time
            
            metrics = {
                'simulation_results': {
                    'win_rate': results.win_rate,
                    'avg_reward': results.avg_reward,
                    'max_reward': results.max_reward
                },
                'bayesian_analysis': {
                    'safest_cells': safest_cells,
                    'riskiest_cells': riskiest_cells
                },
                'human_simulation': {
                    'generated_patterns': len(generated_patterns),
                    'training_patterns': len(human_patterns)
                }
            }
            
            logging.info(f"{test_name} completed successfully in {duration:.2f}s")
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                duration=duration,
                error_message=None,
                metrics=metrics,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{test_name} failed: {str(e)}"
            logging.error(error_msg)
            
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error_message=error_msg,
                metrics={},
                timestamp=datetime.now().isoformat()
            )
    
    def _test_stress_integration(self) -> IntegrationTestResult:
        """Test stress testing integration."""
        test_name = "Stress Test Integration"
        start_time = time.time()
        
        try:
            logging.info(f"Running {test_name}...")
            
            # Create stress test configuration
            stress_config = StressTestConfig(
                total_rounds=1000,  # Reduced for integration testing
                rounds_per_batch=100,
                max_parallel_workers=2,
                memory_limit_mb=1024,
                enable_drl=False,  # Disable for faster testing
                enable_bayesian=False,  # Disable for faster testing
                results_dir=os.path.join(self.results_dir, "stress_test")
            )
            
            # Run stress test
            stress_runner = StressTestRunner(stress_config)
            stress_results = stress_runner.run_stress_test()
            
            # Validate stress test results
            assert 'strategy_results' in stress_results, "No strategy results in stress test"
            assert len(stress_results['strategy_results']) > 0, "No strategy results"
            
            duration = time.time() - start_time
            
            metrics = {
                'total_rounds': stress_config.total_rounds,
                'strategy_count': len(stress_results['strategy_results']),
                'stress_test_successful': True
            }
            
            logging.info(f"{test_name} completed successfully in {duration:.2f}s")
            
            return IntegrationTestResult(
                test_name=test_name,
                success=True,
                duration=duration,
                error_message=None,
                metrics=metrics,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{test_name} failed: {str(e)}"
            logging.error(error_msg)
            
            return IntegrationTestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error_message=error_msg,
                metrics={},
                timestamp=datetime.now().isoformat()
            )
    
    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test summary."""
        total_tests = len(test_results)
        successful_tests = sum(1 for result in test_results.values() if result.success)
        failed_tests = total_tests - successful_tests
        
        total_duration = sum(result.duration for result in test_results.values())
        avg_duration = total_duration / total_tests if total_tests > 0 else 0.0
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0.0,
            'total_duration': total_duration,
            'avg_duration': avg_duration
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save integration test results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        json_file = os.path.join(self.results_dir, f"integration_test_results_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save CSV results
        test_data = []
        for test_name, result in results['test_results'].items():
            test_data.append({
                'test_name': test_name,
                'success': result.success,
                'duration': result.duration,
                'error_message': result.error_message,
                'timestamp': result.timestamp
            })
        
        df = pd.DataFrame(test_data)
        csv_file = os.path.join(self.results_dir, f"integration_test_results_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        
        logging.info(f"Results saved to {self.results_dir}")
    
    def _generate_report(self, results: Dict[str, Any]):
        """Generate integration test report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.results_dir, f"integration_test_report_{timestamp}.md")
        
        with open(report_file, 'w') as f:
            f.write("# Integration Test Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            
            # Summary
            summary = results['summary']
            f.write("## Summary\n\n")
            f.write(f"- Total Tests: {summary['total_tests']}\n")
            f.write(f"- Successful Tests: {summary['successful_tests']}\n")
            f.write(f"- Failed Tests: {summary['failed_tests']}\n")
            f.write(f"- Success Rate: {summary['success_rate']:.2%}\n")
            f.write(f"- Total Duration: {summary['total_duration']:.2f}s\n")
            f.write(f"- Average Duration: {summary['avg_duration']:.2f}s\n\n")
            
            # Test Results
            f.write("## Test Results\n\n")
            f.write("| Test Name | Success | Duration (s) | Error Message |\n")
            f.write("|-----------|---------|--------------|---------------|\n")
            
            for test_name, result in results['test_results'].items():
                error_msg = result.error_message if result.error_message else "None"
                f.write(f"| {test_name} | {'✓' if result.success else '✗'} | "
                       f"{result.duration:.2f} | {error_msg} |\n")
            f.write("\n")
            
            # Detailed Results
            f.write("## Detailed Results\n\n")
            for test_name, result in results['test_results'].items():
                f.write(f"### {test_name}\n\n")
                f.write(f"- **Success**: {'✓' if result.success else '✗'}\n")
                f.write(f"- **Duration**: {result.duration:.2f}s\n")
                f.write(f"- **Timestamp**: {result.timestamp}\n")
                
                if result.error_message:
                    f.write(f"- **Error**: {result.error_message}\n")
                
                if result.metrics:
                    f.write("- **Metrics**:\n")
                    for key, value in result.metrics.items():
                        f.write(f"  - {key}: {value}\n")
                f.write("\n")
        
        logging.info(f"Report generated: {report_file}")


def main():
    """Main function for running integration tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run integration tests for Applied Probability Framework')
    parser.add_argument('--results-dir', type=str, default='integration_test_results', 
                       help='Results directory')
    
    args = parser.parse_args()
    
    # Run integration tests
    test_suite = IntegrationTestSuite(results_dir=args.results_dir)
    results = test_suite.run_all_tests()
    
    # Print summary
    summary = results['summary']
    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETE")
    print("="*60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful Tests: {summary['successful_tests']}")
    print(f"Failed Tests: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.2%}")
    print(f"Total Duration: {summary['total_duration']:.2f}s")
    print(f"Results Directory: {args.results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
