"""
Mines Game Benchmark Runner

This script runs comprehensive benchmarks for all character strategies
and generates detailed results and analysis.

Features:
- Runs 10,000 simulations per strategy
- Generates CSV results files
- Creates performance charts
- Validates 2.12x minimum cash-out threshold
- Provides detailed analysis and recommendations
"""

import os
import sys
import time
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from core.parallel_engine import ParallelSimulationEngine, SimulationConfig
from core.game_simulator import MinesGameSimulator
from strategies.takeshi import TakeshiKovacsStrategy
from strategies.aoi import AoiStrategy
from strategies.yuzu import YuzuStrategy
from strategies.kazuya import KazuyaStrategy
from strategies.lelouch import LelouchStrategy
from game.math import payout_table_25_2, optimal_clicks, expected_value


class MinesBenchmark:
    """Comprehensive benchmark runner for Mines game strategies."""
    
    def __init__(self, num_simulations: int = 10000, board_size: int = 5, mine_count: int = 2):
        self.num_simulations = num_simulations
        self.board_size = board_size
        self.mine_count = mine_count
        self.results_dir = "results"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize strategies
        self.strategies = {
            'Takeshi Kovacs': TakeshiKovacsStrategy(),
            'Aoi': AoiStrategy(),
            'Yuzu': YuzuStrategy(),
            'Kazuya': KazuyaStrategy(),
            'Lelouch': LelouchStrategy()
        }
        
        # Results storage
        self.benchmark_results = {}
        self.detailed_results = {}
        
        # Payout table
        self.payout_table = payout_table_25_2()
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark for all strategies."""
        print("🎯 Starting Mines Game Benchmark")
        print("=" * 50)
        print(f"Simulations per strategy: {self.num_simulations:,}")
        print(f"Board size: {self.board_size}x{self.board_size}")
        print(f"Mine count: {self.mine_count}")
        print(f"Minimum cash-out threshold: 2.12x")
        print("=" * 50)
        
        start_time = time.time()
        
        # Run benchmarks for each strategy
        for name, strategy in self.strategies.items():
            print(f"\n🚀 Running benchmark for {name}...")
            strategy_results = self._run_strategy_benchmark(name, strategy)
            self.benchmark_results[name] = strategy_results
        
        # Generate analysis
        print("\n📊 Generating analysis...")
        analysis = self._generate_analysis()
        
        # Save results
        print("\n💾 Saving results...")
        self._save_results()
        
        # Generate charts
        print("\n📈 Generating charts...")
        self._generate_charts()
        
        total_time = time.time() - start_time
        print(f"\n✅ Benchmark completed in {total_time:.2f} seconds")
        
        return {
            'benchmark_results': self.benchmark_results,
            'analysis': analysis,
            'total_time': total_time,
            'timestamp': self.timestamp
        }
    
    def _run_strategy_benchmark(self, name: str, strategy) -> Dict[str, Any]:
        """Run benchmark for a single strategy."""
        print(f"  Running {self.num_simulations:,} simulations...")
        
        # Create simulator and engine
        simulator = MinesGameSimulator(board_size=self.board_size, mine_count=self.mine_count)
        config = SimulationConfig(
            simulation={'num_simulations': self.num_simulations, 'parallel': True},
            system={'n_jobs': -1}
        )
        engine = ParallelSimulationEngine(config)
        
        # Run simulations
        start_time = time.time()
        results = engine.run_simulation(
            simulator=simulator,
            strategy=strategy,
            num_simulations=self.num_simulations
        )
        duration = time.time() - start_time
        
        # Calculate detailed statistics
        rewards = [r.total_reward for r in results.results]
        wins = [r.win for r in results.results]
        clicks = [r.clicks_made for r in results.results]
        
        # Calculate payout analysis
        payout_analysis = self._analyze_payouts(rewards, clicks)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(rewards)
        
        # Calculate efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(rewards, clicks)
        
        # Store detailed results
        self.detailed_results[name] = {
            'rewards': rewards,
            'wins': wins,
            'clicks': clicks,
            'payout_analysis': payout_analysis,
            'risk_metrics': risk_metrics,
            'efficiency_metrics': efficiency_metrics
        }
        
        # Compile results
        strategy_results = {
            'name': name,
            'total_simulations': self.num_simulations,
            'duration': duration,
            'win_rate': results.win_rate,
            'avg_reward': results.avg_reward,
            'max_reward': results.max_reward,
            'min_reward': results.min_reward,
            'std_reward': np.std(rewards),
            'avg_clicks': np.mean(clicks),
            'max_clicks': np.max(clicks),
            'min_clicks': np.min(clicks),
            'payout_analysis': payout_analysis,
            'risk_metrics': risk_metrics,
            'efficiency_metrics': efficiency_metrics,
            'strategy_info': strategy.get_strategy_info()
        }
        
        print(f"  ✅ {name}: {results.win_rate:.2%} win rate, {results.avg_reward:.2f} avg reward")
        
        return strategy_results
    
    def _analyze_payouts(self, rewards: List[float], clicks: List[int]) -> Dict[str, Any]:
        """Analyze payout patterns and 2.12x threshold compliance."""
        # Count payouts above 2.12x threshold
        above_threshold = sum(1 for r in rewards if r >= 2.12)
        threshold_compliance = above_threshold / len(rewards)
        
        # Analyze payout distribution
        payout_ranges = {
            '0-1x': sum(1 for r in rewards if 0 <= r < 1),
            '1-2x': sum(1 for r in rewards if 1 <= r < 2),
            '2-3x': sum(1 for r in rewards if 2 <= r < 3),
            '3-5x': sum(1 for r in rewards if 3 <= r < 5),
            '5x+': sum(1 for r in rewards if r >= 5)
        }
        
        # Calculate payout efficiency
        total_potential = sum(self.payout_table.get(c, 1.0) for c in clicks)
        actual_payouts = sum(rewards)
        payout_efficiency = actual_payouts / total_potential if total_potential > 0 else 0
        
        return {
            'above_threshold_count': above_threshold,
            'threshold_compliance': threshold_compliance,
            'payout_ranges': payout_ranges,
            'payout_efficiency': payout_efficiency,
            'avg_payout': np.mean(rewards),
            'median_payout': np.median(rewards)
        }
    
    def _calculate_risk_metrics(self, rewards: List[float]) -> Dict[str, Any]:
        """Calculate risk metrics for the strategy."""
        if not rewards:
            return {}
        
        # Basic risk metrics
        volatility = np.std(rewards)
        max_drawdown = self._calculate_max_drawdown(rewards)
        
        # Value at Risk (VaR) - 5% and 1% levels
        var_5 = np.percentile(rewards, 5)
        var_1 = np.percentile(rewards, 1)
        
        # Expected Shortfall (Conditional VaR)
        es_5 = np.mean([r for r in rewards if r <= var_5])
        es_1 = np.mean([r for r in rewards if r <= var_1])
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = np.mean(rewards) / volatility if volatility > 0 else 0
        
        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'var_5_percent': var_5,
            'var_1_percent': var_1,
            'expected_shortfall_5': es_5,
            'expected_shortfall_1': es_1,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _calculate_max_drawdown(self, rewards: List[float]) -> float:
        """Calculate maximum drawdown from peak."""
        if not rewards:
            return 0.0
        
        cumulative = np.cumsum(rewards)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return np.max(drawdown)
    
    def _calculate_efficiency_metrics(self, rewards: List[float], clicks: List[int]) -> Dict[str, Any]:
        """Calculate efficiency metrics for the strategy."""
        if not rewards or not clicks:
            return {}
        
        # Reward per click
        reward_per_click = np.mean([r / c for r, c in zip(rewards, clicks) if c > 0])
        
        # Click efficiency (rewards per click)
        click_efficiency = np.mean([r / c for r, c in zip(rewards, clicks) if c > 0])
        
        # Optimal click analysis
        optimal_k, max_ev = optimal_clicks(self.board_size * self.board_size, self.mine_count, self.payout_table)
        avg_clicks = np.mean(clicks)
        click_optimality = 1 - abs(avg_clicks - optimal_k) / optimal_k
        
        return {
            'reward_per_click': reward_per_click,
            'click_efficiency': click_efficiency,
            'avg_clicks': avg_clicks,
            'optimal_clicks': optimal_k,
            'click_optimality': click_optimality
        }
    
    def _generate_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of benchmark results."""
        analysis = {
            'summary': self._generate_summary(),
            'rankings': self._generate_rankings(),
            'comparisons': self._generate_comparisons(),
            'recommendations': self._generate_recommendations(),
            'threshold_analysis': self._analyze_threshold_compliance()
        }
        
        return analysis
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'total_simulations': self.num_simulations * len(self.strategies),
            'strategies_tested': len(self.strategies),
            'board_configuration': f"{self.board_size}x{self.board_size} with {self.mine_count} mines",
            'minimum_threshold': 2.12,
            'best_strategy': None,
            'worst_strategy': None,
            'overall_win_rate': 0.0,
            'overall_avg_reward': 0.0
        }
        
        # Find best and worst strategies
        win_rates = {name: results['win_rate'] for name, results in self.benchmark_results.items()}
        avg_rewards = {name: results['avg_reward'] for name, results in self.benchmark_results.items()}
        
        summary['best_strategy'] = max(win_rates, key=win_rates.get)
        summary['worst_strategy'] = min(win_rates, key=win_rates.get)
        summary['overall_win_rate'] = np.mean(list(win_rates.values()))
        summary['overall_avg_reward'] = np.mean(list(avg_rewards.values()))
        
        return summary
    
    def _generate_rankings(self) -> Dict[str, List[str]]:
        """Generate strategy rankings by different metrics."""
        rankings = {
            'by_win_rate': [],
            'by_avg_reward': [],
            'by_max_reward': [],
            'by_risk_adjusted': [],
            'by_efficiency': []
        }
        
        # Sort by different metrics
        rankings['by_win_rate'] = sorted(self.benchmark_results.keys(), 
                                       key=lambda x: self.benchmark_results[x]['win_rate'], reverse=True)
        rankings['by_avg_reward'] = sorted(self.benchmark_results.keys(), 
                                         key=lambda x: self.benchmark_results[x]['avg_reward'], reverse=True)
        rankings['by_max_reward'] = sorted(self.benchmark_results.keys(), 
                                         key=lambda x: self.benchmark_results[x]['max_reward'], reverse=True)
        
        # Risk-adjusted ranking (Sharpe ratio)
        rankings['by_risk_adjusted'] = sorted(self.benchmark_results.keys(), 
                                            key=lambda x: self.benchmark_results[x]['risk_metrics'].get('sharpe_ratio', 0), reverse=True)
        
        # Efficiency ranking
        rankings['by_efficiency'] = sorted(self.benchmark_results.keys(), 
                                         key=lambda x: self.benchmark_results[x]['efficiency_metrics'].get('click_efficiency', 0), reverse=True)
        
        return rankings
    
    def _generate_comparisons(self) -> Dict[str, Any]:
        """Generate detailed comparisons between strategies."""
        comparisons = {
            'win_rate_comparison': {},
            'reward_comparison': {},
            'risk_comparison': {},
            'efficiency_comparison': {}
        }
        
        for name, results in self.benchmark_results.items():
            comparisons['win_rate_comparison'][name] = results['win_rate']
            comparisons['reward_comparison'][name] = results['avg_reward']
            comparisons['risk_comparison'][name] = results['risk_metrics'].get('volatility', 0)
            comparisons['efficiency_comparison'][name] = results['efficiency_metrics'].get('click_efficiency', 0)
        
        return comparisons
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        # Find best performing strategy
        best_strategy = max(self.benchmark_results.keys(), 
                          key=lambda x: self.benchmark_results[x]['win_rate'])
        best_win_rate = self.benchmark_results[best_strategy]['win_rate']
        
        recommendations.append(f"Best overall strategy: {best_strategy} with {best_win_rate:.2%} win rate")
        
        # Analyze threshold compliance
        threshold_compliance = {name: results['payout_analysis']['threshold_compliance'] 
                              for name, results in self.benchmark_results.items()}
        best_compliance = max(threshold_compliance, key=threshold_compliance.get)
        
        recommendations.append(f"Best 2.12x threshold compliance: {best_compliance} with {threshold_compliance[best_compliance]:.2%} compliance")
        
        # Risk analysis
        risk_metrics = {name: results['risk_metrics'].get('sharpe_ratio', 0) 
                       for name, results in self.benchmark_results.items()}
        best_risk_adjusted = max(risk_metrics, key=risk_metrics.get)
        
        recommendations.append(f"Best risk-adjusted returns: {best_risk_adjusted} with Sharpe ratio {risk_metrics[best_risk_adjusted]:.2f}")
        
        # Efficiency analysis
        efficiency_metrics = {name: results['efficiency_metrics'].get('click_efficiency', 0) 
                            for name, results in self.benchmark_results.items()}
        best_efficiency = max(efficiency_metrics, key=efficiency_metrics.get)
        
        recommendations.append(f"Most efficient strategy: {best_efficiency} with {efficiency_metrics[best_efficiency]:.2f} reward per click")
        
        return recommendations
    
    def _analyze_threshold_compliance(self) -> Dict[str, Any]:
        """Analyze compliance with 2.12x minimum threshold."""
        threshold_analysis = {
            'overall_compliance': 0.0,
            'strategy_compliance': {},
            'compliance_ranking': [],
            'threshold_effectiveness': {}
        }
        
        # Calculate overall compliance
        total_above_threshold = sum(results['payout_analysis']['above_threshold_count'] 
                                  for results in self.benchmark_results.values())
        total_simulations = sum(results['total_simulations'] 
                              for results in self.benchmark_results.values())
        threshold_analysis['overall_compliance'] = total_above_threshold / total_simulations
        
        # Strategy-specific compliance
        for name, results in self.benchmark_results.items():
            compliance = results['payout_analysis']['threshold_compliance']
            threshold_analysis['strategy_compliance'][name] = compliance
        
        # Compliance ranking
        threshold_analysis['compliance_ranking'] = sorted(
            threshold_analysis['strategy_compliance'].items(), 
            key=lambda x: x[1], reverse=True
        )
        
        return threshold_analysis
    
    def _save_results(self):
        """Save benchmark results to files."""
        # Save detailed results as JSON
        results_file = os.path.join(self.results_dir, f"mines_benchmark_{self.timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump({
                'benchmark_results': self.benchmark_results,
                'detailed_results': self.detailed_results,
                'timestamp': self.timestamp,
                'configuration': {
                    'num_simulations': self.num_simulations,
                    'board_size': self.board_size,
                    'mine_count': self.mine_count
                }
            }, f, indent=2, default=str)
        
        # Save summary as CSV
        summary_file = os.path.join(self.results_dir, f"mines_benchmark_summary_{self.timestamp}.csv")
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Strategy', 'Win Rate', 'Avg Reward', 'Max Reward', 'Std Reward', 'Avg Clicks', 'Threshold Compliance', 'Sharpe Ratio'])
            
            for name, results in self.benchmark_results.items():
                writer.writerow([
                    name,
                    f"{results['win_rate']:.4f}",
                    f"{results['avg_reward']:.4f}",
                    f"{results['max_reward']:.4f}",
                    f"{results['std_reward']:.4f}",
                    f"{results['avg_clicks']:.2f}",
                    f"{results['payout_analysis']['threshold_compliance']:.4f}",
                    f"{results['risk_metrics'].get('sharpe_ratio', 0):.4f}"
                ])
        
        # Save detailed results as CSV
        detailed_file = os.path.join(self.results_dir, f"mines_benchmark_detailed_{self.timestamp}.csv")
        with open(detailed_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Strategy', 'Simulation', 'Reward', 'Wins', 'Clicks'])
            
            for name, detailed in self.detailed_results.items():
                for i, (reward, win, clicks) in enumerate(zip(detailed['rewards'], detailed['wins'], detailed['clicks'])):
                    writer.writerow([name, i+1, reward, win, clicks])
        
        print(f"  📁 Results saved to {self.results_dir}/")
        print(f"  📊 Summary: mines_benchmark_summary_{self.timestamp}.csv")
        print(f"  📈 Detailed: mines_benchmark_detailed_{self.timestamp}.csv")
        print(f"  📋 Complete: mines_benchmark_{self.timestamp}.json")
    
    def _generate_charts(self):
        """Generate performance charts."""
        try:
            # Set up the plotting style
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Mines Game Benchmark Results - {self.timestamp}', fontsize=16)
            
            # Chart 1: Win Rate Comparison
            ax1 = axes[0, 0]
            strategies = list(self.benchmark_results.keys())
            win_rates = [self.benchmark_results[s]['win_rate'] for s in strategies]
            bars1 = ax1.bar(strategies, win_rates, color='skyblue', alpha=0.7)
            ax1.set_title('Win Rate Comparison')
            ax1.set_ylabel('Win Rate')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, rate in zip(bars1, win_rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{rate:.2%}', ha='center', va='bottom')
            
            # Chart 2: Average Reward Comparison
            ax2 = axes[0, 1]
            avg_rewards = [self.benchmark_results[s]['avg_reward'] for s in strategies]
            bars2 = ax2.bar(strategies, avg_rewards, color='lightgreen', alpha=0.7)
            ax2.set_title('Average Reward Comparison')
            ax2.set_ylabel('Average Reward')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, reward in zip(bars2, avg_rewards):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                        f'{reward:.2f}', ha='center', va='bottom')
            
            # Chart 3: Risk vs Reward Scatter
            ax3 = axes[1, 0]
            volatilities = [self.benchmark_results[s]['risk_metrics'].get('volatility', 0) for s in strategies]
            sharpe_ratios = [self.benchmark_results[s]['risk_metrics'].get('sharpe_ratio', 0) for s in strategies]
            scatter = ax3.scatter(volatilities, avg_rewards, s=100, alpha=0.7, c=sharpe_ratios, cmap='viridis')
            ax3.set_xlabel('Volatility (Risk)')
            ax3.set_ylabel('Average Reward')
            ax3.set_title('Risk vs Reward')
            plt.colorbar(scatter, ax=ax3, label='Sharpe Ratio')
            
            # Add strategy labels
            for i, strategy in enumerate(strategies):
                ax3.annotate(strategy, (volatilities[i], avg_rewards[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Chart 4: Threshold Compliance
            ax4 = axes[1, 1]
            threshold_compliance = [self.benchmark_results[s]['payout_analysis']['threshold_compliance'] for s in strategies]
            bars4 = ax4.bar(strategies, threshold_compliance, color='orange', alpha=0.7)
            ax4.set_title('2.12x Threshold Compliance')
            ax4.set_ylabel('Compliance Rate')
            ax4.set_ylim(0, 1)
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, compliance in zip(bars4, threshold_compliance):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{compliance:.2%}', ha='center', va='bottom')
            
            # Add horizontal line for 2.12x threshold
            ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Target')
            ax4.legend()
            
            plt.tight_layout()
            
            # Save chart
            chart_file = os.path.join(self.results_dir, f"mines_benchmark_charts_{self.timestamp}.png")
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📊 Charts saved: mines_benchmark_charts_{self.timestamp}.png")
            
        except Exception as e:
            print(f"  ⚠️ Chart generation failed: {e}")
    
    def print_summary(self):
        """Print benchmark summary to console."""
        print("\n" + "="*60)
        print("🎯 MINES GAME BENCHMARK SUMMARY")
        print("="*60)
        
        # Print strategy results
        print("\n📊 Strategy Performance:")
        print("-" * 60)
        print(f"{'Strategy':<15} {'Win Rate':<10} {'Avg Reward':<12} {'Max Reward':<12} {'Threshold':<10}")
        print("-" * 60)
        
        for name, results in self.benchmark_results.items():
            print(f"{name:<15} {results['win_rate']:<10.2%} {results['avg_reward']:<12.2f} {results['max_reward']:<12.2f} {results['payout_analysis']['threshold_compliance']:<10.2%}")
        
        # Print rankings
        print("\n🏆 Rankings:")
        print("-" * 30)
        
        rankings = self._generate_rankings()
        print("By Win Rate:", " > ".join(rankings['by_win_rate']))
        print("By Avg Reward:", " > ".join(rankings['by_avg_reward']))
        print("By Risk-Adjusted:", " > ".join(rankings['by_risk_adjusted']))
        
        # Print recommendations
        print("\n💡 Recommendations:")
        print("-" * 30)
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*60)


def main():
    """Main function to run the benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Mines Game Benchmark')
    parser.add_argument('--simulations', type=int, default=10000, help='Number of simulations per strategy')
    parser.add_argument('--board-size', type=int, default=5, help='Board size (default: 5)')
    parser.add_argument('--mines', type=int, default=2, help='Number of mines (default: 2)')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark (1000 simulations)')
    
    args = parser.parse_args()
    
    # Adjust simulations for quick mode
    if args.quick:
        args.simulations = 1000
        print("🚀 Running quick benchmark mode...")
    
    # Create and run benchmark
    benchmark = MinesBenchmark(
        num_simulations=args.simulations,
        board_size=args.board_size,
        mine_count=args.mines
    )
    
    # Run benchmark
    results = benchmark.run_benchmark()
    
    # Print summary
    benchmark.print_summary()
    
    return results


if __name__ == "__main__":
    main()
