"""
MEGA AI TOURNAMENT - 10 MILLION GAMES

Run all strategies in a massive tournament to determine the ultimate champion.
Includes real-time progress tracking and comprehensive visualization.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict

from character_strategies import (
    TakeshiKovacsStrategy,
    LelouchViBritanniaStrategy,
    KazuyaKinoshitaStrategy,
    SenkuIshigamiStrategy,
    RintaroOkabeStrategy,
    HybridStrategy,
)
from simulators.mines_simulator import MinesSimulator


class TournamentStats:
    """Track comprehensive tournament statistics."""
    
    def __init__(self):
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.total_reward = 0.0
        self.rewards = []
        self.game_lengths = []
        self.clicks_before_loss = []
        self.max_reward = 0.0
        self.perfect_games = 0
        self.start_time = time.time()
        
    def update(self, reward: float, game_length: int, clicks: int):
        """Update statistics with game result."""
        self.games_played += 1
        self.total_reward += reward
        self.rewards.append(reward)
        self.game_lengths.append(game_length)
        
        if reward > 0:
            self.wins += 1
            if reward > self.max_reward:
                self.max_reward = reward
            # Perfect game = cleared all safe cells
            if clicks >= 22:  # 5x5 board with 3 mines = 22 safe cells
                self.perfect_games += 1
        else:
            self.losses += 1
            self.clicks_before_loss.append(clicks)
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        return {
            'games_played': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.wins / max(self.games_played, 1),
            'avg_reward': np.mean(self.rewards) if self.rewards else 0.0,
            'std_reward': np.std(self.rewards) if self.rewards else 0.0,
            'median_reward': np.median(self.rewards) if self.rewards else 0.0,
            'max_reward': self.max_reward,
            'total_reward': self.total_reward,
            'avg_game_length': np.mean(self.game_lengths) if self.game_lengths else 0.0,
            'avg_clicks_before_loss': np.mean(self.clicks_before_loss) if self.clicks_before_loss else 0.0,
            'perfect_games': self.perfect_games,
            'perfect_rate': self.perfect_games / max(self.games_played, 1),
            'elapsed_time': time.time() - self.start_time,
        }


def run_tournament(
    total_games: int = 10_000_000,
    board_size: int = 5,
    mine_count: int = 3,
    seed: int = 42
) -> Dict[str, TournamentStats]:
    """
    Run massive tournament with all strategies.
    
    Args:
        total_games: Total number of games to play
        board_size: Size of the mines board
        mine_count: Number of mines
        seed: Random seed
        
    Returns:
        Dictionary mapping strategy names to their statistics
    """
    print("\n" + "="*80)
    print("🏆 MEGA AI TOURNAMENT - 10 MILLION GAMES 🏆")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Total Games: {total_games:,}")
    print(f"  Board Size: {board_size}x{board_size}")
    print(f"  Mine Count: {mine_count}")
    print(f"  Random Seed: {seed}")
    
    # Initialize all strategies
    strategies = {
        'Takeshi Kovacs (Aggressive)': TakeshiKovacsStrategy(),
        'Lelouch (Strategic)': LelouchViBritanniaStrategy(),
        'Kazuya (Conservative)': KazuyaKinoshitaStrategy(),
        'Senku (Analytical)': SenkuIshigamiStrategy(),
        'Rintaro Okabe (Mad Scientist + AI)': RintaroOkabeStrategy(config={
            'q_learning_weight': 0.4,
            'learning_rate': 0.1,
            'epsilon': 0.1,
        }),
        'Hybrid (Ultimate + AI)': HybridStrategy(config={
            'q_learning_weight': 0.5,
            'learning_rate': 0.1,
            'epsilon': 0.1,
        }),
    }
    
    # Initialize statistics
    stats = {name: TournamentStats() for name in strategies.keys()}
    
    # Calculate games per strategy
    games_per_strategy = total_games // len(strategies)
    
    print(f"\n  Games per strategy: {games_per_strategy:,}")
    print(f"\n{'='*80}")
    print("Starting tournament...\n")
    
    start_time = time.time()
    last_update = start_time
    
    # Run tournament for each strategy
    for strategy_name, strategy in strategies.items():
        print(f"\n{'='*80}")
        print(f"Testing: {strategy_name}")
        print(f"{'='*80}")
        
        strategy_start = time.time()
        
        # Create simulator
        simulator = MinesSimulator(
            seed=seed,
            config={
                'board_size': board_size,
                'mine_count': mine_count,
                'multiplier_base': 1.1
            }
        )
        
        for game_num in range(games_per_strategy):
            # Reset game
            state = simulator.reset()
            done = False
            game_length = 0
            clicks_made = 0
            episode_reward = 0.0
            
            # Play game
            while not done:
                valid_actions = simulator.get_valid_actions()
                
                if not valid_actions:
                    break
                
                # Select action
                try:
                    action = strategy.select_action(state, valid_actions)
                except Exception as e:
                    # If strategy fails, pick random action
                    action = np.random.choice(valid_actions) if valid_actions else None
                
                if action is None:
                    break
                
                # Take step
                next_state, reward, done, info = simulator.step(action)
                
                # Update strategy (for learning strategies)
                if hasattr(strategy, 'update'):
                    strategy.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                game_length += 1
                clicks_made = state.get('clicks_made', 0) if isinstance(state, dict) else 0
                
                state = next_state
            
            # Update statistics
            stats[strategy_name].update(episode_reward, game_length, clicks_made)
            
            # Progress update every 100,000 games
            if (game_num + 1) % 100_000 == 0:
                elapsed = time.time() - strategy_start
                games_done = game_num + 1
                games_per_sec = games_done / elapsed
                eta_seconds = (games_per_strategy - games_done) / games_per_sec
                eta = timedelta(seconds=int(eta_seconds))
                
                current_stats = stats[strategy_name].get_stats()
                print(f"  Progress: {games_done:,}/{games_per_strategy:,} "
                      f"({100*games_done/games_per_strategy:.1f}%) | "
                      f"Win Rate: {current_stats['win_rate']:.2%} | "
                      f"Avg Reward: {current_stats['avg_reward']:.3f} | "
                      f"Speed: {games_per_sec:.0f} games/sec | "
                      f"ETA: {eta}")
        
        # Strategy complete
        strategy_time = time.time() - strategy_start
        final_stats = stats[strategy_name].get_stats()
        
        print(f"\n  ✓ Completed {games_per_strategy:,} games in {strategy_time:.1f}s")
        print(f"    Win Rate: {final_stats['win_rate']:.2%}")
        print(f"    Avg Reward: {final_stats['avg_reward']:.4f}")
        print(f"    Perfect Games: {final_stats['perfect_games']:,} ({final_stats['perfect_rate']:.2%})")
    
    # Tournament complete
    total_time = time.time() - start_time
    total_games_played = sum(s.games_played for s in stats.values())
    
    print(f"\n{'='*80}")
    print("🏁 TOURNAMENT COMPLETE! 🏁")
    print(f"{'='*80}")
    print(f"Total Games Played: {total_games_played:,}")
    print(f"Total Time: {timedelta(seconds=int(total_time))}")
    print(f"Average Speed: {total_games_played/total_time:.0f} games/sec")
    
    return stats


def visualize_results(stats: Dict[str, TournamentStats], save_path: str = None):
    """
    Create comprehensive visualizations of tournament results.
    
    Args:
        stats: Dictionary of strategy statistics
        save_path: Path to save plots (optional)
    """
    # Get statistics
    strategy_names = list(stats.keys())
    strategy_stats = [s.get_stats() for s in stats.values()]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#C7CEEA']
    
    # 1. Win Rate Comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    win_rates = [s['win_rate'] * 100 for s in strategy_stats]
    bars1 = ax1.bar(range(len(strategy_names)), win_rates, color=colors)
    ax1.set_ylabel('Win Rate (%)', fontweight='bold')
    ax1.set_title('Win Rate Comparison', fontweight='bold', fontsize=14)
    ax1.set_xticks(range(len(strategy_names)))
    ax1.set_xticklabels([name.split('(')[0].strip() for name in strategy_names], 
                         rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    ax1.legend()
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, win_rates)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Average Reward Comparison (top center)
    ax2 = fig.add_subplot(gs[0, 1])
    avg_rewards = [s['avg_reward'] for s in strategy_stats]
    bars2 = ax2.bar(range(len(strategy_names)), avg_rewards, color=colors)
    ax2.set_ylabel('Average Reward', fontweight='bold')
    ax2.set_title('Average Reward per Game', fontweight='bold', fontsize=14)
    ax2.set_xticks(range(len(strategy_names)))
    ax2.set_xticklabels([name.split('(')[0].strip() for name in strategy_names], 
                         rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars2, avg_rewards)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Total Reward (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    total_rewards = [s['total_reward'] for s in strategy_stats]
    bars3 = ax3.bar(range(len(strategy_names)), total_rewards, color=colors)
    ax3.set_ylabel('Total Reward', fontweight='bold')
    ax3.set_title('Total Cumulative Reward', fontweight='bold', fontsize=14)
    ax3.set_xticks(range(len(strategy_names)))
    ax3.set_xticklabels([name.split('(')[0].strip() for name in strategy_names], 
                         rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars3, total_rewards)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val/1000:.0f}K', ha='center', va='bottom', fontweight='bold')
    
    # 4. Perfect Game Rate (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    perfect_rates = [s['perfect_rate'] * 100 for s in strategy_stats]
    bars4 = ax4.bar(range(len(strategy_names)), perfect_rates, color=colors)
    ax4.set_ylabel('Perfect Game Rate (%)', fontweight='bold')
    ax4.set_title('Perfect Games (All Safe Cells Cleared)', fontweight='bold', fontsize=14)
    ax4.set_xticks(range(len(strategy_names)))
    ax4.set_xticklabels([name.split('(')[0].strip() for name in strategy_names], 
                         rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars4, perfect_rates)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 5. Reward Distribution (middle center)
    ax5 = fig.add_subplot(gs[1, 1])
    for i, (name, stat) in enumerate(zip(strategy_names, stats.values())):
        rewards_sample = stat.rewards[::max(1, len(stat.rewards)//1000)]  # Sample for performance
        ax5.hist(rewards_sample, bins=50, alpha=0.5, label=name.split('(')[0].strip(), 
                color=colors[i])
    ax5.set_xlabel('Reward', fontweight='bold')
    ax5.set_ylabel('Frequency', fontweight='bold')
    ax5.set_title('Reward Distribution', fontweight='bold', fontsize=14)
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Risk-Reward Scatter (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    std_rewards = [s['std_reward'] for s in strategy_stats]
    ax6.scatter(std_rewards, avg_rewards, s=500, c=colors, alpha=0.6)
    for i, name in enumerate(strategy_names):
        ax6.annotate(name.split('(')[0].strip(), 
                    (std_rewards[i], avg_rewards[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontweight='bold')
    ax6.set_xlabel('Risk (Std Dev of Rewards)', fontweight='bold')
    ax6.set_ylabel('Return (Avg Reward)', fontweight='bold')
    ax6.set_title('Risk-Return Profile', fontweight='bold', fontsize=14)
    ax6.grid(alpha=0.3)
    
    # 7. Overall Performance Score (bottom left)
    ax7 = fig.add_subplot(gs[2, 0])
    # Composite score: 40% win rate + 40% avg reward + 20% perfect rate
    performance_scores = [
        0.4 * s['win_rate'] * 100 + 
        0.4 * s['avg_reward'] * 20 + 
        0.2 * s['perfect_rate'] * 100
        for s in strategy_stats
    ]
    bars7 = ax7.bar(range(len(strategy_names)), performance_scores, color=colors)
    ax7.set_ylabel('Performance Score', fontweight='bold')
    ax7.set_title('Overall Performance Score', fontweight='bold', fontsize=14)
    ax7.set_xticks(range(len(strategy_names)))
    ax7.set_xticklabels([name.split('(')[0].strip() for name in strategy_names], 
                         rotation=45, ha='right')
    ax7.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars7, performance_scores)):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 8. Efficiency (bottom center)
    ax8 = fig.add_subplot(gs[2, 1])
    avg_clicks = [s['avg_clicks_before_loss'] for s in strategy_stats]
    bars8 = ax8.bar(range(len(strategy_names)), avg_clicks, color=colors)
    ax8.set_ylabel('Average Clicks Before Loss', fontweight='bold')
    ax8.set_title('Risk Management (Higher = Better)', fontweight='bold', fontsize=14)
    ax8.set_xticks(range(len(strategy_names)))
    ax8.set_xticklabels([name.split('(')[0].strip() for name in strategy_names], 
                         rotation=45, ha='right')
    ax8.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars8, avg_clicks)):
        height = bar.get_height()
        if val > 0:
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 9. Champion Podium (bottom right)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Determine rankings
    rankings = sorted(enumerate(performance_scores), key=lambda x: x[1], reverse=True)
    
    # Display podium
    podium_text = "🏆 TOURNAMENT RESULTS 🏆\n\n"
    medals = ['🥇', '🥈', '🥉']
    
    for rank, (idx, score) in enumerate(rankings[:3]):
        name = strategy_names[idx].split('(')[0].strip()
        stat = strategy_stats[idx]
        podium_text += f"{medals[rank]} {name}\n"
        podium_text += f"   Score: {score:.1f}\n"
        podium_text += f"   Win Rate: {stat['win_rate']:.2%}\n"
        podium_text += f"   Avg Reward: {stat['avg_reward']:.4f}\n\n"
    
    ax9.text(0.5, 0.5, podium_text, 
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Main title
    fig.suptitle('MEGA AI TOURNAMENT - 10 MILLION GAMES ANALYSIS', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {save_path}")
    
    plt.show()


def save_results(stats: Dict[str, TournamentStats], filepath: str):
    """Save tournament results to JSON."""
    results = {}
    for name, stat in stats.items():
        results[name] = stat.get_stats()
        # Convert numpy types to Python types
        for key, value in results[name].items():
            if isinstance(value, (np.integer, np.floating)):
                results[name][key] = float(value)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📝 Results saved to: {filepath}")


def print_final_rankings(stats: Dict[str, TournamentStats]):
    """Print final tournament rankings."""
    strategy_names = list(stats.keys())
    strategy_stats = [s.get_stats() for s in stats.values()]
    
    # Calculate performance scores
    performance_scores = [
        0.4 * s['win_rate'] * 100 + 
        0.4 * s['avg_reward'] * 20 + 
        0.2 * s['perfect_rate'] * 100
        for s in strategy_stats
    ]
    
    # Sort by performance score
    rankings = sorted(zip(strategy_names, strategy_stats, performance_scores),
                     key=lambda x: x[2], reverse=True)
    
    print("\n" + "="*80)
    print("📊 FINAL RANKINGS 📊")
    print("="*80)
    
    for rank, (name, stat, score) in enumerate(rankings, 1):
        print(f"\n{'='*80}")
        print(f"Rank #{rank}: {name}")
        print(f"{'='*80}")
        print(f"Performance Score:      {score:.2f}")
        print(f"Games Played:           {stat['games_played']:,}")
        print(f"Win Rate:               {stat['win_rate']:.2%}")
        print(f"Average Reward:         {stat['avg_reward']:.4f}")
        print(f"Median Reward:          {stat['median_reward']:.4f}")
        print(f"Std Dev Reward:         {stat['std_reward']:.4f}")
        print(f"Total Reward:           {stat['total_reward']:,.2f}")
        print(f"Max Reward:             {stat['max_reward']:.2f}")
        print(f"Perfect Games:          {stat['perfect_games']:,} ({stat['perfect_rate']:.2%})")
        print(f"Avg Clicks Before Loss: {stat['avg_clicks_before_loss']:.2f}")
        print(f"Avg Game Length:        {stat['avg_game_length']:.2f}")


def main():
    """Run mega tournament."""
    # Run tournament
    stats = run_tournament(total_games=10_000_000)
    
    # Print final rankings
    print_final_rankings(stats)
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'mega_tournament_{timestamp}.json'
    save_results(stats, str(results_file))
    
    # Visualize results
    plot_file = results_dir / f'mega_tournament_{timestamp}.png'
    visualize_results(stats, str(plot_file))
    
    # Declare champion
    strategy_names = list(stats.keys())
    strategy_stats = [s.get_stats() for s in stats.values()]
    performance_scores = [
        0.4 * s['win_rate'] * 100 + 
        0.4 * s['avg_reward'] * 20 + 
        0.2 * s['perfect_rate'] * 100
        for s in strategy_stats
    ]
    
    champion_idx = np.argmax(performance_scores)
    champion_name = strategy_names[champion_idx]
    
    print("\n" + "="*80)
    print("🏆🏆🏆 THE CHAMPION 🏆🏆🏆")
    print("="*80)
    print(f"\n{champion_name}")
    print(f"\nAfter 10 MILLION games, {champion_name.split('(')[0].strip()} emerges victorious!")
    print(f"Performance Score: {performance_scores[champion_idx]:.2f}")
    print(f"Win Rate: {strategy_stats[champion_idx]['win_rate']:.2%}")
    print(f"Average Reward: {strategy_stats[champion_idx]['avg_reward']:.4f}")
    print("\n" + "="*80)


if __name__ == '__main__':
    try:
        import matplotlib
        main()
    except ImportError:
        print("ERROR: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        sys.exit(1)

