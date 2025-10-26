"""
ULTIMATE AI TOURNAMENT - 10 MILLION GAMES

Run all strategies in a massive tournament to find the champion!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

import numpy as np
import time
from datetime import timedelta
from collections import defaultdict

from character_strategies import (
    TakeshiKovacsStrategy,
    LelouchViBritanniaStrategy,
    KazuyaKinoshitaStrategy,
    SenkuIshigamiStrategy,
    RintaroOkabeStrategy,
    HybridStrategy,
)
from game_simulator import GameSimulator


def run_single_game(strategy, simulator):
    """Run a single game and return results."""
    simulator.initialize_game()
    
    total_reward = 0.0
    clicks = 0
    
    while clicks < simulator.max_clicks:
        # Get valid actions (unrevealed cells)
        valid_actions = [
            (r, c) for r in range(simulator.board_size) 
            for c in range(simulator.board_size)
            if not simulator.revealed[r][c]
        ]
        
        if not valid_actions:
            break
        
        # Get state
        state = {
            'board_size': simulator.board_size,
            'mine_count': simulator.mine_count,
            'clicks_made': clicks,
            'revealed': simulator.revealed.copy(),
        }
        
        # Select action
        try:
            action = strategy.select_action(state, valid_actions)
        except:
            action = valid_actions[np.random.randint(len(valid_actions))]
        
        if action is None:
            break
        
        # Click cell
        r, c = action
        success, result = simulator.click_cell(r, c)
        
        if not success:
            break
        
        clicks += 1
        
        if result == "Mine":
            # Hit a mine - game over
            return 0.0, clicks, False
        else:
            # Safe click - increase reward
            total_reward += 1.1 ** clicks
        
        # Check if won (cleared all safe cells)
        if clicks == simulator.max_clicks:
            return total_reward, clicks, True
    
    return total_reward if clicks == simulator.max_clicks else 0.0, clicks, (clicks == simulator.max_clicks)


def run_tournament(total_games=10_000_000, board_size=5, mine_count=3, seed=42):
    """Run the mega tournament."""
    
    print("\n" + "="*80)
    print("🏆 ULTIMATE AI TOURNAMENT - 10 MILLION GAMES 🏆")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Total Games: {total_games:,}")
    print(f"  Board: {board_size}x{board_size}")
    print(f"  Mines: {mine_count}")
    print(f"  Seed: {seed}")
    
    # Initialize strategies
    strategies = {
        'Takeshi Kovacs': TakeshiKovacsStrategy(),
        'Lelouch vi Britannia': LelouchViBritanniaStrategy(),
        'Kazuya Kinoshita': KazuyaKinoshitaStrategy(),
        'Senku Ishigami': SenkuIshigamiStrategy(),
        'Rintaro Okabe': RintaroOkabeStrategy(),
        'Hybrid Ultimate': HybridStrategy(),
    }
    
    # Statistics tracking
    stats = {}
    for name in strategies:
        stats[name] = {
            'games': 0,
            'wins': 0,
            'losses': 0,
            'total_reward': 0.0,
            'total_clicks': 0,
            'perfect_games': 0,
            'rewards': [],
        }
    
    games_per_strategy = total_games // len(strategies)
    
    print(f"  Games per strategy: {games_per_strategy:,}\n")
    print("="*80)
    
    start_time = time.time()
    
    # Run tournament
    for strategy_name, strategy in strategies.items():
        print(f"\n{'='*80}")
        print(f"🎮 Testing: {strategy_name}")
        print(f"{'='*80}")
        
        strategy_start = time.time()
        simulator = GameSimulator(board_size=board_size, mine_count=mine_count, seed=seed)
        
        for game_num in range(games_per_strategy):
            reward, clicks, won = run_single_game(strategy, simulator)
            
            stats[strategy_name]['games'] += 1
            stats[strategy_name]['total_reward'] += reward
            stats[strategy_name]['total_clicks'] += clicks
            stats[strategy_name]['rewards'].append(reward)
            
            if won:
                stats[strategy_name]['wins'] += 1
                if clicks == simulator.max_clicks:
                    stats[strategy_name]['perfect_games'] += 1
            else:
                stats[strategy_name]['losses'] += 1
            
            # Progress update
            if (game_num + 1) % 50_000 == 0:
                elapsed = time.time() - strategy_start
                games_done = game_num + 1
                rate = games_done / elapsed
                eta = (games_per_strategy - games_done) / rate
                
                win_rate = stats[strategy_name]['wins'] / games_done
                avg_reward = stats[strategy_name]['total_reward'] / games_done
                
                print(f"  Progress: {games_done:,}/{games_per_strategy:,} "
                      f"({100*games_done/games_per_strategy:.1f}%) | "
                      f"Win: {win_rate:.2%} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Speed: {rate:.0f} g/s | "
                      f"ETA: {timedelta(seconds=int(eta))}")
        
        # Strategy complete
        elapsed = time.time() - strategy_start
        win_rate = stats[strategy_name]['wins'] / games_per_strategy
        avg_reward = stats[strategy_name]['total_reward'] / games_per_strategy
        
        print(f"\n  ✅ Completed in {timedelta(seconds=int(elapsed))}")
        print(f"     Win Rate: {win_rate:.2%}")
        print(f"     Avg Reward: {avg_reward:.3f}")
        print(f"     Perfect Games: {stats[strategy_name]['perfect_games']:,}")
    
    # Tournament complete
    total_time = time.time() - start_time
    total_played = sum(s['games'] for s in stats.values())
    
    print(f"\n{'='*80}")
    print("🏁 TOURNAMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"Total Games: {total_played:,}")
    print(f"Total Time: {timedelta(seconds=int(total_time))}")
    print(f"Speed: {total_played/total_time:.0f} games/sec")
    
    return stats


def print_rankings(stats):
    """Print final rankings."""
    
    print("\n" + "="*80)
    print("📊 FINAL RANKINGS 📊")
    print("="*80)
    
    # Calculate scores
    rankings = []
    for name, stat in stats.items():
        win_rate = stat['wins'] / max(stat['games'], 1)
        avg_reward = stat['total_reward'] / max(stat['games'], 1)
        perfect_rate = stat['perfect_games'] / max(stat['games'], 1)
        
        # Composite score
        score = win_rate * 40 + (avg_reward / 10) * 40 + perfect_rate * 20
        
        rankings.append((name, win_rate, avg_reward, stat['total_reward'], 
                        stat['perfect_games'], perfect_rate, score))
    
    rankings.sort(key=lambda x: x[-1], reverse=True)
    
    for rank, (name, win_rate, avg_reward, total_reward, perfects, perfect_rate, score) in enumerate(rankings, 1):
        print(f"\n{'='*80}")
        print(f"Rank #{rank}: {name}")
        print(f"{'='*80}")
        print(f"  Performance Score:  {score:.2f}")
        print(f"  Win Rate:           {win_rate:.2%}")
        print(f"  Average Reward:     {avg_reward:.3f}")
        print(f"  Total Reward:       {total_reward:,.0f}")
        print(f"  Perfect Games:      {perfects:,} ({perfect_rate:.2%})")
    
    # Champion
    champion = rankings[0]
    print(f"\n{'='*80}")
    print("🏆🏆🏆 THE CHAMPION 🏆🏆🏆")
    print(f"{'='*80}")
    print(f"\n{champion[0]}")
    print(f"\nAfter {stats[champion[0]]['games']:,} games per strategy,")
    print(f"{champion[0]} emerges victorious!")
    print(f"\nPerformance Score: {champion[-1]:.2f}")
    print(f"Win Rate: {champion[1]:.2%}")
    print(f"Average Reward: {champion[2]:.3f}")
    print("\n" + "="*80)


def visualize_results(stats):
    """Create visualization if matplotlib available."""
    try:
        import matplotlib.pyplot as plt
        
        names = list(stats.keys())
        win_rates = [s['wins'] / s['games'] * 100 for s in stats.values()]
        avg_rewards = [s['total_reward'] / s['games'] for s in stats.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Win rates
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#C7CEEA']
        bars1 = ax1.bar(range(len(names)), win_rates, color=colors)
        ax1.set_ylabel('Win Rate (%)', fontweight='bold')
        ax1.set_title('Win Rate Comparison', fontweight='bold', fontsize=14)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels([n.split()[0] for n in names], rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars1, win_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Avg rewards
        bars2 = ax2.bar(range(len(names)), avg_rewards, color=colors)
        ax2.set_ylabel('Average Reward', fontweight='bold')
        ax2.set_title('Average Reward Comparison', fontweight='bold', fontsize=14)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels([n.split()[0] for n in names], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars2, avg_rewards):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('ULTIMATE AI TOURNAMENT - 10 MILLION GAMES', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        save_path = results_dir / 'ultimate_tournament.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {save_path}")
        plt.show()
        
    except ImportError:
        print("\n📊 Matplotlib not available - skipping visualization")


def main():
    """Run the ultimate tournament."""
    stats = run_tournament(total_games=10_000_000)
    print_rankings(stats)
    visualize_results(stats)


if __name__ == '__main__':
    main()

