"""
FAST AI TOURNAMENT - 1 MILLION GAMES

Quick but comprehensive tournament to find the champion!
Much faster than 10M games while still being statistically significant.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

import numpy as np
import time
from datetime import timedelta

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
        # Get valid actions
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
            return 0.0, clicks, False
        else:
            total_reward += 1.1 ** clicks
        
        if clicks == simulator.max_clicks:
            return total_reward, clicks, True
    
    return total_reward if clicks == simulator.max_clicks else 0.0, clicks, (clicks == simulator.max_clicks)


def run_tournament(total_games=1_000_000):
    """Run tournament."""
    
    print("\n" + "="*80)
    print("🏆 ULTIMATE AI TOURNAMENT - 1 MILLION GAMES 🏆")
    print("="*80)
    print(f"\n📊 Total Games: {total_games:,}")
    print("🎮 Board: 5x5 with 3 mines")
    print("⚡ Statistically significant results in minutes!\n")
    
    # Initialize strategies
    strategies = {
        'Takeshi Kovacs (Aggressive)': TakeshiKovacsStrategy(),
        'Lelouch vi Britannia (Strategic)': LelouchViBritanniaStrategy(),
        'Kazuya Kinoshita (Conservative)': KazuyaKinoshitaStrategy(),
        'Senku Ishigami (Analytical)': SenkuIshigamiStrategy(),
        'Rintaro Okabe (Mad Scientist)': RintaroOkabeStrategy(),
        'Hybrid Ultimate (AI Evolution)': HybridStrategy(),
    }
    
    # Statistics
    stats = {name: {'games': 0, 'wins': 0, 'total_reward': 0.0, 'perfect': 0}
             for name in strategies}
    
    games_per_strategy = total_games // len(strategies)
    print(f"📈 Games per strategy: {games_per_strategy:,}\n")
    print("="*80)
    
    start_time = time.time()
    
    # Run tournament
    for strategy_name, strategy in strategies.items():
        print(f"\n🎮 {strategy_name}")
        strategy_start = time.time()
        simulator = GameSimulator(board_size=5, mine_count=3, seed=42)
        
        for game_num in range(games_per_strategy):
            reward, clicks, won = run_single_game(strategy, simulator)
            
            stats[strategy_name]['games'] += 1
            stats[strategy_name]['total_reward'] += reward
            
            if won:
                stats[strategy_name]['wins'] += 1
                if clicks == 22:  # Perfect game
                    stats[strategy_name]['perfect'] += 1
            
            # Progress
            if (game_num + 1) % 25_000 == 0:
                progress = (game_num + 1) / games_per_strategy * 100
                wins = stats[strategy_name]['wins']
                win_rate = wins / (game_num + 1)
                avg_reward = stats[strategy_name]['total_reward'] / (game_num + 1)
                print(f"   {progress:5.1f}% | Win Rate: {win_rate:6.2%} | Avg Reward: {avg_reward:6.2f}")
        
        # Complete
        elapsed = time.time() - strategy_start
        win_rate = stats[strategy_name]['wins'] / games_per_strategy
        avg_reward = stats[strategy_name]['total_reward'] / games_per_strategy
        print(f"   ✓ Done in {elapsed:.1f}s | Win: {win_rate:.2%} | Reward: {avg_reward:.2f}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"🏁 COMPLETE! {total_games:,} games in {timedelta(seconds=int(total_time))}")
    print(f"⚡ Speed: {total_games/total_time:.0f} games/second")
    print(f"{'='*80}")
    
    return stats


def print_rankings(stats):
    """Print rankings with visual bars."""
    
    print("\n" + "="*80)
    print("🏆 FINAL RANKINGS 🏆")
    print("="*80)
    
    # Calculate scores
    rankings = []
    for name, stat in stats.items():
        win_rate = stat['wins'] / stat['games']
        avg_reward = stat['total_reward'] / stat['games']
        perfect_rate = stat['perfect'] / stat['games']
        
        # Composite score
        score = win_rate * 40 + (min(avg_reward / 10, 4)) * 40 + perfect_rate * 20
        
        rankings.append((name, win_rate, avg_reward, stat['total_reward'], 
                        stat['perfect'], perfect_rate, score))
    
    rankings.sort(key=lambda x: x[-1], reverse=True)
    
    medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣', '6️⃣']
    
    for rank, (name, win_rate, avg_reward, total_reward, perfects, perfect_rate, score) in enumerate(rankings):
        print(f"\n{medals[rank]} Rank #{rank+1}: {name.split('(')[0].strip()}")
        print(f"{'─'*80}")
        print(f"   Performance Score:  {score:6.2f} {'█' * int(score/2)}")
        print(f"   Win Rate:           {win_rate:6.2%} {'█' * int(win_rate*50)}")
        print(f"   Average Reward:     {avg_reward:6.2f} {'█' * int(avg_reward)}")
        print(f"   Total Reward:       {total_reward:,.0f}")
        print(f"   Perfect Games:      {perfects:,} ({perfect_rate:.2%})")
    
    # Champion announcement
    champion = rankings[0]
    print(f"\n{'='*80}")
    print(f"{'='*80}")
    print("🏆🏆🏆 THE ULTIMATE CHAMPION 🏆🏆🏆")
    print(f"{'='*80}")
    print(f"{'='*80}")
    print(f"\n  {champion[0]}")
    print(f"\n  Performance Score: {champion[-1]:.2f}")
    print(f"  Win Rate: {champion[1]:.2%}")
    print(f"  Average Reward: {champion[2]:.3f}")
    print(f"  Total Reward: {champion[3]:,.0f}")
    print(f"\n{'='*80}")


def visualize(stats):
    """Create awesome visualization."""
    try:
        import matplotlib.pyplot as plt
        
        names = [n.split('(')[0].strip() for n in stats.keys()]
        win_rates = [s['wins'] / s['games'] * 100 for s in stats.values()]
        avg_rewards = [s['total_reward'] / s['games'] for s in stats.values()]
        perfect_rates = [s['perfect'] / s['games'] * 100 for s in stats.values()]
        
        fig = plt.figure(figsize=(18, 10))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#C7CEEA']
        
        # Win Rate
        ax1 = plt.subplot(2, 2, 1)
        bars = ax1.bar(names, win_rates, color=colors, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('🎯 Win Rate', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, win_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Average Reward
        ax2 = plt.subplot(2, 2, 2)
        bars = ax2.bar(names, avg_rewards, color=colors, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
        ax2.set_title('💰 Average Reward', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, avg_rewards):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Perfect Game Rate
        ax3 = plt.subplot(2, 2, 3)
        bars = ax3.bar(names, perfect_rates, color=colors, edgecolor='black', linewidth=2)
        ax3.set_ylabel('Perfect Game Rate (%)', fontsize=12, fontweight='bold')
        ax3.set_title('⭐ Perfect Games', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, perfect_rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Composite Score
        ax4 = plt.subplot(2, 2, 4)
        scores = [
            s['wins']/s['games']*40 + min((s['total_reward']/s['games'])/10*40, 40) + s['perfect']/s['games']*20
            for s in stats.values()
        ]
        bars = ax4.bar(names, scores, color=colors, edgecolor='black', linewidth=2)
        ax4.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
        ax4.set_title('🏆 Overall Performance', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('🏆 ULTIMATE AI TOURNAMENT - 1 MILLION GAMES 🏆', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        save_path = results_dir / 'ultimate_tournament_1M.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {save_path}")
        plt.show()
        
    except ImportError:
        print("\n📊 (Matplotlib not available - visualization skipped)")


def main():
    """Run tournament."""
    stats = run_tournament(total_games=1_000_000)
    print_rankings(stats)
    visualize(stats)
    
    print("\n💡 Want 10 million games? Run: python examples/ultimate_tournament.py")
    print("   (Takes ~10x longer but gives even more precise results)\n")


if __name__ == '__main__':
    main()

