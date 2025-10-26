"""
INSTANT TOURNAMENT - 50K Games

Super fast demo that runs in seconds and shows you the champion!
Scale up to millions after seeing it works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

import numpy as np
import time

from character_strategies import (
    TakeshiKovacsStrategy,
    LelouchViBritanniaStrategy,
    KazuyaKinoshitaStrategy,
    SenkuIshigamiStrategy,
    RintaroOkabeStrategy,
    HybridStrategy,
)
from game_simulator import GameSimulator


def run_game(strategy, simulator):
    """Run one game."""
    simulator.initialize_game()
    reward, clicks = 0.0, 0
    
    while clicks < 22:  # 22 safe cells max
        valid = [(r,c) for r in range(5) for c in range(5) if not simulator.revealed[r][c]]
        if not valid:
            break
        
        state = {'board_size': 5, 'mine_count': 3, 'clicks_made': clicks}
        
        try:
            action = strategy.select_action(state, valid)
        except:
            action = valid[0]
        
        if not action:
            break
        
        success, result = simulator.click_cell(*action)
        clicks += 1
        
        if result == "Mine":
            return 0.0, clicks, False
        reward += 1.1 ** clicks
        
        if clicks == 22:
            return reward, clicks, True
    
    return (reward if clicks == 22 else 0.0), clicks, (clicks == 22)


print("\n" + "="*80)
print("🏆⚡ INSTANT AI TOURNAMENT - 50,000 GAMES ⚡🏆")
print("="*80)
print("\n⚡ Running at lightning speed...\n")

strategies = {
    'Takeshi': TakeshiKovacsStrategy(),
    'Lelouch': LelouchViBritanniaStrategy(),
    'Kazuya': KazuyaKinoshitaStrategy(),
    'Senku': SenkuIshigamiStrategy(),
    'Okabe': RintaroOkabeStrategy(),
    'Hybrid': HybridStrategy(),
}

stats = {name: {'wins': 0, 'reward': 0.0, 'perfect': 0, 'games': 0}
         for name in strategies}

games_each = 50_000 // len(strategies)

start = time.time()

for name, strategy in strategies.items():
    print(f"🎮 {name:15} ", end='', flush=True)
    sim = GameSimulator(5, 3, 42)
    
    for _ in range(games_each):
        r, c, won = run_game(strategy, sim)
        stats[name]['games'] += 1
        stats[name]['reward'] += r
        if won:
            stats[name]['wins'] += 1
            if c == 22:
                stats[name]['perfect'] += 1
    
    wr = stats[name]['wins'] / games_each
    ar = stats[name]['reward'] / games_each
    print(f"✓ | Win: {wr:6.2%} | Reward: {ar:6.2f}")

elapsed = time.time() - start

print(f"\n⚡ Completed 50,000 games in {elapsed:.1f} seconds!")
print(f"⚡ Speed: {50_000/elapsed:.0f} games/second\n")

# Rankings
print("="*80)
print("🏆 RANKINGS 🏆")
print("="*80)

rankings = []
for name, s in stats.items():
    wr = s['wins'] / s['games']
    ar = s['reward'] / s['games']
    pr = s['perfect'] / s['games']
    score = wr * 40 + min(ar/10*40, 40) + pr * 20
    rankings.append((name, wr, ar, s['reward'], s['perfect'], pr, score))

rankings.sort(key=lambda x: x[-1], reverse=True)

medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣', '6️⃣']

for rank, (name, wr, ar, tr, pf, pr, score) in enumerate(rankings):
    bar_len = int(score / 2)
    print(f"\n{medals[rank]} {name:15} Score: {score:5.1f} {'█' * bar_len}")
    print(f"   Win Rate:    {wr:6.2%}  {'█' * int(wr*30)}")
    print(f"   Avg Reward:  {ar:6.2f}  {'█' * int(ar*2)}")
    print(f"   Perfect:     {pf:5,} ({pr:5.2%})")

# Champion
champ = rankings[0]
print(f"\n{'='*80}")
print("🏆🏆🏆 CHAMPION 🏆🏆🏆")
print(f"{'='*80}")
print(f"\n  {champ[0]}")
print(f"\n  Performance Score: {champ[-1]:.2f}")
print(f"  Win Rate: {champ[1]:.2%}")
print(f"  Average Reward: {champ[2]:.2f}")
print(f"\n{'='*80}")

# Visualization
try:
    import matplotlib.pyplot as plt
    
    names = [r[0] for r in rankings]
    scores = [r[-1] for r in rankings]
    win_rates = [r[1]*100 for r in rankings]
    avg_rewards = [r[2] for r in rankings]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#C7CEEA']
    
    # Performance Score
    ax = axes[0]
    bars = ax.barh(names, scores, color=colors, edgecolor='black', linewidth=2)
    ax.set_xlabel('Performance Score', fontweight='bold')
    ax.set_title('🏆 Overall Performance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, scores)):
        ax.text(val, i, f' {val:.1f}', va='center', fontweight='bold')
    
    # Win Rate
    ax = axes[1]
    bars = ax.barh(names, win_rates, color=colors, edgecolor='black', linewidth=2)
    ax.set_xlabel('Win Rate (%)', fontweight='bold')
    ax.set_title('🎯 Win Rate', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, win_rates)):
        ax.text(val, i, f' {val:.1f}%', va='center', fontweight='bold')
    
    # Avg Reward
    ax = axes[2]
    bars = ax.barh(names, avg_rewards, color=colors, edgecolor='black', linewidth=2)
    ax.set_xlabel('Average Reward', fontweight='bold')
    ax.set_title('💰 Average Reward', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, avg_rewards)):
        ax.text(val, i, f' {val:.1f}', va='center', fontweight='bold')
    
    plt.suptitle('⚡ INSTANT TOURNAMENT - 50,000 GAMES ⚡', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / 'instant_tournament.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {save_path}")
    plt.show()
    
except ImportError:
    print("\n(matplotlib not available for visualization)")

print("\n💡 Scale up options:")
print("   • 1 Million games:  python examples/fast_tournament.py")
print("   • 10 Million games: python examples/ultimate_tournament.py")
print(f"\n⚡ This instant demo proves the system works!\n")

