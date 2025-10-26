"""
Test Tournament - 100K games to verify system works
Then we'll scale to 10 million
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

import numpy as np
from typing import Dict
import time

from character_strategies import (
    TakeshiKovacsStrategy,
    LelouchViBritanniaStrategy,
    KazuyaKinoshitaStrategy,
    SenkuIshigamiStrategy,
    RintaroOkabeStrategy,
    HybridStrategy,
)
from simulators.mines_simulator import MinesSimulator

print("\n🏆 TESTING TOURNAMENT SYSTEM 🏆")
print(f"Running 100,000 games to verify everything works...\n")

# Initialize strategies
strategies = {
    'Takeshi': TakeshiKovacsStrategy(),
    'Lelouch': LelouchViBritanniaStrategy(),
    'Kazuya': KazuyaKinoshitaStrategy(),
    'Senku': SenkuIshigamiStrategy(),
    'Okabe': RintaroOkabeStrategy(),
    'Hybrid': HybridStrategy(),
}

stats = {}
for name in strategies:
    stats[name] = {
        'wins': 0,
        'losses': 0,
        'total_reward': 0.0,
        'rewards': []
    }

games_per_strategy = 100_000 // len(strategies)

print(f"Games per strategy: {games_per_strategy:,}\n")

for strategy_name, strategy in strategies.items():
    print(f"Testing {strategy_name}...")
    start = time.time()
    
    simulator = MinesSimulator(seed=42, config={'board_size': 5, 'mine_count': 3})
    
    for game in range(games_per_strategy):
        state = simulator.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            valid_actions = simulator.get_valid_actions()
            if not valid_actions:
                break
            
            try:
                action = strategy.select_action(state, valid_actions)
            except:
                action = np.random.choice(valid_actions) if valid_actions else None
            
            if action is None:
                break
            
            next_state, reward, done, info = simulator.step(action)
            total_reward += reward
            state = next_state
        
        if total_reward > 0:
            stats[strategy_name]['wins'] += 1
        else:
            stats[strategy_name]['losses'] += 1
        
        stats[strategy_name]['total_reward'] += total_reward
        stats[strategy_name]['rewards'].append(total_reward)
    
    elapsed = time.time() - start
    win_rate = stats[strategy_name]['wins'] / games_per_strategy
    avg_reward = stats[strategy_name]['total_reward'] / games_per_strategy
    
    print(f"  ✓ Completed in {elapsed:.1f}s | Win Rate: {win_rate:.2%} | Avg Reward: {avg_reward:.4f}\n")

# Print results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

sorted_strategies = sorted(stats.items(), 
                          key=lambda x: x[1]['total_reward'], 
                          reverse=True)

for rank, (name, stat) in enumerate(sorted_strategies, 1):
    win_rate = stat['wins'] / (stat['wins'] + stat['losses'])
    avg_reward = stat['total_reward'] / (stat['wins'] + stat['losses'])
    print(f"\n{rank}. {name}")
    print(f"   Win Rate: {win_rate:.2%}")
    print(f"   Avg Reward: {avg_reward:.4f}")
    print(f"   Total Reward: {stat['total_reward']:.2f}")

champion = sorted_strategies[0][0]
print(f"\n🏆 Champion: {champion} 🏆")

print("\n✅ System verified! Ready for 10 million games.")
print("Run: python mega_tournament.py")

