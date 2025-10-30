#!/usr/bin/env python3
"""
Character Tournament - Mines Game

Runs a comprehensive tournament with all 5 character strategies:
- Takeshi Kovacs (Aggressive Doubling)
- Yuzu (Controlled Chaos)
- Aoi (Cooperative Sync)
- Kazuya (Dagger Strike)
- Lelouch vi Britannia (Strategic Mastermind)

Usage:
    python character_tournament.py [--num-games N] [--seed S] [--output FILE]
"""

import sys
import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Add src/python to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src' / 'python'))

from strategies import (
    TakeshiStrategy,
    YuzuStrategy,
    AoiStrategy,
    KazuyaStrategy,
    LelouchStrategy,
)

# Import DRL strategy with fallback
try:
    from strategies.drl_strategy import DRLStrategy
    DRL_AVAILABLE = True
except ImportError:
    DRL_AVAILABLE = False
    print("Warning: DRL strategy not available (torch may be missing)")
from game.math import win_probability, expected_value, get_observed_payout


@dataclass
class StrategyResult:
    """Results for a single strategy."""
    name: str
    games_played: int
    wins: int
    losses: int
    win_rate: float
    total_profit: float
    avg_profit: float
    final_bankroll: float
    max_drawdown: float
    target_clicks: int


class MinesGameSimulator:
    """Simple Mines game simulator for tournament."""
    
    def __init__(self, board_size: int = 5, mine_count: int = 2, 
                 initial_bankroll: float = 1000.0):
        """Initialize simulator."""
        self.board_size = board_size
        self.mine_count = mine_count
        self.initial_bankroll = initial_bankroll
        self.rng = random.Random()
    
    def run_game(self, strategy, seed: int = None) -> Dict[str, Any]:
        """Run a single game with given strategy."""
        if seed is not None:
            self.rng.seed(seed)
        
        # Create board with mines
        positions = list(range(self.board_size * self.board_size))
        self.rng.shuffle(positions)
        mine_positions = set(positions[:self.mine_count])
        
        # Initialize game state
        revealed = [[False] * self.board_size for _ in range(self.board_size)]
        board = [[False] * self.board_size for _ in range(self.board_size)]
        
        for pos in mine_positions:
            row = pos // self.board_size
            col = pos % self.board_size
            board[row][col] = True  # True = mine
        
        # Get bet decision
        state = {
            'revealed': revealed,
            'board': board,
            'board_size': self.board_size,
            'mine_count': self.mine_count,
            'clicks_made': 0,
            'current_multiplier': 1.0,
            'bankroll': strategy.bankroll,
            'game_active': False
        }
        
        decision = strategy.decide(state)
        
        if decision.get('action') != 'bet':
            return None  # Invalid
        
        bet_amount = decision.get('bet', strategy.base_bet)
        max_clicks = decision.get('max_clicks', 5)
        
        # Play game
        clicks_made = 0
        hit_mine = False
        
        state['game_active'] = True
        
        while clicks_made < max_clicks and not hit_mine:
            state['clicks_made'] = clicks_made
            state['current_multiplier'] = get_observed_payout(clicks_made + 1) if clicks_made < 15 else 1.0
            
            decision = strategy.decide(state)
            
            if decision.get('action') == 'cashout':
                break
            
            if decision.get('action') != 'click':
                break  # Invalid action
            
            row, col = decision.get('position', (0, 0))
            
            # Check bounds
            if not (0 <= row < self.board_size and 0 <= col < self.board_size):
                break
            
            # Check already revealed
            if revealed[row][col]:
                continue  # Skip already revealed
            
            # Reveal tile
            revealed[row][col] = True
            clicks_made += 1
            
            # Check if mine
            if board[row][col]:
                hit_mine = True
                break
        
        # Calculate result
        won = not hit_mine and clicks_made > 0
        payout_mult = get_observed_payout(clicks_made) if won else 0.0
        profit = (payout_mult * bet_amount - bet_amount) if won else -bet_amount
        
        result = {
            'win': won,
            'payout': payout_mult,
            'clicks': clicks_made,
            'profit': profit,
            'final_bankroll': strategy.bankroll + profit,
            'bet_amount': bet_amount
        }
        
        # Update strategy
        strategy.on_result(result)
        
        return result


def run_tournament(num_games: int = 50000, seed: int = 42, 
                   verbose: bool = True) -> Dict[str, StrategyResult]:
    """
    Run tournament with all character strategies.
    
    Args:
        num_games: Number of games per strategy
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary mapping strategy name to results
    """
    # Initialize strategies
    strategies = {
        'Takeshi Kovacs': TakeshiStrategy(config={
            'base_bet': 10.0,
            'target_clicks': 8,
            'initial_bankroll': 1000.0
        }),
        'Yuzu': YuzuStrategy(config={
            'base_bet': 10.0,
            'target_clicks': 7,
            'initial_bankroll': 1000.0
        }),
        'Aoi': AoiStrategy(config={
            'base_bet': 10.0,
            'min_target_clicks': 6,
            'max_target_clicks': 7,
            'initial_bankroll': 1000.0
        }),
        'Kazuya': KazuyaStrategy(config={
            'base_bet': 10.0,
            'normal_target': 5,
            'dagger_target': 10,
            'initial_bankroll': 1000.0
        }),
        'Lelouch vi Britannia': LelouchStrategy(config={
            'base_bet': 10.0,
            'min_target': 6,
            'max_target': 8,
            'initial_bankroll': 1000.0
        }),
    }
    
    # Add DRL strategy if available
    if DRL_AVAILABLE:
        strategies['DRLStrategy'] = DRLStrategy(config={
            'base_bet': 10.0,
            'n_cells': 25,
            'episodes_trained': 1000,  # Reduced for faster tournament
            'initial_bankroll': 1000.0
        })
    
    results = {}
    simulator = MinesGameSimulator()
    
    if verbose:
        print(f"🏆 CHARACTER TOURNAMENT - {num_games} games per strategy")
        print(f"=" * 70)
    
    for name, strategy in strategies.items():
        if verbose:
            print(f"\n▶ Running {name}...", end='', flush=True)
        
        start_time = time.time()
        wins = 0
        total_profit = 0.0
        max_bankroll = strategy.bankroll
        max_drawdown = 0.0
        
        for i in range(num_games):
            result = simulator.run_game(strategy, seed=seed + i)
            
            if result and result.get('win', False):
                wins += 1
            
            if result:
                total_profit += result.get('profit', 0.0)
            
            # Track drawdown
            max_bankroll = max(max_bankroll, strategy.bankroll)
            drawdown = max_bankroll - strategy.bankroll
            max_drawdown = max(max_drawdown, drawdown)
            
            if verbose and (i + 1) % (num_games // 10) == 0:
                print('.', end='', flush=True)
        
        elapsed = time.time() - start_time
        
        # Compile results
        result = StrategyResult(
            name=name,
            games_played=num_games,
            wins=wins,
            losses=num_games - wins,
            win_rate=wins / num_games,
            total_profit=total_profit,
            avg_profit=total_profit / num_games,
            final_bankroll=strategy.bankroll,
            max_drawdown=max_drawdown,
            target_clicks=getattr(strategy, 'target_clicks', 
                                 getattr(strategy, 'normal_target', 0))
        )
        
        results[name] = result
        
        if verbose:
            print(f" ✓ ({elapsed:.1f}s)")
    
    return results


def print_results(results: Dict[str, StrategyResult]):
    """Print formatted tournament results."""
    print(f"\n{'='*70}")
    print(f"{'TOURNAMENT RESULTS':^70}")
    print(f"{'='*70}\n")
    
    # Sort by total profit
    sorted_results = sorted(results.items(), key=lambda x: x[1].total_profit, reverse=True)
    
    print(f"{'Rank':<6} {'Strategy':<25} {'Win Rate':<12} {'Net Profit':<15} {'ROI':<10}")
    print(f"{'-'*70}")
    
    for rank, (name, result) in enumerate(sorted_results, 1):
        roi = (result.total_profit / (result.games_played * 10)) * 100  # Assuming $10 avg bet
        emoji = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        
        print(f"{emoji} {rank:<4} {name:<25} {result.win_rate:>10.2%}  "
              f"${result.total_profit:>12.2f}  {roi:>8.2f}%")
    
    print(f"{'-'*70}\n")
    
    # Detailed statistics
    print("Detailed Statistics:")
    print(f"{'='*70}\n")
    
    for name, result in sorted_results:
        print(f"{name}:")
        print(f"  Games Played: {result.games_played:,}")
        print(f"  Wins/Losses: {result.wins:,} / {result.losses:,}")
        print(f"  Win Rate: {result.win_rate:.2%}")
        print(f"  Total Profit: ${result.total_profit:.2f}")
        print(f"  Avg Profit/Game: ${result.avg_profit:.4f}")
        print(f"  Final Bankroll: ${result.final_bankroll:.2f}")
        print(f"  Max Drawdown: ${result.max_drawdown:.2f}")
        print(f"  Target Clicks: {result.target_clicks}")
        print()


def save_results(results: Dict[str, StrategyResult], filepath: str):
    """Save results to JSON file."""
    data = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'tournament_type': 'character_strategies',
        'strategies': {}
    }
    
    for name, result in results.items():
        data['strategies'][name] = {
            'strategy_name': result.name,
            'games_played': result.games_played,
            'wins': result.wins,
            'losses': result.losses,
            'win_rate': result.win_rate,
            'total_profit': result.total_profit,
            'avg_profit': result.avg_profit,
            'final_bankroll': result.final_bankroll,
            'max_drawdown': result.max_drawdown,
            'target_clicks': result.target_clicks,
        }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to: {filepath}")


def main():
    """Main tournament runner."""
    parser = argparse.ArgumentParser(description='Run character strategy tournament')
    parser.add_argument('--num-games', type=int, default=50000,
                       help='Number of games per strategy (default: 50000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Run tournament
    results = run_tournament(
        num_games=args.num_games,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    # Print results
    if not args.quiet:
        print_results(results)
    
    # Save to file if requested
    if args.output:
        save_results(results, args.output)


if __name__ == '__main__':
    main()

