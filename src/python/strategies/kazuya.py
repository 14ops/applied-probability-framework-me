"""
Kazuya Strategy - Conservative with Periodic Strikes

Personality: Disciplined martial artist. Plays conservatively most of the time,
but every N rounds executes a "Dagger Strike" with high-risk play.

Core Mechanics:
1. Always uses base bet (no progression)
2. Targets 5-6 clicks normally (conservative)
3. Every 6 rounds: "Dagger Strike" with 10+ clicks
4. Methodical position selection

Mathematical Foundation:
- Normal mode: 5-6 clicks (win prob = 63-57%, payout = 1.53-1.72x)
- Dagger Strike: 10+ clicks (win prob = 35%, payout = 3.02x+)
- EV-optimized blend of safety and aggression
"""

from typing import Dict, Any, Optional
import random

from .base import StrategyBase
from game.math import win_probability, expected_value, get_observed_payout


class KazuyaStrategy(StrategyBase):
    """
    Kazuya - Conservative strategy with periodic high-risk "Dagger Strike".
    
    Key Features:
    - Flat betting (always base bet)
    - Conservative 5-6 clicks normally
    - Every Nth round: Dagger Strike (10+ clicks)
    - Methodical click patterns
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 rng: Optional[random.Random] = None):
        """
        Initialize Kazuya strategy.
        
        Config parameters:
            - base_bet: Base bet amount (default: 10.0)
            - normal_target: Normal click target (default: 5)
            - dagger_target: Dagger strike clicks (default: 10)
            - dagger_interval: Rounds between daggers (default: 6)
            - initial_bankroll: Starting bankroll (default: 1000.0)
        """
        super().__init__(config, rng)
        
        self.name = "Kazuya"
        self.normal_target = self.config.get('normal_target', 5)
        self.dagger_target = self.config.get('dagger_target', 10)
        self.dagger_interval = self.config.get('dagger_interval', 6)
        
        # Dagger strike tracking
        self.rounds_until_dagger = self.dagger_interval
        
    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decision using Kazuya's disciplined strategy.
        
        Args:
            state: Current game state
            
        Returns:
            Decision dictionary
        """
        # If game not active, place bet
        if not state.get('game_active', False):
            bet_amount = self.base_bet  # Always flat bet
            target = self._determine_target()
            return {
                "action": "bet",
                "bet": min(bet_amount, self.bankroll),
                "max_clicks": target
            }
        
        # In-game: methodical clicking until target
        clicks_made = state.get('clicks_made', 0)
        target = self._current_game_target(state)
        
        if clicks_made < target:
            position = self._choose_methodical_position(state)
            return {
                "action": "click",
                "position": position
            }
        else:
            # Reached target - cash out with discipline
            return {"action": "cashout"}
    
    def _determine_target(self) -> int:
        """
        Determine target based on dagger strike timing.
        
        Returns:
            Target click count
        """
        if self.rounds_until_dagger <= 0:
            # DAGGER STRIKE!
            return self.dagger_target
        else:
            # Normal conservative play
            return self.normal_target
    
    def _current_game_target(self, state: Dict[str, Any]) -> int:
        """
        Get current game's target (stored at game start).
        
        Args:
            state: Current game state
            
        Returns:
            Target clicks for current game
        """
        # Check if we're in dagger mode this game
        if self.rounds_until_dagger <= 0:
            return self.dagger_target
        return self.normal_target
    
    def _choose_methodical_position(self, state: Dict[str, Any]) -> tuple:
        """
        Choose position using methodical scanning pattern.
        
        Kazuya prefers systematic top-to-bottom, left-to-right scanning.
        
        Args:
            state: Current game state
            
        Returns:
            (row, col) tuple
        """
        board_size = state.get('board_size', 5)
        revealed = state.get('revealed', [[False] * board_size for _ in range(board_size)])
        
        # Scan in order: top-to-bottom, left-to-right
        for r in range(board_size):
            for c in range(board_size):
                if not revealed[r][c]:
                    return (r, c)
        
        # Fallback (shouldn't reach here)
        return (0, 0)
    
    def on_result(self, result: Dict[str, Any]) -> None:
        """
        Process result and update dagger strike counter.
        
        Args:
            result: Game result dictionary
        """
        # Update base tracking
        self.update_tracking(result)
        
        # Update dagger strike counter
        if self.rounds_until_dagger <= 0:
            # Just completed dagger strike - reset counter
            self.rounds_until_dagger = self.dagger_interval
        else:
            # Decrement counter
            self.rounds_until_dagger -= 1
    
    def reset(self) -> None:
        """Reset strategy state for new session."""
        super().reset()
        self.rounds_until_dagger = self.dagger_interval
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize strategy state."""
        data = super().serialize()
        data.update({
            'rounds_until_dagger': self.rounds_until_dagger,
            'normal_target': self.normal_target,
            'dagger_target': self.dagger_target,
            'dagger_interval': self.dagger_interval,
        })
        return data
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Load strategy state."""
        super().deserialize(data)
        self.rounds_until_dagger = data.get('rounds_until_dagger', self.dagger_interval)
        self.normal_target = data.get('normal_target', 5)
        self.dagger_target = data.get('dagger_target', 10)
        self.dagger_interval = data.get('dagger_interval', 6)


if __name__ == "__main__":
    # Test Kazuya strategy
    strategy = KazuyaStrategy(config={
        'base_bet': 10.0,
        'initial_bankroll': 1000.0,
        'normal_target': 5,
        'dagger_target': 10,
        'dagger_interval': 6
    })
    
    print(f"Strategy: {strategy.name}")
    print(f"Normal target: {strategy.normal_target} clicks")
    print(f"Dagger target: {strategy.dagger_target} clicks")
    print(f"Dagger interval: every {strategy.dagger_interval} rounds")
    
    print(f"\nNormal play:")
    print(f"  Win prob: {win_probability(strategy.normal_target):.2%}")
    print(f"  Payout: {get_observed_payout(strategy.normal_target):.2f}x")
    
    print(f"\nDagger Strike:")
    print(f"  Win prob: {win_probability(strategy.dagger_target):.2%}")
    print(f"  Payout: {get_observed_payout(strategy.dagger_target):.2f}x")
    
    # Simulate game sequence
    print("\n--- Simulating game sequence ---")
    for i in range(8):
        target = strategy._determine_target()
        mode = "DAGGER STRIKE" if target == strategy.dagger_target else "Normal"
        print(f"Game {i+1}: Target {target} clicks ({mode}), "
              f"rounds until dagger: {strategy.rounds_until_dagger}")
        
        # Simulate win
        result = {
            'win': True,
            'payout': get_observed_payout(target),
            'clicks': target,
            'profit': strategy.base_bet * (get_observed_payout(target) - 1),
            'final_bankroll': strategy.bankroll + strategy.base_bet * (get_observed_payout(target) - 1),
            'bet_amount': strategy.base_bet
        }
        strategy.on_result(result)
    
    print(f"\nFinal state: {strategy}")

