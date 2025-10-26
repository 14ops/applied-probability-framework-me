"""
Yuzu Strategy - Controlled Chaos

Personality: Unpredictable yet calculated. Operates in the "sweet spot"
of risk/reward with exactly 7 clicks.

Core Mechanics:
1. Always targets exactly 7 clicks (no more, no less)
2. Variable bet sizing based on "chaos factor"
3. Random position selection (embraces uncertainty)
4. Momentum-based betting (increases on wins)

Mathematical Foundation:
- Target clicks: 7 (win prob = 51.0%, payout = 1.95x)
- EV at 7 clicks with $10 bet: +$5.05
- Optimal risk/reward balance
"""

from typing import Dict, Any, Optional
import random

from .base import StrategyBase
from game.math import win_probability, expected_value, get_observed_payout


class YuzuStrategy(StrategyBase):
    """
    Yuzu - Controlled chaos with fixed 7-click strategy.
    
    Key Features:
    - Always plays exactly 7 clicks
    - Variable bet sizing with "chaos factor"
    - Momentum-based betting on win streaks
    - Random click patterns (embraces RNG)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 rng: Optional[random.Random] = None):
        """
        Initialize Yuzu strategy.
        
        Config parameters:
            - base_bet: Base bet amount (default: 10.0)
            - max_bet: Maximum bet allowed (default: 100.0)
            - target_clicks: Fixed at 7 (Yuzu's sweet spot)
            - chaos_factor: Randomness in betting (default: 0.3)
            - initial_bankroll: Starting bankroll (default: 1000.0)
        """
        super().__init__(config, rng)
        
        self.name = "Yuzu"
        self.target_clicks = 7  # Fixed - Yuzu's signature
        self.chaos_factor = self.config.get('chaos_factor', 0.3)
        
        # Momentum tracking
        self.momentum = 0  # Increases on wins, decreases on losses
        self.max_momentum = 3
        
    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decision using Yuzu's controlled chaos strategy.
        
        Args:
            state: Current game state
            
        Returns:
            Decision dictionary
        """
        # If game not active, place bet
        if not state.get('game_active', False):
            bet_amount = self._calculate_bet_amount()
            return {
                "action": "bet",
                "bet": bet_amount,
                "max_clicks": self.target_clicks
            }
        
        # In-game: continue until exactly 7 clicks
        clicks_made = state.get('clicks_made', 0)
        
        if clicks_made < self.target_clicks:
            position = self._choose_random_position(state)
            return {
                "action": "click",
                "position": position
            }
        else:
            # Exactly 7 clicks reached - cash out
            return {"action": "cashout"}
    
    def _calculate_bet_amount(self) -> float:
        """
        Calculate bet with chaos factor and momentum.
        
        Returns:
            Bet amount
        """
        # Base bet with chaos variation
        chaos_mult = 1.0 + (self.rng.random() * 2 - 1) * self.chaos_factor
        
        # Momentum bonus
        momentum_mult = 1.0 + (self.momentum / self.max_momentum) * 0.5
        
        bet = self.base_bet * chaos_mult * momentum_mult
        bet = max(self.base_bet * 0.5, min(bet, self.max_bet, self.bankroll))
        
        return bet
    
    def _choose_random_position(self, state: Dict[str, Any]) -> tuple:
        """
        Choose completely random unrevealed position.
        
        Yuzu embraces chaos - no pattern seeking.
        
        Args:
            state: Current game state
            
        Returns:
            (row, col) tuple
        """
        board_size = state.get('board_size', 5)
        revealed = state.get('revealed', [[False] * board_size for _ in range(board_size)])
        
        # Find all unrevealed positions
        unrevealed = []
        for r in range(board_size):
            for c in range(board_size):
                if not revealed[r][c]:
                    unrevealed.append((r, c))
        
        if unrevealed:
            return self.rng.choice(unrevealed)
        
        return (0, 0)
    
    def on_result(self, result: Dict[str, Any]) -> None:
        """
        Process result and update momentum.
        
        Args:
            result: Game result dictionary
        """
        # Update base tracking
        self.update_tracking(result)
        
        won = result.get('win', False)
        
        if won:
            # Increase momentum on win
            self.momentum = min(self.momentum + 1, self.max_momentum)
        else:
            # Decrease momentum on loss
            self.momentum = max(self.momentum - 1, -self.max_momentum)
    
    def reset(self) -> None:
        """Reset strategy state for new session."""
        super().reset()
        self.momentum = 0
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize strategy state."""
        data = super().serialize()
        data.update({
            'momentum': self.momentum,
            'target_clicks': self.target_clicks,
            'chaos_factor': self.chaos_factor,
        })
        return data
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Load strategy state."""
        super().deserialize(data)
        self.momentum = data.get('momentum', 0)
        self.chaos_factor = data.get('chaos_factor', 0.3)


if __name__ == "__main__":
    # Test Yuzu strategy
    strategy = YuzuStrategy(config={
        'base_bet': 10.0,
        'max_bet': 50.0,
        'initial_bankroll': 1000.0,
        'chaos_factor': 0.3
    })
    
    print(f"Strategy: {strategy.name}")
    print(f"Target clicks: {strategy.target_clicks} (FIXED)")
    print(f"Win probability: {win_probability(strategy.target_clicks):.2%}")
    print(f"Expected payout: {get_observed_payout(strategy.target_clicks):.2f}x")
    print(f"Chaos factor: {strategy.chaos_factor}")
    
    # Simulate betting sequence
    print("\n--- Simulating bet sequence ---")
    for i in range(5):
        bet = strategy._calculate_bet_amount()
        print(f"Game {i+1}: Bet ${bet:.2f} (momentum: {strategy.momentum})")
        
        # Alternate wins and losses
        won = i % 2 == 0
        result = {
            'win': won,
            'payout': 1.95 if won else 0.0,
            'clicks': 7 if won else 4,
            'profit': bet * 0.95 if won else -bet,
            'final_bankroll': strategy.bankroll + (bet * 0.95 if won else -bet),
            'bet_amount': bet
        }
        strategy.on_result(result)
    
    print(f"\nFinal state: {strategy}")

