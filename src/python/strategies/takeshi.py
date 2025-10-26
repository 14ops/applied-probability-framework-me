"""
Takeshi Kovacs Strategy - Aggressive Berserker

Personality: Former CTAC Envoy turned street fighter. Aggressive risk-taker
who doubles down after losses but knows when to pause.

Core Mechanics:
1. Martingale-style doubling after losses (with cap)
2. Tranquility mode after 2 consecutive doubles (return to base bet)
3. Targets 8 clicks (high-risk, high-reward zone)
4. Aggressive bet sizing when winning

Mathematical Foundation:
- Target clicks: 8 (win prob = 45.33%, payout = 2.12x)
- EV at 8 clicks with $10 bet: +$4.14
- Doubling sequence: $10 → $20 → $40 → (tranquility) → $10
"""

from typing import Dict, Any, Optional
import random

from .base import StrategyBase
from game.math import win_probability, expected_value, get_observed_payout


class TakeshiStrategy(StrategyBase):
    """
    Takeshi Kovacs - Aggressive berserker with doubling mechanic.
    
    Key Features:
    - Doubles bet after each loss (max 2 doubles)
    - Tranquility mode: Resets to base bet after 2 doubles
    - Always targets 8 clicks for optimal risk/reward
    - Aggressive betting when on winning streaks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 rng: Optional[random.Random] = None):
        """
        Initialize Takeshi strategy.
        
        Config parameters:
            - base_bet: Base bet amount (default: 10.0)
            - max_bet: Maximum bet allowed (default: 100.0)
            - target_clicks: Target number of clicks (default: 8)
            - max_doubles: Maximum consecutive doubles (default: 2)
            - initial_bankroll: Starting bankroll (default: 1000.0)
        """
        super().__init__(config, rng)
        
        self.name = "Takeshi Kovacs"
        self.target_clicks = self.config.get('target_clicks', 8)
        self.max_doubles = self.config.get('max_doubles', 2)
        
        # Doubling tracking
        self.current_bet = self.base_bet
        self.consecutive_losses = 0
        self.in_tranquility_mode = False
        self.tranquility_games_remaining = 0
        
        # Takeshi's aggressive parameters
        self.aggression_multiplier = 1.5  # Bet more when winning
        
    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decision using Takeshi's aggressive doubling strategy.
        
        Args:
            state: Current game state
            
        Returns:
            Decision dictionary with action, bet, position
        """
        # If game not active, place bet
        if not state.get('game_active', False):
            bet_amount = self._calculate_bet_amount()
            return {
                "action": "bet",
                "bet": bet_amount,
                "max_clicks": self.target_clicks
            }
        
        # In-game: decide whether to continue clicking or cashout
        clicks_made = state.get('clicks_made', 0)
        current_multiplier = state.get('current_multiplier', 1.0)
        
        # Always go for target clicks (aggressive!)
        if clicks_made < self.target_clicks:
            position = self._choose_click_position(state)
            return {
                "action": "click",
                "position": position
            }
        else:
            # Reached target, cash out
            return {"action": "cashout"}
    
    def _calculate_bet_amount(self) -> float:
        """
        Calculate bet amount based on doubling strategy.
        
        Returns:
            Bet amount
        """
        if self.in_tranquility_mode:
            # In tranquility mode, use base bet
            return min(self.base_bet, self.bankroll)
        
        # Normal betting: double after losses up to max_doubles
        bet = min(self.current_bet, self.max_bet, self.bankroll)
        return bet
    
    def _choose_click_position(self, state: Dict[str, Any]) -> tuple:
        """
        Choose next click position using aggressive logic.
        
        Takeshi prefers edge/corner positions (more information gain).
        
        Args:
            state: Current game state
            
        Returns:
            (row, col) tuple
        """
        board_size = state.get('board_size', 5)
        revealed = state.get('revealed', [[False] * board_size for _ in range(board_size)])
        
        # Find all unrevealed positions
        unrevealed = []
        edge_positions = []
        corner_positions = []
        
        for r in range(board_size):
            for c in range(board_size):
                if not revealed[r][c]:
                    pos = (r, c)
                    unrevealed.append(pos)
                    
                    # Identify edges and corners
                    is_edge = (r == 0 or r == board_size - 1 or 
                              c == 0 or c == board_size - 1)
                    is_corner = ((r == 0 or r == board_size - 1) and 
                                (c == 0 or c == board_size - 1))
                    
                    if is_corner:
                        corner_positions.append(pos)
                    elif is_edge:
                        edge_positions.append(pos)
        
        # Preference order: corners > edges > random
        if corner_positions and self.rng.random() < 0.6:
            return self.rng.choice(corner_positions)
        elif edge_positions and self.rng.random() < 0.4:
            return self.rng.choice(edge_positions)
        elif unrevealed:
            return self.rng.choice(unrevealed)
        
        # Fallback
        return (0, 0)
    
    def on_result(self, result: Dict[str, Any]) -> None:
        """
        Process result and update doubling strategy.
        
        Args:
            result: Game result dictionary
        """
        # Update base tracking
        self.update_tracking(result)
        
        won = result.get('win', False)
        
        if won:
            # Victory! Reset doubling counter
            self.consecutive_losses = 0
            
            # Increase bet slightly on wins (aggression)
            if self.win_streak >= 2:
                self.current_bet = min(
                    self.base_bet * self.aggression_multiplier,
                    self.max_bet
                )
            else:
                self.current_bet = self.base_bet
            
            # Exit tranquility mode on win
            self.in_tranquility_mode = False
            self.tranquility_games_remaining = 0
            
        else:
            # Loss - implement doubling mechanic
            self.consecutive_losses += 1
            
            if self.in_tranquility_mode:
                # In tranquility, just count down
                self.tranquility_games_remaining -= 1
                if self.tranquility_games_remaining <= 0:
                    self.in_tranquility_mode = False
                    self.consecutive_losses = 0
                self.current_bet = self.base_bet
                
            elif self.consecutive_losses <= self.max_doubles:
                # Double the bet (Martingale)
                self.current_bet = min(
                    self.current_bet * 2,
                    self.max_bet,
                    self.bankroll
                )
                
            else:
                # Hit max doubles - enter tranquility mode
                self.in_tranquility_mode = True
                self.tranquility_games_remaining = 3  # 3 games at base bet
                self.current_bet = self.base_bet
                self.consecutive_losses = 0
    
    def reset(self) -> None:
        """Reset strategy state for new session."""
        super().reset()
        self.current_bet = self.base_bet
        self.consecutive_losses = 0
        self.in_tranquility_mode = False
        self.tranquility_games_remaining = 0
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize strategy state."""
        data = super().serialize()
        data.update({
            'current_bet': self.current_bet,
            'consecutive_losses': self.consecutive_losses,
            'in_tranquility_mode': self.in_tranquility_mode,
            'tranquility_games_remaining': self.tranquility_games_remaining,
            'target_clicks': self.target_clicks,
        })
        return data
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Load strategy state."""
        super().deserialize(data)
        self.current_bet = data.get('current_bet', self.base_bet)
        self.consecutive_losses = data.get('consecutive_losses', 0)
        self.in_tranquility_mode = data.get('in_tranquility_mode', False)
        self.tranquility_games_remaining = data.get('tranquility_games_remaining', 0)
        self.target_clicks = data.get('target_clicks', 8)


if __name__ == "__main__":
    # Test Takeshi strategy
    strategy = TakeshiStrategy(config={
        'base_bet': 10.0,
        'max_bet': 100.0,
        'initial_bankroll': 1000.0
    })
    
    print(f"Strategy: {strategy.name}")
    print(f"Target clicks: {strategy.target_clicks}")
    print(f"Win probability: {win_probability(strategy.target_clicks):.2%}")
    print(f"Expected payout: {get_observed_payout(strategy.target_clicks):.2f}x")
    
    # Simulate betting sequence with losses
    print("\n--- Simulating loss sequence ---")
    for i in range(5):
        bet = strategy._calculate_bet_amount()
        print(f"Game {i+1}: Bet ${bet:.2f} "
              f"(losses: {strategy.consecutive_losses}, "
              f"tranquility: {strategy.in_tranquility_mode})")
        
        # Simulate loss
        result = {
            'win': False,
            'payout': 0.0,
            'clicks': 3,
            'profit': -bet,
            'final_bankroll': strategy.bankroll - bet,
            'bet_amount': bet
        }
        strategy.on_result(result)
    
    print(f"\nFinal state: {strategy}")

