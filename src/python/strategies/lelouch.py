"""
Lelouch Strategy - Strategic Streak-Based Betting

Personality: Brilliant strategist. Adjusts bets based on win/loss patterns.
Escalates on winning streaks, retreats after losses.

Core Mechanics:
1. Base bet normally
2. 2+ win streak → increase bet by 50%
3. Loss after big bet → drop to safety bet ($2.50)
4. Targets 6-8 clicks (adaptive based on confidence)
5. Pattern-seeking click selection

Mathematical Foundation:
- Target range: 6-8 clicks (win prob = 45-57%, payout = 1.72-2.12x)
- Streak-based bankroll management
- Risk escalation on winning momentum
"""

from typing import Dict, Any, Optional
import random

from .base import StrategyBase
from game.math import win_probability, expected_value, get_observed_payout


class LelouchStrategy(StrategyBase):
    """
    Lelouch - Strategic betting with streak-based escalation.
    
    Key Features:
    - Streak-aware bet sizing
    - 2+ wins → 1.5x bet increase
    - Loss after big bet → safety bet
    - Adaptive target selection (6-8 clicks)
    - Pattern-seeking intelligence
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 rng: Optional[random.Random] = None):
        """
        Initialize Lelouch strategy.
        
        Config parameters:
            - base_bet: Base bet amount (default: 10.0)
            - safety_bet: Safety bet after big loss (default: 2.50)
            - max_bet: Maximum bet allowed (default: 100.0)
            - min_target: Minimum clicks (default: 6)
            - max_target: Maximum clicks (default: 8)
            - streak_multiplier: Bet increase on streak (default: 1.5)
            - initial_bankroll: Starting bankroll (default: 1000.0)
        """
        super().__init__(config, rng)
        
        self.name = "Lelouch vi Britannia"
        self.safety_bet = self.config.get('safety_bet', 2.50)
        self.min_target = self.config.get('min_target', 6)
        self.max_target = self.config.get('max_target', 8)
        self.streak_multiplier = self.config.get('streak_multiplier', 1.5)
        
        # Betting state
        self.current_bet = self.base_bet
        self.in_safety_mode = False
        self.last_bet_was_big = False
        
        # Strategic state
        self.confidence = 1.0  # Affects target selection
        
    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decision using Lelouch's strategic analysis.
        
        Args:
            state: Current game state
            
        Returns:
            Decision dictionary
        """
        # If game not active, place bet
        if not state.get('game_active', False):
            bet_amount = self._calculate_strategic_bet()
            target = self._determine_strategic_target()
            return {
                "action": "bet",
                "bet": bet_amount,
                "max_clicks": target
            }
        
        # In-game: strategic clicking
        clicks_made = state.get('clicks_made', 0)
        target = self._get_current_target(state)
        current_mult = state.get('current_multiplier', 1.0)
        
        # Strategic decision: continue or take profit?
        if clicks_made >= target:
            return {"action": "cashout"}
        
        # Early cashout if multiplier looks good and confidence is low
        if clicks_made >= self.min_target and self.confidence < 0.7 and current_mult >= 1.7:
            return {"action": "cashout"}
        
        # Continue clicking
        position = self._choose_strategic_position(state)
        return {
            "action": "click",
            "position": position
        }
    
    def _calculate_strategic_bet(self) -> float:
        """
        Calculate bet based on streak analysis and risk management.
        
        Returns:
            Bet amount
        """
        if self.in_safety_mode:
            # Safety mode after big loss
            return min(self.safety_bet, self.bankroll)
        
        # Base bet
        bet = self.base_bet
        
        # Escalate on win streaks
        if self.win_streak >= 2:
            # Apply streak multiplier
            multiplier = 1 + (self.win_streak - 1) * (self.streak_multiplier - 1)
            bet = self.base_bet * min(multiplier, 3.0)  # Cap at 3x
            self.last_bet_was_big = (bet > self.base_bet * 1.5)
        else:
            self.last_bet_was_big = False
        
        # Apply confidence scaling
        bet = bet * self.confidence
        
        # Enforce limits
        bet = max(self.safety_bet, min(bet, self.max_bet, self.bankroll))
        
        return bet
    
    def _determine_strategic_target(self) -> int:
        """
        Determine target clicks based on confidence and streak.
        
        Returns:
            Target click count
        """
        # High confidence + winning streak → aggressive
        if self.win_streak >= 2 and self.confidence >= 0.8:
            return self.max_target
        
        # Low confidence → conservative
        if self.confidence < 0.5 or self.loss_streak >= 2:
            return self.min_target
        
        # Balanced approach
        return (self.min_target + self.max_target) // 2
    
    def _get_current_target(self, state: Dict[str, Any]) -> int:
        """
        Get target for current game (set at start).
        
        Args:
            state: Current game state
            
        Returns:
            Target clicks
        """
        # Would normally store this at game start, but for now recalculate
        return self._determine_strategic_target()
    
    def _choose_strategic_position(self, state: Dict[str, Any]) -> tuple:
        """
        Choose position using pattern-seeking intelligence.
        
        Lelouch analyzes revealed patterns and avoids clustering.
        
        Args:
            state: Current game state
            
        Returns:
            (row, col) tuple
        """
        board_size = state.get('board_size', 5)
        revealed = state.get('revealed', [[False] * board_size for _ in range(board_size)])
        
        # Find unrevealed positions
        unrevealed = []
        isolated_positions = []  # Positions far from revealed tiles
        
        for r in range(board_size):
            for c in range(board_size):
                if not revealed[r][c]:
                    pos = (r, c)
                    unrevealed.append(pos)
                    
                    # Check if position is isolated (no adjacent revealed tiles)
                    is_isolated = True
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < board_size and 0 <= nc < board_size:
                                if revealed[nr][nc]:
                                    is_isolated = False
                                    break
                        if not is_isolated:
                            break
                    
                    if is_isolated:
                        isolated_positions.append(pos)
        
        # Strategic preference: isolated positions (spread out pattern)
        if isolated_positions and self.rng.random() < 0.6:
            return self.rng.choice(isolated_positions)
        elif unrevealed:
            return self.rng.choice(unrevealed)
        
        return (0, 0)
    
    def on_result(self, result: Dict[str, Any]) -> None:
        """
        Process result and update strategic state.
        
        Args:
            result: Game result dictionary
        """
        # Update base tracking
        self.update_tracking(result)
        
        won = result.get('win', False)
        
        if won:
            # Win increases confidence
            self.confidence = min(self.confidence + 0.1, 1.0)
            
            # Exit safety mode on win
            if self.in_safety_mode:
                self.in_safety_mode = False
            
        else:
            # Loss decreases confidence
            self.confidence = max(self.confidence - 0.15, 0.3)
            
            # Enter safety mode if last bet was big
            if self.last_bet_was_big:
                self.in_safety_mode = True
    
    def reset(self) -> None:
        """Reset strategy state for new session."""
        super().reset()
        self.current_bet = self.base_bet
        self.in_safety_mode = False
        self.last_bet_was_big = False
        self.confidence = 1.0
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize strategy state."""
        data = super().serialize()
        data.update({
            'current_bet': self.current_bet,
            'in_safety_mode': self.in_safety_mode,
            'last_bet_was_big': self.last_bet_was_big,
            'confidence': self.confidence,
            'min_target': self.min_target,
            'max_target': self.max_target,
        })
        return data
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Load strategy state."""
        super().deserialize(data)
        self.current_bet = data.get('current_bet', self.base_bet)
        self.in_safety_mode = data.get('in_safety_mode', False)
        self.last_bet_was_big = data.get('last_bet_was_big', False)
        self.confidence = data.get('confidence', 1.0)
        self.min_target = data.get('min_target', 6)
        self.max_target = data.get('max_target', 8)


if __name__ == "__main__":
    # Test Lelouch strategy
    strategy = LelouchStrategy(config={
        'base_bet': 10.0,
        'safety_bet': 2.50,
        'max_bet': 100.0,
        'initial_bankroll': 1000.0,
        'min_target': 6,
        'max_target': 8,
        'streak_multiplier': 1.5
    })
    
    print(f"Strategy: {strategy.name}")
    print(f"Target range: {strategy.min_target}-{strategy.max_target} clicks")
    print(f"Streak multiplier: {strategy.streak_multiplier}x")
    print(f"Safety bet: ${strategy.safety_bet}")
    
    # Simulate win streak
    print("\n--- Simulating win streak ---")
    for i in range(5):
        bet = strategy._calculate_strategic_bet()
        target = strategy._determine_strategic_target()
        print(f"Game {i+1}: Bet ${bet:.2f}, Target {target}, "
              f"Confidence {strategy.confidence:.2f}, "
              f"Streak: {strategy.win_streak}")
        
        # Simulate win
        result = {
            'win': True,
            'payout': 1.95,
            'clicks': target,
            'profit': bet * 0.95,
            'final_bankroll': strategy.bankroll + bet * 0.95,
            'bet_amount': bet
        }
        strategy.on_result(result)
    
    # Simulate big loss
    print("\n--- Big loss after streak ---")
    bet = strategy._calculate_strategic_bet()
    result = {
        'win': False,
        'payout': 0.0,
        'clicks': 3,
        'profit': -bet,
        'final_bankroll': strategy.bankroll - bet,
        'bet_amount': bet
    }
    strategy.on_result(result)
    
    print(f"\nAfter loss: Safety mode = {strategy.in_safety_mode}")
    print(f"Next bet: ${strategy._calculate_strategic_bet():.2f}")
    print(f"Final state: {strategy}")

