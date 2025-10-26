"""
Aoi Strategy - Cooperative Sync

Personality: Team player who adapts based on peer performance.
After 3 consecutive losses, "syncs" with the best performer.

Core Mechanics:
1. Moderate betting with 6-7 click range
2. Sync mechanism after 3 losses (copies best strategy's behavior)
3. Adaptive target selection based on recent success
4. Conservative during sync phase

Mathematical Foundation:
- Primary targets: 6-7 clicks (win prob = 51-57%, payout = 1.72-1.95x)
- Balanced risk/reward profile
- Syncs to proven patterns
"""

from typing import Dict, Any, Optional
import random

from .base import StrategyBase
from game.math import win_probability, expected_value, get_observed_payout


class AoiStrategy(StrategyBase):
    """
    Aoi - Cooperative strategy with sync mechanism.
    
    Key Features:
    - Targets 6-7 clicks (adaptive based on performance)
    - Sync mode after 3 consecutive losses
    - Copies successful patterns from "team"
    - Conservative bet sizing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 rng: Optional[random.Random] = None,
                 registry: Optional[Any] = None):
        """
        Initialize Aoi strategy.
        
        Config parameters:
            - base_bet: Base bet amount (default: 10.0)
            - max_bet: Maximum bet allowed (default: 80.0)
            - min_target_clicks: Minimum clicks (default: 6)
            - max_target_clicks: Maximum clicks (default: 7)
            - sync_threshold: Losses before sync (default: 3)
            - initial_bankroll: Starting bankroll (default: 1000.0)
            
        Args:
            config: Configuration dictionary
            rng: Random number generator
            registry: Optional registry to access other strategies for sync
        """
        super().__init__(config, rng)
        
        self.name = "Aoi"
        self.min_target_clicks = self.config.get('min_target_clicks', 6)
        self.max_target_clicks = self.config.get('max_target_clicks', 7)
        self.sync_threshold = self.config.get('sync_threshold', 3)
        
        # Current target
        self.current_target = self.max_target_clicks
        
        # Sync state
        self.in_sync_mode = False
        self.sync_rounds_remaining = 0
        self.synced_target = None
        
        # Registry for accessing other strategies
        self.registry = registry
        
    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decision using Aoi's cooperative strategy.
        
        Args:
            state: Current game state
            
        Returns:
            Decision dictionary
        """
        # If game not active, place bet
        if not state.get('game_active', False):
            bet_amount = self._calculate_bet_amount()
            target = self._determine_target_clicks()
            return {
                "action": "bet",
                "bet": bet_amount,
                "max_clicks": target
            }
        
        # In-game: continue until target
        clicks_made = state.get('clicks_made', 0)
        target = self.synced_target if self.in_sync_mode else self.current_target
        
        if clicks_made < target:
            position = self._choose_position(state)
            return {
                "action": "click",
                "position": position
            }
        else:
            # Reached target - cash out
            return {"action": "cashout"}
    
    def _calculate_bet_amount(self) -> float:
        """
        Calculate conservative bet amount.
        
        Returns:
            Bet amount
        """
        # Conservative during sync mode
        if self.in_sync_mode:
            bet = self.base_bet * 0.8
        else:
            bet = self.base_bet
        
        return min(bet, self.max_bet, self.bankroll)
    
    def _determine_target_clicks(self) -> int:
        """
        Determine target number of clicks.
        
        Returns:
            Target click count
        """
        if self.in_sync_mode and self.synced_target:
            return self.synced_target
        
        # Adaptive: adjust based on recent performance
        if len(self.result_history) >= 5:
            recent_wins = sum(1 for r in self.result_history[-5:] if r.get('win', False))
            if recent_wins >= 4:
                # Doing well - try for more clicks
                self.current_target = self.max_target_clicks
            elif recent_wins <= 1:
                # Struggling - be more conservative
                self.current_target = self.min_target_clicks
        
        return self.current_target
    
    def _choose_position(self, state: Dict[str, Any]) -> tuple:
        """
        Choose position with slight preference for center.
        
        Aoi prefers balanced, central positions.
        
        Args:
            state: Current game state
            
        Returns:
            (row, col) tuple
        """
        board_size = state.get('board_size', 5)
        revealed = state.get('revealed', [[False] * board_size for _ in range(board_size)])
        
        # Find unrevealed positions
        unrevealed = []
        center_positions = []
        
        center = board_size // 2
        
        for r in range(board_size):
            for c in range(board_size):
                if not revealed[r][c]:
                    pos = (r, c)
                    unrevealed.append(pos)
                    
                    # Identify center region (middle 3x3 for 5x5 board)
                    if abs(r - center) <= 1 and abs(c - center) <= 1:
                        center_positions.append(pos)
        
        # Prefer center positions 50% of the time
        if center_positions and self.rng.random() < 0.5:
            return self.rng.choice(center_positions)
        elif unrevealed:
            return self.rng.choice(unrevealed)
        
        return (0, 0)
    
    def _activate_sync(self) -> None:
        """
        Activate sync mode - analyze and copy best performer.
        
        If registry is available, finds the best performing strategy
        and syncs to their target click count.
        """
        self.in_sync_mode = True
        self.sync_rounds_remaining = 5  # Sync for 5 games
        
        # If no registry, use default sync target
        if not self.registry:
            self.synced_target = self.min_target_clicks  # Conservative default
            return
        
        # Try to find best performer from registry
        try:
            best_strategy = self.registry.get_best_strategy()
            if best_strategy and hasattr(best_strategy, 'target_clicks'):
                self.synced_target = best_strategy.target_clicks
            else:
                self.synced_target = self.min_target_clicks
        except:
            self.synced_target = self.min_target_clicks
    
    def on_result(self, result: Dict[str, Any]) -> None:
        """
        Process result and handle sync mechanism.
        
        Args:
            result: Game result dictionary
        """
        # Update base tracking
        self.update_tracking(result)
        
        won = result.get('win', False)
        
        if won:
            # Win resets sync counter
            if self.in_sync_mode:
                self.sync_rounds_remaining -= 1
                if self.sync_rounds_remaining <= 0:
                    self.in_sync_mode = False
                    self.synced_target = None
        else:
            # Loss handling
            if self.in_sync_mode:
                # Already syncing
                self.sync_rounds_remaining -= 1
                if self.sync_rounds_remaining <= 0:
                    self.in_sync_mode = False
                    self.synced_target = None
            elif self.loss_streak >= self.sync_threshold:
                # Trigger sync after threshold losses
                self._activate_sync()
    
    def reset(self) -> None:
        """Reset strategy state for new session."""
        super().reset()
        self.current_target = self.max_target_clicks
        self.in_sync_mode = False
        self.sync_rounds_remaining = 0
        self.synced_target = None
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize strategy state."""
        data = super().serialize()
        data.update({
            'current_target': self.current_target,
            'in_sync_mode': self.in_sync_mode,
            'sync_rounds_remaining': self.sync_rounds_remaining,
            'synced_target': self.synced_target,
        })
        return data
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Load strategy state."""
        super().deserialize(data)
        self.current_target = data.get('current_target', self.max_target_clicks)
        self.in_sync_mode = data.get('in_sync_mode', False)
        self.sync_rounds_remaining = data.get('sync_rounds_remaining', 0)
        self.synced_target = data.get('synced_target', None)


if __name__ == "__main__":
    # Test Aoi strategy
    strategy = AoiStrategy(config={
        'base_bet': 10.0,
        'max_bet': 80.0,
        'initial_bankroll': 1000.0,
        'min_target_clicks': 6,
        'max_target_clicks': 7,
        'sync_threshold': 3
    })
    
    print(f"Strategy: {strategy.name}")
    print(f"Target clicks range: {strategy.min_target_clicks}-{strategy.max_target_clicks}")
    print(f"Sync threshold: {strategy.sync_threshold} losses")
    
    # Simulate loss sequence to trigger sync
    print("\n--- Simulating loss sequence ---")
    for i in range(5):
        target = strategy._determine_target_clicks()
        bet = strategy._calculate_bet_amount()
        print(f"Game {i+1}: Bet ${bet:.2f}, Target {target} clicks "
              f"(sync: {strategy.in_sync_mode}, losses: {strategy.loss_streak})")
        
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
    print(f"Sync activated: {strategy.in_sync_mode}")

