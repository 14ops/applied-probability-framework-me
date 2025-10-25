"""
Kazuya Kinoshita - The Conservative Survivor

Core Philosophy: "Survival is paramount. Every loss is a lesson, every gain is a blessing."
Risk aversion and capital preservation with slow, steady growth.
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from core.base_strategy import BaseStrategy


class KazuyaKinoshitaStrategy(BaseStrategy):
    """
    The Conservative Survivor - Extreme risk aversion and capital preservation.
    
    Key Features:
    - Maximum Tolerable Risk (MTR) threshold
    - Safest cell prioritization
    - Strict stop-loss and win-target
    - Dynamic board state risk assessment
    """
    
    register = True
    plugin_name = "kazuya"
    
    def __init__(self, name: str = "Kazuya Kinoshita - Conservative Survivor", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
        # Risk parameters
        self.max_tolerable_risk = config.get('max_tolerable_risk', 0.20) if config else 0.20
        self.stop_loss = config.get('stop_loss', 0.10) if config else 0.10  # 10% of bankroll
        self.win_target = config.get('win_target', 0.05) if config else 0.05  # 5% of bankroll
        
        # Tracking
        self.starting_bankroll = 1000.0
        self.current_bankroll = 1000.0
        self.current_profit = 0.0
        self.current_loss = 0.0
        
    def select_action(self, state: Any, valid_actions: Any) -> Any:
        """
        Select the safest cell with positive EV, or cash out if risk is too high.
        
        Args:
            state: Current game state
            valid_actions: List of valid (row, col) tuples
            
        Returns:
            (row, col) tuple or None to signal cash out
        """
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Check stop-loss and win-target
        loss_threshold = self.starting_bankroll * self.stop_loss
        win_threshold = self.starting_bankroll * self.win_target
        
        if abs(self.current_loss) >= loss_threshold:
            return None  # Cash out - stop loss hit
        
        if self.current_profit >= win_threshold:
            return None  # Cash out - win target hit
        
        # Evaluate board state risk
        if self._is_board_too_risky(state, valid_actions):
            return None  # Cash out - board is too dangerous
        
        # Find safest cell with positive EV
        safest_cell = None
        lowest_risk = float('inf')
        
        for cell in valid_actions:
            risk = self._calculate_probability_of_loss(state, cell, valid_actions)
            ev = self._calculate_ev(state, cell, valid_actions)
            
            # Only consider cells below MTR and with positive EV
            if risk < self.max_tolerable_risk and ev > 0:
                if risk < lowest_risk:
                    lowest_risk = risk
                    safest_cell = cell
        
        # If no safe cell found, cash out
        if safest_cell is None:
            return None
        
        return safest_cell
    
    def _calculate_probability_of_loss(self, state: Dict, cell: Tuple[int, int], valid_actions: List) -> float:
        """Calculate the exact probability of hitting a mine in this cell."""
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 3)
        clicks_made = state.get('clicks_made', 0)
        
        total_cells = board_size * board_size
        unrevealed_count = len(valid_actions)
        
        if unrevealed_count == 0:
            return 1.0
        
        # Basic probability: mines / unrevealed cells
        probability_of_mine = mine_count / unrevealed_count
        
        return probability_of_mine
    
    def _calculate_ev(self, state: Dict, cell: Tuple[int, int], valid_actions: List) -> float:
        """Calculate expected value of clicking this cell."""
        p_loss = self._calculate_probability_of_loss(state, cell, valid_actions)
        p_safe = 1.0 - p_loss
        
        # Simplified payout calculation
        clicks_made = state.get('clicks_made', 0)
        payout = 1.0 + (clicks_made * 0.1)  # Increasing multiplier
        loss = 1.0
        
        ev = (p_safe * payout) - (p_loss * loss)
        return ev
    
    def _is_board_too_risky(self, state: Dict, valid_actions: List) -> bool:
        """
        Assess if the overall board state is too risky to continue.
        
        Returns True if should cash out.
        """
        if not valid_actions:
            return True
        
        # Calculate average risk across all remaining cells
        total_risk = 0.0
        for cell in valid_actions:
            risk = self._calculate_probability_of_loss(state, cell, valid_actions)
            total_risk += risk
        
        avg_risk = total_risk / len(valid_actions)
        
        # If average risk is high, board is too dangerous
        return avg_risk > (self.max_tolerable_risk * 1.5)
    
    def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """Update bankroll tracking."""
        super().update(state, action, reward, next_state, done)
        
        # Track profit/loss
        if reward > 0:
            self.current_profit += reward
            self.current_bankroll += reward
        else:
            self.current_loss += abs(reward)
            self.current_bankroll -= abs(reward)
    
    def reset(self) -> None:
        """Reset for new game."""
        super().reset()
        self.current_profit = 0.0
        self.current_loss = 0.0
        # Don't reset bankroll - it persists across games


