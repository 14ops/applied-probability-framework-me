"""
Takeshi Kovacs - The Aggressive Berserker Strategy

Core Philosophy: "Live fast, die young, leave a beautiful corpse... or a mountain of profit."
High-octane, aggressive assault designed for maximum impact and rapid exploitation of advantages.
"""

import random
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from core.base_strategy import BaseStrategy


class TakeshiKovacsStrategy(BaseStrategy):
    """
    The Aggressive Berserker - Dynamic aggression with rapid adaptation.
    
    Key Features:
    - Dynamic bet sizing based on perceived advantage
    - Rapid exploitation of favorable conditions
    - Strict stop-loss and win-target mechanisms
    - Highest EV cell prioritization
    """
    
    register = True
    plugin_name = "takeshi"
    
    def __init__(self, name: str = "Takeshi Kovacs - Aggressive Berserker", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
        # Configuration parameters
        self.base_bet = config.get('base_bet', 10.0) if config else 10.0
        self.aggression_factor = config.get('aggression_factor', 0.5) if config else 0.5
        self.aggression_threshold = config.get('aggression_threshold', 0.5) if config else 0.5
        self.win_target = config.get('win_target', 500.0) if config else 500.0
        self.stop_loss = config.get('stop_loss', 100.0) if config else 100.0
        
        # Tracking
        self.current_profit = 0.0
        self.current_loss = 0.0
        
    def select_action(self, state: Any, valid_actions: Any) -> Any:
        """
        Select action based on perceived advantage and highest EV.
        
        Args:
            state: Current game state with board info
            valid_actions: List of (row, col) tuples for valid cells
            
        Returns:
            (row, col) tuple for the selected action
        """
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Calculate perceived advantage
        perceived_advantage = self._calculate_perceived_advantage(state)
        
        # Check if we should cash out based on thresholds
        if self.current_profit >= self.win_target or abs(self.current_loss) >= self.stop_loss:
            # Return None to signal cash out (will be handled by game logic)
            return None
        
        # If perceived advantage is high, be aggressive
        if perceived_advantage > self.aggression_threshold:
            # Find highest EV cell
            return self._select_highest_ev_cell(state, valid_actions)
        else:
            # Still select highest EV but might cash out if risk is too high
            best_cell = self._select_highest_ev_cell(state, valid_actions)
            cell_risk = self._calculate_cell_risk(state, best_cell)
            
            if cell_risk > 0.7:  # High risk threshold
                return None  # Signal cash out
            return best_cell
    
    def _calculate_perceived_advantage(self, state: Dict) -> float:
        """
        Calculate the perceived advantage based on multiple factors.
        
        Components:
        - Mine density in unrevealed cells
        - Payout multiplier potential
        - Board edge proximity
        - Recent game state volatility
        """
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 3)
        clicks_made = state.get('clicks_made', 0)
        revealed = state.get('revealed', [])
        
        # Count unrevealed cells
        total_cells = board_size * board_size
        revealed_count = sum(sum(row) for row in revealed) if isinstance(revealed, list) else clicks_made
        unrevealed_count = total_cells - revealed_count
        
        if unrevealed_count == 0:
            return 0.0
        
        # Mine density (lower is better)
        mine_density = mine_count / max(unrevealed_count, 1)
        mine_density_factor = 1.0 - mine_density
        
        # Multiplier potential (more clicks = higher multiplier potential)
        multiplier_potential = min(clicks_made / (total_cells * 0.5), 1.0)
        
        # Edge proximity factor (simplified - would need actual cell positions)
        edge_proximity_factor = 0.5
        
        # Recent volatility (simplified - would track actual game history)
        recent_volatility = 0.5
        
        # Weighted sum
        w1, w2, w3, w4 = 0.4, 0.3, 0.2, 0.1
        pa = (w1 * mine_density_factor + 
              w2 * multiplier_potential + 
              w3 * edge_proximity_factor + 
              w4 * recent_volatility)
        
        return pa
    
    def _select_highest_ev_cell(self, state: Dict, valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Select the cell with the highest expected value."""
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 3)
        revealed = state.get('revealed', [])
        
        # Calculate EV for each cell
        best_cell = None
        best_ev = float('-inf')
        
        for cell in valid_actions:
            row, col = cell
            
            # Calculate probability of safety
            unrevealed_count = sum(1 for r, c in valid_actions)
            p_safe = 1.0 - (mine_count / max(unrevealed_count, 1))
            
            # Simple EV calculation (would be more sophisticated in practice)
            payout = 1.5  # Simplified multiplier
            loss = 1.0
            ev = (p_safe * payout) - ((1 - p_safe) * loss)
            
            if ev > best_ev:
                best_ev = ev
                best_cell = cell
        
        return best_cell if best_cell else valid_actions[0]
    
    def _calculate_cell_risk(self, state: Dict, cell: Tuple[int, int]) -> float:
        """Calculate the risk level of clicking a specific cell."""
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 3)
        revealed = state.get('revealed', [])
        
        # Count unrevealed cells
        total_cells = board_size * board_size
        clicks_made = state.get('clicks_made', 0)
        unrevealed_count = total_cells - clicks_made
        
        if unrevealed_count == 0:
            return 1.0
        
        # Basic risk calculation
        risk = mine_count / unrevealed_count
        return risk
    
    def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """Update strategy based on outcome."""
        super().update(state, action, reward, next_state, done)
        
        # Track profit/loss
        if reward > 0:
            self.current_profit += reward
        else:
            self.current_loss += abs(reward)
    
    def reset(self) -> None:
        """Reset strategy state for a new game."""
        super().reset()
        self.current_profit = 0.0
        self.current_loss = 0.0


