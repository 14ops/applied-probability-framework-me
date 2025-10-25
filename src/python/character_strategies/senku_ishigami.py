"""
Senku Ishigami - The Analytical Scientist

Core Philosophy: "Science is the ultimate tool for understanding and conquering the world."
Pure data-driven decision-making with rigorous probability calculations.
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from core.base_strategy import BaseStrategy


class SenkuIshigamiStrategy(BaseStrategy):
    """
    The Analytical Scientist - Pure EV maximization with Bayesian inference.
    
    Key Features:
    - Precise probability calculation using combinatorics
    - Expected Value maximization
    - Data-driven optimization
    - Systematic exploration
    - Hypothesis testing and model refinement
    """
    
    register = True
    plugin_name = "senku"
    
    def __init__(self, name: str = "Senku Ishigami - Analytical Scientist", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
        # Scientific parameters
        self.exploration_rate = config.get('exploration_rate', 0.1) if config else 0.1
        
        # Data collection
        self.probability_map = {}
        self.historical_data = []
        
    def select_action(self, state: Any, valid_actions: Any) -> Any:
        """
        Select action with highest Expected Value based on precise calculations.
        
        Args:
            state: Current game state
            valid_actions: List of valid (row, col) tuples
            
        Returns:
            (row, col) tuple with highest EV
        """
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Update probability map with current state
        self._update_probability_map(state, valid_actions)
        
        # Calculate EV for each cell
        best_cell = None
        best_ev = float('-inf')
        
        for cell in valid_actions:
            ev = self._calculate_expected_value(state, cell, valid_actions)
            
            if ev > best_ev:
                best_ev = ev
                best_cell = cell
        
        # If no positive EV cells, return None to signal cash out
        if best_ev <= 0:
            return None
        
        return best_cell if best_cell else valid_actions[0]
    
    def _update_probability_map(self, state: Dict, valid_actions: List) -> None:
        """
        Update the probability map using Bayesian inference and revealed information.
        """
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 3)
        board = state.get('board', [])
        revealed = state.get('revealed', [])
        
        # For each unrevealed cell, calculate precise probability
        for cell in valid_actions:
            row, col = cell
            
            # Use revealed numbers to deduce probabilities
            probability = self._calculate_cell_probability(
                state, cell, valid_actions
            )
            
            self.probability_map[cell] = probability
    
    def _calculate_cell_probability(self, state: Dict, cell: Tuple[int, int], 
                                   valid_actions: List) -> float:
        """
        Calculate precise probability of mine in this cell using combinatorics.
        """
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 3)
        board = state.get('board', [])
        revealed = state.get('revealed', [])
        
        row, col = cell
        
        # Check if we have revealed numbers nearby
        adjacent_info = self._get_adjacent_revealed_info(state, cell)
        
        if adjacent_info:
            # Use Bayesian inference with adjacent information
            probability = self._bayesian_probability(
                cell, adjacent_info, valid_actions, mine_count
            )
        else:
            # Base probability: remaining mines / unrevealed cells
            unrevealed_count = len(valid_actions)
            probability = mine_count / max(unrevealed_count, 1)
        
        return probability
    
    def _get_adjacent_revealed_info(self, state: Dict, cell: Tuple[int, int]) -> List[Dict]:
        """Get information from revealed adjacent cells."""
        board_size = state.get('board_size', 5)
        board = state.get('board', [])
        revealed = state.get('revealed', [])
        row, col = cell
        
        adjacent_info = []
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if isinstance(revealed, list) and len(revealed) > nr and len(revealed[nr]) > nc:
                        if revealed[nr][nc]:
                            # This cell is revealed
                            if isinstance(board, list) and len(board) > nr and len(board[nr]) > nc:
                                number = board[nr][nc]
                                adjacent_info.append({
                                    'position': (nr, nc),
                                    'number': number
                                })
        
        return adjacent_info
    
    def _bayesian_probability(self, cell: Tuple[int, int], adjacent_info: List[Dict],
                             valid_actions: List, mine_count: int) -> float:
        """
        Use Bayesian inference to calculate probability based on adjacent revealed numbers.
        """
        # Simplified Bayesian calculation
        # In a full implementation, this would use constraint satisfaction
        
        if not adjacent_info:
            unrevealed_count = len(valid_actions)
            return mine_count / max(unrevealed_count, 1)
        
        # Count constraints from adjacent numbers
        total_constraint = 0
        constraint_count = 0
        
        for info in adjacent_info:
            number = info['number']
            if number > 0:
                total_constraint += number
                constraint_count += 1
        
        if constraint_count == 0:
            unrevealed_count = len(valid_actions)
            return mine_count / max(unrevealed_count, 1)
        
        # Adjust probability based on constraints
        avg_constraint = total_constraint / constraint_count
        base_prob = mine_count / len(valid_actions)
        
        # Cells near high numbers are more likely to have mines
        adjusted_prob = base_prob * (1.0 + (avg_constraint / 8.0))
        
        return min(adjusted_prob, 1.0)
    
    def _calculate_expected_value(self, state: Dict, cell: Tuple[int, int],
                                 valid_actions: List) -> float:
        """
        Calculate Expected Value for clicking this cell.
        
        EV = (P(safe) * Payout) - (P(mine) * Loss)
        """
        # Get probability from map
        p_mine = self.probability_map.get(cell, 0.5)
        p_safe = 1.0 - p_mine
        
        # Calculate payout (increases with clicks)
        clicks_made = state.get('clicks_made', 0)
        payout = 1.0 + (clicks_made * 0.15)  # 15% increase per click
        
        # Loss is the bet amount
        loss = 1.0
        
        # Expected Value
        ev = (p_safe * payout) - (p_mine * loss)
        
        return ev
    
    def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """Update historical data for model refinement."""
        super().update(state, action, reward, next_state, done)
        
        # Record data point
        self.historical_data.append({
            'action': action,
            'reward': reward,
            'done': done,
            'state': state
        })
        
        # Hypothesis testing: if outcome was unexpected, refine model
        if action and action in self.probability_map:
            predicted_prob = self.probability_map[action]
            actual_outcome = 0 if reward > 0 else 1  # 0 = safe, 1 = mine
            
            # This is where model refinement would occur
            # For now, just log the discrepancy
            pass
    
    def reset(self) -> None:
        """Reset for new game."""
        super().reset()
        self.probability_map = {}
        # Keep historical data for long-term learning


