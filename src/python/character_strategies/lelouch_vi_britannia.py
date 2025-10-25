"""
Lelouch vi Britannia - The Strategic Mastermind

Core Philosophy: "The only ones who should kill are those who are prepared to be killed."
Grand strategic campaign with foresight, planning, and system manipulation.
"""

import random
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from core.base_strategy import BaseStrategy


class LelouchViBritanniaStrategy(BaseStrategy):
    """
    The Strategic Mastermind - Trinity-Force evaluation with long-term planning.
    
    Key Features:
    - Expected Value (EV) engine
    - Strategic Impact (SI) engine - information gain, board control, pathing
    - Psychological Advantage (PA) engine - pattern recognition, system exploitation
    - State-space planning and contingency analysis
    """
    
    register = True
    plugin_name = "lelouch"
    
    def __init__(self, name: str = "Lelouch vi Britannia - Strategic Mastermind", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
        # Trinity-Force weights
        self.w_ev = config.get('w_ev', 0.4) if config else 0.4
        self.w_si = config.get('w_si', 0.4) if config else 0.4
        self.w_pa = config.get('w_pa', 0.2) if config else 0.2
        
        # Strategic parameters
        self.lookahead_depth = config.get('lookahead_depth', 2) if config else 2
        self.simulation_count = config.get('simulation_count', 100) if config else 100
        
        # Geass Modeler - pattern recognition
        self.pattern_history = []
        self.detected_patterns = {}
        
    def select_action(self, state: Any, valid_actions: Any) -> Any:
        """
        Select action using Trinity-Force evaluation.
        
        Args:
            state: Current game state
            valid_actions: List of valid (row, col) tuples
            
        Returns:
            (row, col) tuple for selected action
        """
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        best_cell = None
        best_score = float('-inf')
        
        for cell in valid_actions:
            # Calculate Trinity-Force score
            ev_score = self._calculate_ev(state, cell)
            si_score = self._calculate_strategic_impact(state, cell, valid_actions)
            pa_score = self._calculate_psychological_advantage(state, cell)
            
            # Weighted combination
            total_score = (self.w_ev * ev_score + 
                          self.w_si * si_score + 
                          self.w_pa * pa_score)
            
            # Contingency planning - check if this leads to traps
            if self._contingency_check(state, cell):
                total_score *= 0.5  # Penalize risky moves
            
            if total_score > best_score:
                best_score = total_score
                best_cell = cell
        
        return best_cell if best_cell else valid_actions[0]
    
    def _calculate_ev(self, state: Dict, cell: Tuple[int, int]) -> float:
        """Engine I: Expected Value calculation."""
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 3)
        clicks_made = state.get('clicks_made', 0)
        
        total_cells = board_size * board_size
        unrevealed_count = total_cells - clicks_made
        
        if unrevealed_count == 0:
            return 0.0
        
        # Probability of win (cell is safe)
        p_win = 1.0 - (mine_count / unrevealed_count)
        
        # Potential gain (simplified multiplier)
        potential_gain = 1.5 + (clicks_made * 0.1)
        
        # Potential loss
        potential_loss = 1.0
        
        # EV = (P_win * Gain) - (P_loss * Loss)
        ev = (p_win * potential_gain) - ((1 - p_win) * potential_loss)
        
        return ev
    
    def _calculate_strategic_impact(self, state: Dict, cell: Tuple[int, int], valid_actions: List) -> float:
        """
        Engine II: Strategic Impact calculation.
        
        Components:
        - Information Gain: How much uncertainty does this reduce?
        - Board Control: Is this a central, influential position?
        - Pathing Potential: Does this open up future opportunities?
        """
        board_size = state.get('board_size', 5)
        row, col = cell
        
        # Information Gain: cells with more unrevealed neighbors provide more info
        info_gain = self._count_unrevealed_neighbors(state, cell, valid_actions)
        info_gain_value = info_gain / 8.0  # Normalize (max 8 neighbors)
        
        # Board Control: center cells are more valuable
        center_row, center_col = board_size // 2, board_size // 2
        distance_from_center = abs(row - center_row) + abs(col - center_col)
        board_control_value = 1.0 - (distance_from_center / (board_size * 2))
        
        # Pathing Potential: does this connect to other safe areas?
        pathing_value = self._evaluate_pathing(state, cell, valid_actions)
        
        # Weighted combination
        w1, w2, w3 = 0.4, 0.3, 0.3
        si_score = (w1 * info_gain_value + 
                   w2 * board_control_value + 
                   w3 * pathing_value)
        
        return si_score
    
    def _calculate_psychological_advantage(self, state: Dict, cell: Tuple[int, int]) -> float:
        """
        Engine III: Psychological Advantage (Geass Modeler).
        
        Detects patterns and exploits system biases.
        """
        board_size = state.get('board_size', 5)
        row, col = cell
        
        # Check for detected patterns
        pa_score = 0.5  # Baseline
        
        # Center cell bias detection (example pattern)
        center_row, center_col = board_size // 2, board_size // 2
        if row == center_row and col == center_col:
            # Geass Modeler might have detected center is safer
            pa_score += 0.2
        
        # Edge avoidance pattern
        if row == 0 or row == board_size - 1 or col == 0 or col == board_size - 1:
            pa_score -= 0.1
        
        # Human-like randomness factor
        pa_score += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, pa_score))
    
    def _count_unrevealed_neighbors(self, state: Dict, cell: Tuple[int, int], valid_actions: List) -> int:
        """Count how many unrevealed neighbors a cell has."""
        row, col = cell
        board_size = state.get('board_size', 5)
        
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if (nr, nc) in valid_actions:
                        count += 1
        
        return count
    
    def _evaluate_pathing(self, state: Dict, cell: Tuple[int, int], valid_actions: List) -> float:
        """Evaluate if this cell creates good paths for future clicks."""
        # Simplified: cells with many valid neighbors have better pathing
        neighbor_count = self._count_unrevealed_neighbors(state, cell, valid_actions)
        return neighbor_count / 8.0
    
    def _contingency_check(self, state: Dict, cell: Tuple[int, int]) -> bool:
        """
        Contingency Planning: Check if this move leads to forced loss states.
        
        Returns True if move is risky (should be penalized).
        """
        # Simplified: check if this is a high-risk cell
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 3)
        clicks_made = state.get('clicks_made', 0)
        
        total_cells = board_size * board_size
        unrevealed_count = total_cells - clicks_made
        
        if unrevealed_count == 0:
            return False
        
        risk = mine_count / unrevealed_count
        
        # If risk is very high, flag as risky
        return risk > 0.5
    
    def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """Update Geass Modeler with new patterns."""
        super().update(state, action, reward, next_state, done)
        
        # Record pattern for future analysis
        if action:
            self.pattern_history.append({
                'action': action,
                'reward': reward,
                'done': done
            })
    
    def reset(self) -> None:
        """Reset for new game."""
        super().reset()
        # Keep pattern history for learning across games


