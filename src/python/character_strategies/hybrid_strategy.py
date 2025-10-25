"""
Hybrid Strategy - The Ultimate Fusion

Core Philosophy: "The strongest strategy is one that adapts and evolves."
Dynamic blending of Senku (analytical) and Lelouch (strategic) based on game conditions.
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from core.base_strategy import BaseStrategy
from .senku_ishigami import SenkuIshigamiStrategy
from .lelouch_vi_britannia import LelouchViBritanniaStrategy


class HybridStrategy(BaseStrategy):
    """
    The Ultimate Fusion - Dynamic blending of Senku and Lelouch.
    
    Key Features:
    - Game phase detection (Early, Mid, Late)
    - Board complexity assessment
    - Performance-based weight adjustment
    - Seamless strategy blending via sigmoid function
    """
    
    register = True
    plugin_name = "hybrid"
    
    def __init__(self, name: str = "Hybrid Strategy - Ultimate Fusion", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
        # Initialize sub-strategies
        self.senku = SenkuIshigamiStrategy(config=config)
        self.lelouch = LelouchViBritanniaStrategy(config=config)
        
        # Blending parameters
        self.strategic_threshold = config.get('strategic_threshold', 0.5) if config else 0.5
        
        # Performance tracking
        self.recent_performance = []
        self.max_history = 10
        
    def select_action(self, state: Any, valid_actions: Any) -> Any:
        """
        Select action by dynamically blending Senku and Lelouch strategies.
        
        Args:
            state: Current game state
            valid_actions: List of valid (row, col) tuples
            
        Returns:
            (row, col) tuple for selected action
        """
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Determine current game phase
        game_phase = self._determine_game_phase(state)
        
        # Calculate board complexity
        board_complexity = self._calculate_board_complexity(state, valid_actions)
        
        # Calculate performance metrics
        performance_score = self._calculate_performance_score()
        
        # Calculate strategic score for weight determination
        strategic_score = self._calculate_strategic_score(
            game_phase, board_complexity, performance_score
        )
        
        # Calculate weights using sigmoid function
        weight_lelouch = self._sigmoid(strategic_score - self.strategic_threshold)
        weight_senku = 1.0 - weight_lelouch
        
        # Get decisions from both strategies
        senku_action = self.senku.select_action(state, valid_actions)
        lelouch_action = self.lelouch.select_action(state, valid_actions)
        
        # Handle cash-out signals (None)
        if senku_action is None and lelouch_action is None:
            return None
        elif senku_action is None:
            return lelouch_action
        elif lelouch_action is None:
            return senku_action
        
        # Blend decisions based on weights
        if weight_lelouch > 0.7:
            # Strongly favor Lelouch
            return lelouch_action
        elif weight_senku > 0.7:
            # Strongly favor Senku
            return senku_action
        else:
            # Mixed regime - use weighted random selection
            if np.random.random() < weight_lelouch:
                return lelouch_action
            else:
                return senku_action
    
    def _determine_game_phase(self, state: Dict) -> str:
        """
        Determine current game phase: Early, Mid, or Late.
        """
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 3)
        clicks_made = state.get('clicks_made', 0)
        
        total_cells = board_size * board_size
        safe_cells = total_cells - mine_count
        
        progress = clicks_made / safe_cells if safe_cells > 0 else 0
        
        if progress < 0.3:
            return 'Early'
        elif progress < 0.7:
            return 'Mid'
        else:
            return 'Late'
    
    def _calculate_board_complexity(self, state: Dict, valid_actions: List) -> float:
        """
        Calculate board complexity (0.0 = simple, 1.0 = complex).
        
        Factors:
        - Number of unrevealed cells
        - Mine density
        - Ambiguity of revealed information
        """
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 3)
        clicks_made = state.get('clicks_made', 0)
        
        total_cells = board_size * board_size
        unrevealed_count = len(valid_actions)
        
        # Complexity increases with unrevealed cells
        unrevealed_factor = unrevealed_count / total_cells
        
        # Complexity increases with mine density
        mine_density = mine_count / max(unrevealed_count, 1)
        
        # Weighted combination
        complexity = (0.5 * unrevealed_factor) + (0.5 * mine_density)
        
        return min(complexity, 1.0)
    
    def _calculate_performance_score(self) -> float:
        """
        Calculate recent performance score.
        
        Returns value between 0.0 (poor) and 1.0 (excellent).
        """
        if not self.recent_performance:
            return 0.5  # Neutral
        
        # Calculate win rate from recent games
        wins = sum(1 for outcome in self.recent_performance if outcome > 0)
        total = len(self.recent_performance)
        
        win_rate = wins / total if total > 0 else 0.5
        
        return win_rate
    
    def _calculate_strategic_score(self, game_phase: str, board_complexity: float,
                                   performance_score: float) -> float:
        """
        Calculate composite strategic score for weight determination.
        
        Higher score = more Lelouch influence
        Lower score = more Senku influence
        """
        # Phase contribution
        phase_scores = {'Early': 0.2, 'Mid': 0.5, 'Late': 0.8}
        phase_score = phase_scores.get(game_phase, 0.5)
        
        # Complexity contribution (high complexity favors Lelouch)
        complexity_score = board_complexity
        
        # Performance contribution (poor performance might trigger Lelouch's strategic thinking)
        performance_contribution = 1.0 - performance_score  # Invert: poor performance = higher score
        
        # Weighted combination
        w_phase, w_complexity, w_performance = 0.4, 0.4, 0.2
        strategic_score = (w_phase * phase_score + 
                          w_complexity * complexity_score + 
                          w_performance * performance_contribution)
        
        return strategic_score
    
    def _sigmoid(self, x: float) -> float:
        """
        Sigmoid function for smooth weight transition.
        
        Maps (-inf, +inf) to (0, 1)
        """
        return 1.0 / (1.0 + np.exp(-10 * x))  # Steeper sigmoid (factor of 10)
    
    def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """Update both sub-strategies and performance tracking."""
        super().update(state, action, reward, next_state, done)
        
        # Update sub-strategies
        self.senku.update(state, action, reward, next_state, done)
        self.lelouch.update(state, action, reward, next_state, done)
        
        # Track performance
        if done:
            self.recent_performance.append(reward)
            
            # Keep only recent history
            if len(self.recent_performance) > self.max_history:
                self.recent_performance.pop(0)
    
    def reset(self) -> None:
        """Reset both sub-strategies."""
        super().reset()
        self.senku.reset()
        self.lelouch.reset()


