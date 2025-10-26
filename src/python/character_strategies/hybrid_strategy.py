"""
Hybrid Strategy - The Ultimate Fusion

Core Philosophy: "The strongest strategy is one that adapts and evolves."
Dynamic blending of Senku (analytical) and Lelouch (strategic) based on game conditions.
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from core.adaptive_strategy import AdaptiveLearningStrategy
from .senku_ishigami import SenkuIshigamiStrategy
from .lelouch_vi_britannia import LelouchViBritanniaStrategy


class HybridStrategy(AdaptiveLearningStrategy):
    """
    The Ultimate Fusion - Dynamic blending of Senku and Lelouch with AI Evolution.
    
    Key Features:
    - Game phase detection (Early, Mid, Late)
    - Board complexity assessment
    - Performance-based weight adjustment
    - Seamless strategy blending via sigmoid function
    - Q-Learning Matrix for action value estimation
    - Experience Replay for efficient learning
    - Parameter Evolution via Genetic Algorithms
    """
    
    register = True
    plugin_name = "hybrid"
    
    def __init__(self, name: str = "Hybrid Strategy - Ultimate Fusion", config: Optional[Dict[str, Any]] = None):
        # Define evolvable parameters
        parameter_ranges = {
            'strategic_threshold': (0.3, 0.7),
            'w_phase': (0.2, 0.6),
            'w_complexity': (0.2, 0.6),
            'w_performance': (0.1, 0.4),
        }
        
        # Initialize adaptive learning with evolution
        super().__init__(
            name=name,
            config=config,
            use_q_learning=True,
            use_experience_replay=True,
            use_parameter_evolution=True,
            parameter_ranges=parameter_ranges
        )
        
        # Initialize sub-strategies
        self.senku = SenkuIshigamiStrategy(config=config)
        self.lelouch = LelouchViBritanniaStrategy(config=config)
        
        # Blending parameters (can be evolved)
        self.strategic_threshold = config.get('strategic_threshold', 0.5) if config else 0.5
        self.w_phase = config.get('w_phase', 0.4) if config else 0.4
        self.w_complexity = config.get('w_complexity', 0.4) if config else 0.4
        self.w_performance = config.get('w_performance', 0.2) if config else 0.2
        
        # Performance tracking
        self.recent_performance = []
        self.max_history = 10
        
        # Q-learning weight (balance between learned and heuristic)
        self.q_learning_weight = config.get('q_learning_weight', 0.3) if config else 0.3
        
    def select_action_heuristic(self, state: Any, valid_actions: Any) -> Any:
        """
        Heuristic action selection (override of AdaptiveLearningStrategy).
        Dynamically blends Senku and Lelouch strategies.
        
        Args:
            state: Current game state
            valid_actions: List of valid (row, col) tuples
            
        Returns:
            (row, col) tuple for selected action
        """
        if not valid_actions:
            return None
        
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
        
        # Weighted combination (using evolved weights)
        strategic_score = (self.w_phase * phase_score + 
                          self.w_complexity * complexity_score + 
                          self.w_performance * performance_contribution)
        
        return strategic_score
    
    def _sigmoid(self, x: float) -> float:
        """
        Sigmoid function for smooth weight transition.
        
        Maps (-inf, +inf) to (0, 1)
        """
        return 1.0 / (1.0 + np.exp(-10 * x))  # Steeper sigmoid (factor of 10)
    
    def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """Update learning matrices, sub-strategies, and performance tracking."""
        # Call adaptive strategy update (handles Q-learning, replay, evolution)
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
    
    def _apply_evolved_parameters(self, params: Dict[str, float]) -> None:
        """Apply evolved parameters to strategy."""
        super()._apply_evolved_parameters(params)
        
        # Update strategy-specific parameters
        self.strategic_threshold = params.get('strategic_threshold', self.strategic_threshold)
        self.w_phase = params.get('w_phase', self.w_phase)
        self.w_complexity = params.get('w_complexity', self.w_complexity)
        self.w_performance = params.get('w_performance', self.w_performance)
    
    def reset(self) -> None:
        """Reset both sub-strategies."""
        super().reset()
        self.senku.reset()
        self.lelouch.reset()



