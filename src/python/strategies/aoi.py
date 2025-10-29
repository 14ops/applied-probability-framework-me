"""
Aoi Strategy - Analytical Optimizer

This strategy implements a data-driven, analytical approach with careful
probability calculations and optimal decision-making.

Key characteristics:
- High analytical focus (0.9+)
- Moderate risk tolerance (0.4-0.6)
- Cash-out threshold: 2.12x minimum (as specified)
- Uses mathematical optimization
- Calculated decision-making with extensive analysis
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from ..game.math import win_probability, expected_value, payout_table_25_2, kelly_criterion_fraction, optimal_clicks


@dataclass
class AoiConfig:
    """Configuration for Aoi strategy."""
    analytical_focus: float = 0.95
    risk_tolerance: float = 0.5
    cash_out_threshold: float = 2.12  # Minimum 2.12x as specified
    max_clicks: int = 18
    learning_rate: float = 0.05
    confidence_threshold: float = 0.8
    optimization_weight: float = 0.7
    probability_threshold: float = 0.3


class AoiStrategy:
    """
    Aoi - Analytical Optimizer Strategy
    
    Implements a data-driven approach with careful probability calculations,
    mathematical optimization, and a minimum 2.12x cash-out threshold.
    """
    
    def __init__(self, config: Optional[AoiConfig] = None):
        self.config = config or AoiConfig()
        self.name = "Aoi"
        self.description = "Analytical optimizer with mathematical precision"
        
        # Strategy state
        self.current_analytical_focus = self.config.analytical_focus
        self.current_risk_tolerance = self.config.risk_tolerance
        self.clicks_made = 0
        self.current_payout = 1.0
        self.game_history = []
        
        # Performance tracking
        self.total_games = 0
        self.total_wins = 0
        self.total_reward = 0.0
        self.max_reward = 0.0
        
        # Analytical data
        self.probability_cache = {}
        self.expected_value_cache = {}
        self.optimization_history = []
        
        # Payout table
        self.payout_table = payout_table_25_2()
    
    def select_action(self, state: Dict[str, Any], valid_actions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Select action using Aoi's analytical approach.
        
        Args:
            state: Current game state
            valid_actions: List of valid (row, col) tuples
            
        Returns:
            Selected action or None to cash out
        """
        if not valid_actions:
            return None
        
        # Update strategy state
        self.clicks_made = state.get('clicks_made', 0)
        self.current_payout = state.get('current_payout', 1.0)
        
        # Check if we should cash out
        if self._should_cash_out(state):
            return None
        
        # Calculate optimal action using analytical methods
        action = self._select_analytical_action(state, valid_actions)
        
        return action
    
    def _should_cash_out(self, state: Dict[str, Any]) -> bool:
        """
        Determine if we should cash out based on Aoi's analytical criteria.
        
        Args:
            state: Current game state
            
        Returns:
            True if we should cash out
        """
        current_payout = state.get('current_payout', 1.0)
        clicks_made = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 2)
        
        # Always cash out if we've reached the minimum threshold
        if current_payout >= self.config.cash_out_threshold:
            return True
        
        # Cash out if we've made too many clicks
        if clicks_made >= self.config.max_clicks:
            return True
        
        # Calculate remaining tiles and mines
        total_tiles = board_size * board_size
        remaining_tiles = total_tiles - clicks_made
        remaining_mines = mine_count
        
        if remaining_tiles <= remaining_mines:
            return True
        
        # Analytical decision: calculate expected value for continuing
        expected_val_continue = self._calculate_expected_value_continue(state)
        expected_val_cash_out = current_payout
        
        # Cash out if expected value of continuing is lower
        if expected_val_continue < expected_val_cash_out:
            return True
        
        # Additional analytical check: probability threshold
        win_prob = self._calculate_win_probability(state)
        if win_prob < self.config.probability_threshold:
            return True
        
        return False
    
    def _select_analytical_action(self, state: Dict[str, Any], valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select action using analytical optimization.
        
        Args:
            state: Current game state
            valid_actions: List of valid actions
            
        Returns:
            Selected action
        """
        # Calculate analytical scores for all actions
        action_scores = []
        
        for action in valid_actions:
            score = self._calculate_analytical_score(action, state)
            action_scores.append((action, score))
        
        # Sort by score (highest first)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply analytical focus: more likely to choose optimal actions
        if random.random() < self.current_analytical_focus:
            # Choose from top 20% of actions (most optimal)
            top_actions = action_scores[:max(1, len(action_scores) // 5)]
            selected_action = random.choice(top_actions)[0]
        else:
            # Choose from top 40% of actions
            top_actions = action_scores[:max(1, len(action_scores) // 2.5)]
            selected_action = random.choice(top_actions)[0]
        
        return selected_action
    
    def _calculate_analytical_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate analytical score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Analytical score (higher is better)
        """
        # Expected value score
        ev_score = self._calculate_expected_value_score(action, state)
        
        # Probability score
        prob_score = self._calculate_probability_score(action, state)
        
        # Optimization score
        opt_score = self._calculate_optimization_score(action, state)
        
        # Risk-adjusted score
        risk_score = self._calculate_risk_adjusted_score(action, state)
        
        # Combine scores with weights
        total_score = (
            ev_score * 0.4 +
            prob_score * 0.3 +
            opt_score * 0.2 +
            risk_score * 0.1
        )
        
        return total_score
    
    def _calculate_expected_value_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate expected value score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Expected value score
        """
        clicks_made = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 2)
        
        # Calculate expected value for continuing
        remaining_tiles = board_size * board_size - clicks_made
        remaining_mines = mine_count
        
        if remaining_tiles <= remaining_mines:
            return 0.0
        
        # Calculate expected value for next click
        next_clicks = clicks_made + 1
        win_prob = win_probability(next_clicks, board_size * board_size, mine_count)
        payout = self.payout_table.get(next_clicks, 1.0)
        
        expected_val = win_prob * payout - (1 - win_prob) * 1.0
        
        # Normalize to 0-1
        return max(0.0, min(1.0, expected_val / 5.0))
    
    def _calculate_probability_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate probability score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Probability score
        """
        clicks_made = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 2)
        
        # Calculate win probability for next click
        next_clicks = clicks_made + 1
        win_prob = win_probability(next_clicks, board_size * board_size, mine_count)
        
        return win_prob
    
    def _calculate_optimization_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate optimization score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Optimization score
        """
        clicks_made = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 2)
        
        # Calculate optimal clicks for current configuration
        optimal_k, max_ev = optimal_clicks(board_size * board_size, mine_count, self.payout_table)
        
        # Score based on how close we are to optimal
        distance_from_optimal = abs(clicks_made + 1 - optimal_k)
        max_distance = board_size * board_size - mine_count
        
        # Normalize to 0-1 (closer to optimal = higher score)
        opt_score = 1.0 - (distance_from_optimal / max_distance)
        
        return max(0.0, min(1.0, opt_score))
    
    def _calculate_risk_adjusted_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate risk-adjusted score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Risk-adjusted score
        """
        clicks_made = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 2)
        
        # Calculate Kelly Criterion fraction
        next_clicks = clicks_made + 1
        kelly_fraction = kelly_criterion_fraction(next_clicks, board_size * board_size, mine_count, self.payout_table)
        
        # Risk-adjusted score based on Kelly Criterion
        risk_score = kelly_fraction * self.current_risk_tolerance
        
        return risk_score
    
    def _calculate_expected_value_continue(self, state: Dict[str, Any]) -> float:
        """
        Calculate expected value of continuing.
        
        Args:
            state: Current game state
            
        Returns:
            Expected value of continuing
        """
        clicks_made = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 2)
        
        # Calculate expected value for next click
        next_clicks = clicks_made + 1
        return expected_value(next_clicks, board_size * board_size, mine_count, self.payout_table)
    
    def _calculate_win_probability(self, state: Dict[str, Any]) -> float:
        """
        Calculate win probability for current state.
        
        Args:
            state: Current game state
            
        Returns:
            Win probability
        """
        clicks_made = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 2)
        
        return win_probability(clicks_made, board_size * board_size, mine_count)
    
    def update(self, state: Dict[str, Any], action: Optional[Tuple[int, int]], 
               reward: float, next_state: Dict[str, Any], done: bool) -> None:
        """
        Update strategy based on game outcome.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether game is done
        """
        # Update game history
        self.game_history.append({
            'state': state,
            'action': action,
            'reward': reward,
            'done': done
        })
        
        # Update performance tracking
        self.total_games += 1
        if done and reward > 0:
            self.total_wins += 1
            self.total_reward += reward
            self.max_reward = max(self.max_reward, reward)
        
        # Update optimization history
        if action is not None:
            self.optimization_history.append({
                'action': action,
                'expected_value': self._calculate_expected_value_continue(state),
                'win_probability': self._calculate_win_probability(state),
                'reward': reward
            })
        
        # Keep only recent history (last 100 games)
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
        
        # Adaptive learning: adjust analytical focus based on performance
        self._adapt_strategy()
    
    def _adapt_strategy(self) -> None:
        """
        Adapt strategy parameters based on recent performance.
        """
        if len(self.optimization_history) < 10:
            return
        
        # Calculate recent performance metrics
        recent_rewards = [h['reward'] for h in self.optimization_history[-10:]]
        recent_ev_accuracy = self._calculate_ev_accuracy()
        
        # Adjust analytical focus based on performance
        if recent_ev_accuracy > 0.7:
            # Increase analytical focus if predictions are accurate
            self.current_analytical_focus = min(0.98, self.current_analytical_focus + 0.01)
        elif recent_ev_accuracy < 0.5:
            # Decrease analytical focus if predictions are inaccurate
            self.current_analytical_focus = max(0.8, self.current_analytical_focus - 0.02)
        
        # Adjust risk tolerance based on recent rewards
        avg_recent_reward = np.mean(recent_rewards)
        if avg_recent_reward > 2.0:
            self.current_risk_tolerance = min(0.7, self.current_risk_tolerance + 0.01)
        elif avg_recent_reward < 1.0:
            self.current_risk_tolerance = max(0.3, self.current_risk_tolerance - 0.01)
    
    def _calculate_ev_accuracy(self) -> float:
        """
        Calculate accuracy of expected value predictions.
        
        Returns:
            Accuracy score (0-1)
        """
        if len(self.optimization_history) < 5:
            return 0.5
        
        recent_history = self.optimization_history[-10:]
        accuracy_scores = []
        
        for h in recent_history:
            predicted_ev = h['expected_value']
            actual_reward = h['reward']
            
            # Calculate accuracy based on how close prediction was to actual
            if actual_reward > 0:
                accuracy = min(1.0, predicted_ev / actual_reward) if actual_reward > 0 else 0.0
            else:
                accuracy = max(0.0, 1.0 - abs(predicted_ev))
            
            accuracy_scores.append(accuracy)
        
        return np.mean(accuracy_scores) if accuracy_scores else 0.5
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get strategy performance statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'name': self.name,
            'description': self.description,
            'total_games': self.total_games,
            'total_wins': self.total_wins,
            'win_rate': self.total_wins / self.total_games if self.total_games > 0 else 0.0,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / self.total_games if self.total_games > 0 else 0.0,
            'max_reward': self.max_reward,
            'current_analytical_focus': self.current_analytical_focus,
            'current_risk_tolerance': self.current_risk_tolerance,
            'ev_accuracy': self._calculate_ev_accuracy(),
            'optimization_history_length': len(self.optimization_history)
        }
    
    def reset(self) -> None:
        """Reset strategy state for new game."""
        self.clicks_made = 0
        self.current_payout = 1.0
        self.game_history = []
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get detailed strategy information.
        
        Returns:
            Dictionary of strategy information
        """
        return {
            'name': self.name,
            'description': self.description,
            'config': {
                'analytical_focus': self.config.analytical_focus,
                'risk_tolerance': self.config.risk_tolerance,
                'cash_out_threshold': self.config.cash_out_threshold,
                'max_clicks': self.config.max_clicks,
                'learning_rate': self.config.learning_rate,
                'confidence_threshold': self.config.confidence_threshold,
                'optimization_weight': self.config.optimization_weight,
                'probability_threshold': self.config.probability_threshold
            },
            'current_state': {
                'current_analytical_focus': self.current_analytical_focus,
                'current_risk_tolerance': self.current_risk_tolerance,
                'ev_accuracy': self._calculate_ev_accuracy()
            }
        }


if __name__ == "__main__":
    # Test the strategy
    strategy = AoiStrategy()
    
    # Test state
    test_state = {
        'board_size': 5,
        'mine_count': 2,
        'clicks_made': 5,
        'current_payout': 2.5,
        'revealed_cells': [(0, 0), (0, 1), (1, 0), (2, 2), (3, 3)]
    }
    
    # Test valid actions
    valid_actions = [(0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
    
    # Test action selection
    action = strategy.select_action(test_state, valid_actions)
    print(f"Selected action: {action}")
    
    # Test statistics
    stats = strategy.get_statistics()
    print(f"Strategy statistics: {stats}")
    
    # Test strategy info
    info = strategy.get_strategy_info()
    print(f"Strategy info: {info}")