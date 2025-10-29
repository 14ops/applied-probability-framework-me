"""
Kazuya Strategy - Conservative Risk Avoider

This strategy implements a conservative approach with low risk tolerance
and safety-first decision-making.

Key characteristics:
- Low aggression level (0.2-0.4)
- Low risk tolerance (0.2-0.4)
- Cash-out threshold: 2.12x minimum (as specified)
- Safety-first approach
- Early cash-out tendencies
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from ..game.math import win_probability, expected_value, payout_table_25_2, kelly_criterion_fraction


@dataclass
class KazuyaConfig:
    """Configuration for Kazuya strategy."""
    aggression_level: float = 0.3
    risk_tolerance: float = 0.25
    cash_out_threshold: float = 2.12  # Minimum 2.12x as specified
    max_clicks: int = 12
    learning_rate: float = 0.15
    confidence_threshold: float = 0.9
    safety_weight: float = 0.8
    early_cash_out_threshold: float = 1.5


class KazuyaStrategy:
    """
    Kazuya - Conservative Risk Avoider Strategy
    
    Implements a conservative approach with low risk tolerance,
    safety-first decision-making, and a minimum 2.12x cash-out threshold.
    """
    
    def __init__(self, config: Optional[KazuyaConfig] = None):
        self.config = config or KazuyaConfig()
        self.name = "Kazuya"
        self.description = "Conservative risk avoider with safety-first approach"
        
        # Strategy state
        self.current_aggression = self.config.aggression_level
        self.current_risk_tolerance = self.config.risk_tolerance
        self.clicks_made = 0
        self.current_payout = 1.0
        self.game_history = []
        
        # Performance tracking
        self.total_games = 0
        self.total_wins = 0
        self.total_reward = 0.0
        self.max_reward = 0.0
        
        # Safety tracking
        self.safety_scores = []
        self.risk_avoidance_scores = []
        self.recent_performance = []
        
        # Payout table
        self.payout_table = payout_table_25_2()
    
    def select_action(self, state: Dict[str, Any], valid_actions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Select action using Kazuya's conservative approach.
        
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
        
        # Check if we should cash out (Kazuya is quick to cash out)
        if self._should_cash_out(state):
            return None
        
        # Calculate conservative action selection
        action = self._select_conservative_action(state, valid_actions)
        
        return action
    
    def _should_cash_out(self, state: Dict[str, Any]) -> bool:
        """
        Determine if we should cash out based on Kazuya's conservative criteria.
        
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
        
        # Cash out if we've made too many clicks (conservative limit)
        if clicks_made >= self.config.max_clicks:
            return True
        
        # Early cash out if payout is decent and risk is increasing
        if current_payout >= self.config.early_cash_out_threshold:
            risk_level = self._calculate_risk_level(state)
            if risk_level > 0.3:  # Low risk threshold
                return True
        
        # Calculate remaining tiles and mines
        total_tiles = board_size * board_size
        remaining_tiles = total_tiles - clicks_made
        remaining_mines = mine_count
        
        if remaining_tiles <= remaining_mines:
            return True
        
        # Conservative decision: cash out if risk is too high
        risk_level = self._calculate_risk_level(state)
        if risk_level > (1 - self.current_risk_tolerance):
            return True
        
        # Additional safety check: high confidence threshold
        win_prob = self._calculate_win_probability(state)
        if win_prob < self.config.confidence_threshold:
            return True
        
        return False
    
    def _select_conservative_action(self, state: Dict[str, Any], valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select action using conservative decision-making.
        
        Args:
            state: Current game state
            valid_actions: List of valid actions
            
        Returns:
            Selected action
        """
        # Calculate conservative scores for all actions
        action_scores = []
        
        for action in valid_actions:
            score = self._calculate_conservative_score(action, state)
            action_scores.append((action, score))
        
        # Sort by score (highest first)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply conservative approach: choose from safest actions
        if random.random() < (1 - self.current_aggression):
            # Choose from top 20% of actions (safest)
            top_actions = action_scores[:max(1, len(action_scores) // 5)]
            selected_action = random.choice(top_actions)[0]
        else:
            # Choose from top 40% of actions (still conservative)
            top_actions = action_scores[:max(1, len(action_scores) // 2.5)]
            selected_action = random.choice(top_actions)[0]
        
        return selected_action
    
    def _calculate_conservative_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate conservative score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Conservative score (higher is better)
        """
        # Safety score (most important for Kazuya)
        safety_score = self._calculate_safety_score(action, state)
        
        # Risk avoidance score
        risk_avoidance_score = self._calculate_risk_avoidance_score(action, state)
        
        # Conservative expected value
        conservative_ev = self._calculate_conservative_ev(action, state)
        
        # Stability score
        stability_score = self._calculate_stability_score(action, state)
        
        # Combine scores with safety emphasis
        total_score = (
            safety_score * 0.4 +
            risk_avoidance_score * 0.3 +
            conservative_ev * 0.2 +
            stability_score * 0.1
        )
        
        return total_score
    
    def _calculate_safety_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate safety score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Safety score (higher = safer)
        """
        clicks_made = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 2)
        
        # Calculate win probability
        next_clicks = clicks_made + 1
        win_prob = win_probability(next_clicks, board_size * board_size, mine_count)
        
        # Safety score based on win probability
        safety_score = win_prob
        
        # Apply safety weight
        safety_score *= self.config.safety_weight
        
        return safety_score
    
    def _calculate_risk_avoidance_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate risk avoidance score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Risk avoidance score (higher = less risk)
        """
        # Risk avoidance is inverse of risk level
        risk_level = self._calculate_risk_level(state)
        risk_avoidance_score = 1.0 - risk_level
        
        # Apply risk tolerance
        risk_avoidance_score *= (1 - self.current_risk_tolerance)
        
        return risk_avoidance_score
    
    def _calculate_conservative_ev(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate conservative expected value for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Conservative expected value
        """
        clicks_made = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 2)
        
        # Calculate expected value for next click
        next_clicks = clicks_made + 1
        win_prob = win_probability(next_clicks, board_size * board_size, mine_count)
        payout = self.payout_table.get(next_clicks, 1.0)
        
        expected_val = win_prob * payout - (1 - win_prob) * 1.0
        
        # Apply conservative adjustment (reduce expected value)
        conservative_ev = expected_val * (1 - self.current_aggression)
        
        # Normalize to 0-1
        return max(0.0, min(1.0, conservative_ev / 5.0))
    
    def _calculate_stability_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate stability score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Stability score (higher = more stable)
        """
        # Stability based on position (edge positions are more stable)
        row, col = action
        board_size = state.get('board_size', 5)
        
        # Edge positions are more stable
        distance_from_edge = min(row, col, board_size - 1 - row, board_size - 1 - col)
        max_distance = board_size // 2
        
        stability_score = distance_from_edge / max_distance
        
        return stability_score
    
    def _calculate_risk_level(self, state: Dict[str, Any]) -> float:
        """
        Calculate current risk level.
        
        Args:
            state: Current game state
            
        Returns:
            Risk level (0-1)
        """
        clicks_made = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 2)
        
        # Risk increases with more clicks made
        total_tiles = board_size * board_size
        risk_factor = clicks_made / total_tiles
        
        # Risk increases with fewer remaining tiles
        remaining_tiles = total_tiles - clicks_made
        remaining_mines = mine_count
        
        if remaining_tiles <= 0:
            return 1.0
        
        remaining_risk = remaining_mines / remaining_tiles
        
        # Combine risk factors
        total_risk = (risk_factor + remaining_risk) / 2
        
        return min(1.0, total_risk)
    
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
            self.recent_performance.append(1.0)
        else:
            self.recent_performance.append(0.0)
        
        # Update safety tracking
        if action is not None:
            safety_score = self._calculate_safety_score(action, state)
            self.safety_scores.append(safety_score)
            
            risk_avoidance_score = self._calculate_risk_avoidance_score(action, state)
            self.risk_avoidance_scores.append(risk_avoidance_score)
        
        # Keep only recent history (last 50 games)
        if len(self.recent_performance) > 50:
            self.recent_performance = self.recent_performance[-50:]
        if len(self.safety_scores) > 50:
            self.safety_scores = self.safety_scores[-50:]
        if len(self.risk_avoidance_scores) > 50:
            self.risk_avoidance_scores = self.risk_avoidance_scores[-50:]
        
        # Adaptive learning: adjust conservatism based on performance
        self._adapt_strategy()
    
    def _adapt_strategy(self) -> None:
        """
        Adapt strategy parameters based on recent performance.
        """
        if len(self.recent_performance) < 10:
            return
        
        recent_success_rate = np.mean(self.recent_performance[-10:])
        recent_safety_scores = self.safety_scores[-10:] if self.safety_scores else [0.5]
        recent_risk_avoidance = self.risk_avoidance_scores[-10:] if self.risk_avoidance_scores else [0.5]
        
        # Adjust aggression based on performance
        if recent_success_rate > 0.7:
            # Slightly increase aggression if doing well
            self.current_aggression = min(0.5, self.current_aggression + 0.02)
        elif recent_success_rate < 0.3:
            # Decrease aggression if doing poorly
            self.current_aggression = max(0.1, self.current_aggression - 0.02)
        
        # Adjust risk tolerance based on safety scores
        avg_safety_score = np.mean(recent_safety_scores)
        if avg_safety_score > 0.8:
            # Slightly increase risk tolerance if safety is good
            self.current_risk_tolerance = min(0.4, self.current_risk_tolerance + 0.01)
        elif avg_safety_score < 0.5:
            # Decrease risk tolerance if safety is poor
            self.current_risk_tolerance = max(0.1, self.current_risk_tolerance - 0.01)
    
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
            'current_aggression': self.current_aggression,
            'current_risk_tolerance': self.current_risk_tolerance,
            'avg_safety_score': np.mean(self.safety_scores) if self.safety_scores else 0.0,
            'avg_risk_avoidance': np.mean(self.risk_avoidance_scores) if self.risk_avoidance_scores else 0.0,
            'recent_performance': self.recent_performance[-10:] if self.recent_performance else []
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
                'aggression_level': self.config.aggression_level,
                'risk_tolerance': self.config.risk_tolerance,
                'cash_out_threshold': self.config.cash_out_threshold,
                'max_clicks': self.config.max_clicks,
                'learning_rate': self.config.learning_rate,
                'confidence_threshold': self.config.confidence_threshold,
                'safety_weight': self.config.safety_weight,
                'early_cash_out_threshold': self.config.early_cash_out_threshold
            },
            'current_state': {
                'current_aggression': self.current_aggression,
                'current_risk_tolerance': self.current_risk_tolerance,
                'avg_safety_score': np.mean(self.safety_scores) if self.safety_scores else 0.0,
                'avg_risk_avoidance': np.mean(self.risk_avoidance_scores) if self.risk_avoidance_scores else 0.0
            }
        }


if __name__ == "__main__":
    # Test the strategy
    strategy = KazuyaStrategy()
    
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