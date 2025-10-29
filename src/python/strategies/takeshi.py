"""
Takeshi Kovacs Strategy - Aggressive Risk-Taker

This strategy implements an aggressive approach with high risk tolerance,
focusing on maximizing potential rewards through bold decision-making.

Key characteristics:
- High aggression level (0.8-0.9)
- High risk tolerance (0.7-0.8)
- Cash-out threshold: 2.12x minimum (as specified)
- Prefers higher payout targets
- Quick decision-making with minimal hesitation
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from ..game.math import win_probability, expected_value, payout_table_25_2, kelly_criterion_fraction


@dataclass
class TakeshiConfig:
    """Configuration for Takeshi Kovacs strategy."""
    aggression_level: float = 0.85
    risk_tolerance: float = 0.75
    cash_out_threshold: float = 2.12  # Minimum 2.12x as specified
    max_clicks: int = 20
    learning_rate: float = 0.1
    confidence_threshold: float = 0.6
    payout_target_multiplier: float = 1.2


class TakeshiKovacsStrategy:
    """
    Takeshi Kovacs - Aggressive Risk-Taker Strategy
    
    Implements a high-risk, high-reward approach with aggressive decision-making
    and a minimum 2.12x cash-out threshold as specified.
    """
    
    def __init__(self, config: Optional[TakeshiConfig] = None):
        self.config = config or TakeshiConfig()
        self.name = "Takeshi Kovacs"
        self.description = "Aggressive risk-taker with high payout targets"
        
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
        
        # Adaptive learning
        self.success_rate = 0.5
        self.recent_performance = []
        
        # Payout table
        self.payout_table = payout_table_25_2()
    
    def select_action(self, state: Dict[str, Any], valid_actions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Select action using Takeshi's aggressive strategy.
        
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
        
        # Calculate aggressive action selection
        action = self._select_aggressive_action(state, valid_actions)
        
        return action
    
    def _should_cash_out(self, state: Dict[str, Any]) -> bool:
        """
        Determine if we should cash out based on Takeshi's criteria.
        
        Args:
            state: Current game state
            
        Returns:
            True if we should cash out
        """
        current_payout = state.get('current_payout', 1.0)
        clicks_made = state.get('clicks_made', 0)
        
        # Always cash out if we've reached the minimum threshold
        if current_payout >= self.config.cash_out_threshold:
            return True
        
        # Cash out if we've made too many clicks
        if clicks_made >= self.config.max_clicks:
            return True
        
        # Calculate expected value for continuing
        remaining_tiles = state.get('board_size', 5) ** 2 - clicks_made
        remaining_mines = state.get('mine_count', 2)
        
        if remaining_tiles <= remaining_mines:
            return True
        
        # Aggressive threshold: only cash out if payout is very high
        # Takeshi prefers to keep going for higher rewards
        aggressive_threshold = self.config.cash_out_threshold * self.config.payout_target_multiplier
        
        if current_payout >= aggressive_threshold:
            # Still might continue if confidence is high
            confidence = self._calculate_confidence(state)
            if confidence < self.config.confidence_threshold:
                return True
        
        return False
    
    def _select_aggressive_action(self, state: Dict[str, Any], valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select action using aggressive decision-making.
        
        Args:
            state: Current game state
            valid_actions: List of valid actions
            
        Returns:
            Selected action
        """
        # Calculate action scores based on aggression
        action_scores = []
        
        for action in valid_actions:
            score = self._calculate_action_score(action, state)
            action_scores.append((action, score))
        
        # Sort by score (highest first)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply aggression: more likely to choose high-risk, high-reward actions
        if random.random() < self.current_aggression:
            # Choose from top 30% of actions
            top_actions = action_scores[:max(1, len(action_scores) // 3)]
            selected_action = random.choice(top_actions)[0]
        else:
            # Choose from top 50% of actions
            top_actions = action_scores[:max(1, len(action_scores) // 2)]
            selected_action = random.choice(top_actions)[0]
        
        return selected_action
    
    def _calculate_action_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate score for an action based on Takeshi's criteria.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Action score (higher is better)
        """
        row, col = action
        board_size = state.get('board_size', 5)
        clicks_made = state.get('clicks_made', 0)
        current_payout = state.get('current_payout', 1.0)
        
        # Base score from position
        position_score = self._calculate_position_score(row, col, board_size)
        
        # Risk score (Takeshi likes higher risk)
        risk_score = self._calculate_risk_score(action, state)
        
        # Payout potential score
        payout_score = self._calculate_payout_potential(action, state)
        
        # Aggression bonus
        aggression_bonus = self.current_aggression * 0.5
        
        # Combine scores
        total_score = (
            position_score * 0.3 +
            risk_score * 0.4 +
            payout_score * 0.2 +
            aggression_bonus * 0.1
        )
        
        return total_score
    
    def _calculate_position_score(self, row: int, col: int, board_size: int) -> float:
        """
        Calculate position-based score.
        
        Args:
            row: Row position
            col: Column position
            board_size: Size of the board
            
        Returns:
            Position score
        """
        # Takeshi prefers center positions (higher risk, higher reward)
        center = board_size // 2
        distance_from_center = abs(row - center) + abs(col - center)
        max_distance = board_size - 1
        
        # Normalize to 0-1 (closer to center = higher score)
        position_score = 1.0 - (distance_from_center / max_distance)
        
        return position_score
    
    def _calculate_risk_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate risk score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Risk score (higher = more risk, which Takeshi prefers)
        """
        clicks_made = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 2)
        
        # Calculate remaining tiles and mines
        total_tiles = board_size * board_size
        remaining_tiles = total_tiles - clicks_made
        remaining_mines = mine_count
        
        # Risk increases with more clicks made
        risk_factor = clicks_made / total_tiles
        
        # Risk increases with fewer remaining tiles
        remaining_risk = remaining_mines / remaining_tiles if remaining_tiles > 0 else 1.0
        
        # Combine risk factors
        total_risk = (risk_factor + remaining_risk) / 2
        
        return total_risk
    
    def _calculate_payout_potential(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate potential payout score.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Payout potential score
        """
        current_payout = state.get('current_payout', 1.0)
        clicks_made = state.get('clicks_made', 0)
        
        # Calculate potential payout if we continue
        potential_clicks = clicks_made + 1
        potential_payout = self.payout_table.get(potential_clicks, current_payout * 1.1)
        
        # Score based on payout potential
        payout_score = min(potential_payout / 10.0, 1.0)  # Normalize to 0-1
        
        return payout_score
    
    def _calculate_confidence(self, state: Dict[str, Any]) -> float:
        """
        Calculate confidence in current position.
        
        Args:
            state: Current game state
            
        Returns:
            Confidence level (0-1)
        """
        clicks_made = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 2)
        
        # Calculate win probability for current position
        remaining_tiles = board_size * board_size - clicks_made
        remaining_mines = mine_count
        
        if remaining_tiles <= remaining_mines:
            return 0.0
        
        # Calculate probability of winning with current clicks
        win_prob = win_probability(clicks_made, board_size * board_size, mine_count)
        
        # Adjust based on recent performance
        performance_factor = np.mean(self.recent_performance[-10:]) if self.recent_performance else 0.5
        
        # Combine probability and performance
        confidence = (win_prob + performance_factor) / 2
        
        return confidence
    
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
        
        # Keep only recent performance (last 50 games)
        if len(self.recent_performance) > 50:
            self.recent_performance = self.recent_performance[-50:]
        
        # Update success rate
        self.success_rate = self.total_wins / self.total_games if self.total_games > 0 else 0.5
        
        # Adaptive learning: adjust aggression based on performance
        self._adapt_strategy()
    
    def _adapt_strategy(self) -> None:
        """
        Adapt strategy parameters based on recent performance.
        """
        if len(self.recent_performance) < 10:
            return
        
        recent_success_rate = np.mean(self.recent_performance[-10:])
        
        # Adjust aggression based on performance
        if recent_success_rate > 0.6:
            # Increase aggression if doing well
            self.current_aggression = min(0.95, self.current_aggression + 0.05)
        elif recent_success_rate < 0.3:
            # Decrease aggression if doing poorly
            self.current_aggression = max(0.5, self.current_aggression - 0.05)
        
        # Adjust risk tolerance
        if recent_success_rate > 0.7:
            self.current_risk_tolerance = min(0.9, self.current_risk_tolerance + 0.02)
        elif recent_success_rate < 0.4:
            self.current_risk_tolerance = max(0.5, self.current_risk_tolerance - 0.02)
    
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
            'success_rate': self.success_rate,
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
                'payout_target_multiplier': self.config.payout_target_multiplier
            },
            'current_state': {
                'current_aggression': self.current_aggression,
                'current_risk_tolerance': self.current_risk_tolerance,
                'success_rate': self.success_rate
            }
        }


if __name__ == "__main__":
    # Test the strategy
    strategy = TakeshiKovacsStrategy()
    
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