"""
Yuzu Strategy - Balanced Risk Manager

This strategy implements a balanced approach with moderate risk tolerance
and careful risk management.

Key characteristics:
- Balanced approach (0.5-0.7)
- Moderate risk tolerance (0.4-0.6)
- Cash-out threshold: 2.12x minimum (as specified)
- Risk management focus
- Conservative decision-making with calculated risks
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from ..game.math import win_probability, expected_value, payout_table_25_2, kelly_criterion_fraction


@dataclass
class YuzuConfig:
    """Configuration for Yuzu strategy."""
    balance_level: float = 0.6
    risk_tolerance: float = 0.5
    cash_out_threshold: float = 2.12  # Minimum 2.12x as specified
    max_clicks: int = 16
    learning_rate: float = 0.08
    confidence_threshold: float = 0.7
    risk_management_weight: float = 0.6
    safety_margin: float = 0.1


class YuzuStrategy:
    """
    Yuzu - Balanced Risk Manager Strategy
    
    Implements a balanced approach with moderate risk tolerance,
    careful risk management, and a minimum 2.12x cash-out threshold.
    """
    
    def __init__(self, config: Optional[YuzuConfig] = None):
        self.config = config or YuzuConfig()
        self.name = "Yuzu"
        self.description = "Balanced risk manager with calculated decisions"
        
        # Strategy state
        self.current_balance_level = self.config.balance_level
        self.current_risk_tolerance = self.config.risk_tolerance
        self.clicks_made = 0
        self.current_payout = 1.0
        self.game_history = []
        
        # Performance tracking
        self.total_games = 0
        self.total_wins = 0
        self.total_reward = 0.0
        self.max_reward = 0.0
        
        # Risk management data
        self.risk_assessments = []
        self.safety_margins = []
        self.recent_performance = []
        
        # Payout table
        self.payout_table = payout_table_25_2()
    
    def select_action(self, state: Dict[str, Any], valid_actions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Select action using Yuzu's balanced approach.
        
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
        
        # Calculate balanced action selection
        action = self._select_balanced_action(state, valid_actions)
        
        return action
    
    def _should_cash_out(self, state: Dict[str, Any]) -> bool:
        """
        Determine if we should cash out based on Yuzu's balanced criteria.
        
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
        
        # Balanced decision: consider both risk and reward
        risk_level = self._calculate_risk_level(state)
        reward_potential = self._calculate_reward_potential(state)
        
        # Cash out if risk is too high relative to reward
        if risk_level > (reward_potential * (1 + self.config.safety_margin)):
            return True
        
        # Additional safety check: probability threshold
        win_prob = self._calculate_win_probability(state)
        if win_prob < (self.config.confidence_threshold - self.config.safety_margin):
            return True
        
        return False
    
    def _select_balanced_action(self, state: Dict[str, Any], valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select action using balanced decision-making.
        
        Args:
            state: Current game state
            valid_actions: List of valid actions
            
        Returns:
            Selected action
        """
        # Calculate balanced scores for all actions
        action_scores = []
        
        for action in valid_actions:
            score = self._calculate_balanced_score(action, state)
            action_scores.append((action, score))
        
        # Sort by score (highest first)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply balance level: choose from top actions with some randomness
        if random.random() < self.current_balance_level:
            # Choose from top 40% of actions (balanced approach)
            top_actions = action_scores[:max(1, int(len(action_scores) * 0.4))]
            selected_action = random.choice(top_actions)[0]
        else:
            # Choose from top 60% of actions (more conservative)
            top_actions = action_scores[:max(1, int(len(action_scores) * 0.6))]
            selected_action = random.choice(top_actions)[0]
        
        return selected_action
    
    def _calculate_balanced_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate balanced score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Balanced score (higher is better)
        """
        # Risk-adjusted expected value
        risk_adj_ev = self._calculate_risk_adjusted_ev(action, state)
        
        # Safety score
        safety_score = self._calculate_safety_score(action, state)
        
        # Reward potential score
        reward_score = self._calculate_reward_potential_score(action, state)
        
        # Balance score (combination of risk and reward)
        balance_score = self._calculate_balance_score(action, state)
        
        # Combine scores with weights
        total_score = (
            risk_adj_ev * 0.3 +
            safety_score * 0.3 +
            reward_score * 0.2 +
            balance_score * 0.2
        )
        
        return total_score
    
    def _calculate_risk_adjusted_ev(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate risk-adjusted expected value for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Risk-adjusted expected value
        """
        clicks_made = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 2)
        
        # Calculate expected value for next click
        next_clicks = clicks_made + 1
        win_prob = win_probability(next_clicks, board_size * board_size, mine_count)
        payout = self.payout_table.get(next_clicks, 1.0)
        
        expected_val = win_prob * payout - (1 - win_prob) * 1.0
        
        # Apply risk adjustment
        risk_factor = self._calculate_risk_factor(action, state)
        risk_adjusted_ev = expected_val * (1 - risk_factor * self.current_risk_tolerance)
        
        # Normalize to 0-1
        return max(0.0, min(1.0, risk_adjusted_ev / 5.0))
    
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
        
        # Apply safety margin
        safety_score *= (1 - self.config.safety_margin)
        
        return safety_score
    
    def _calculate_reward_potential_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate reward potential score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Reward potential score
        """
        clicks_made = state.get('clicks_made', 0)
        current_payout = state.get('current_payout', 1.0)
        
        # Calculate potential payout
        next_clicks = clicks_made + 1
        potential_payout = self.payout_table.get(next_clicks, current_payout * 1.1)
        
        # Score based on payout potential
        reward_score = min(potential_payout / 10.0, 1.0)  # Normalize to 0-1
        
        return reward_score
    
    def _calculate_balance_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate balance score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Balance score
        """
        # Balance score based on how well the action balances risk and reward
        risk_level = self._calculate_risk_level(state)
        reward_potential = self._calculate_reward_potential(state)
        
        # Ideal balance is when risk and reward are proportional
        ideal_ratio = 0.5  # 50% risk, 50% reward
        actual_ratio = risk_level / (risk_level + reward_potential) if (risk_level + reward_potential) > 0 else 0.5
        
        # Score based on how close to ideal ratio
        balance_score = 1.0 - abs(actual_ratio - ideal_ratio)
        
        return balance_score
    
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
    
    def _calculate_reward_potential(self, state: Dict[str, Any]) -> float:
        """
        Calculate reward potential.
        
        Args:
            state: Current game state
            
        Returns:
            Reward potential (0-1)
        """
        current_payout = state.get('current_payout', 1.0)
        clicks_made = state.get('clicks_made', 0)
        
        # Calculate potential payout
        next_clicks = clicks_made + 1
        potential_payout = self.payout_table.get(next_clicks, current_payout * 1.1)
        
        # Normalize to 0-1
        return min(potential_payout / 10.0, 1.0)
    
    def _calculate_risk_factor(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate risk factor for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Risk factor (0-1)
        """
        # Risk factor based on position and game state
        row, col = action
        board_size = state.get('board_size', 5)
        
        # Center positions are riskier
        center = board_size // 2
        distance_from_center = abs(row - center) + abs(col - center)
        max_distance = board_size - 1
        
        position_risk = distance_from_center / max_distance
        
        # Game state risk
        game_risk = self._calculate_risk_level(state)
        
        # Combine risks
        total_risk = (position_risk + game_risk) / 2
        
        return total_risk
    
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
        
        # Update risk assessments
        if action is not None:
            risk_level = self._calculate_risk_level(state)
            self.risk_assessments.append(risk_level)
            
            # Calculate safety margin
            safety_margin = self._calculate_safety_margin(state, reward)
            self.safety_margins.append(safety_margin)
        
        # Keep only recent history (last 50 games)
        if len(self.recent_performance) > 50:
            self.recent_performance = self.recent_performance[-50:]
        if len(self.risk_assessments) > 50:
            self.risk_assessments = self.risk_assessments[-50:]
        if len(self.safety_margins) > 50:
            self.safety_margins = self.safety_margins[-50:]
        
        # Adaptive learning: adjust balance and risk tolerance
        self._adapt_strategy()
    
    def _calculate_safety_margin(self, state: Dict[str, Any], reward: float) -> float:
        """
        Calculate safety margin for a decision.
        
        Args:
            state: Game state
            reward: Reward received
            
        Returns:
            Safety margin
        """
        risk_level = self._calculate_risk_level(state)
        reward_potential = self._calculate_reward_potential(state)
        
        # Safety margin is the difference between reward potential and risk level
        safety_margin = reward_potential - risk_level
        
        return safety_margin
    
    def _adapt_strategy(self) -> None:
        """
        Adapt strategy parameters based on recent performance.
        """
        if len(self.recent_performance) < 10:
            return
        
        recent_success_rate = np.mean(self.recent_performance[-10:])
        recent_risk_assessments = self.risk_assessments[-10:] if self.risk_assessments else [0.5]
        recent_safety_margins = self.safety_margins[-10:] if self.safety_margins else [0.0]
        
        # Adjust balance level based on performance
        if recent_success_rate > 0.6:
            # Increase balance level if doing well
            self.current_balance_level = min(0.8, self.current_balance_level + 0.02)
        elif recent_success_rate < 0.4:
            # Decrease balance level if doing poorly
            self.current_balance_level = max(0.4, self.current_balance_level - 0.02)
        
        # Adjust risk tolerance based on safety margins
        avg_safety_margin = np.mean(recent_safety_margins)
        if avg_safety_margin > 0.2:
            # Increase risk tolerance if safety margins are good
            self.current_risk_tolerance = min(0.7, self.current_risk_tolerance + 0.01)
        elif avg_safety_margin < -0.1:
            # Decrease risk tolerance if safety margins are poor
            self.current_risk_tolerance = max(0.3, self.current_risk_tolerance - 0.01)
    
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
            'current_balance_level': self.current_balance_level,
            'current_risk_tolerance': self.current_risk_tolerance,
            'avg_risk_assessment': np.mean(self.risk_assessments) if self.risk_assessments else 0.0,
            'avg_safety_margin': np.mean(self.safety_margins) if self.safety_margins else 0.0,
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
                'balance_level': self.config.balance_level,
                'risk_tolerance': self.config.risk_tolerance,
                'cash_out_threshold': self.config.cash_out_threshold,
                'max_clicks': self.config.max_clicks,
                'learning_rate': self.config.learning_rate,
                'confidence_threshold': self.config.confidence_threshold,
                'risk_management_weight': self.config.risk_management_weight,
                'safety_margin': self.config.safety_margin
            },
            'current_state': {
                'current_balance_level': self.current_balance_level,
                'current_risk_tolerance': self.current_risk_tolerance,
                'avg_risk_assessment': np.mean(self.risk_assessments) if self.risk_assessments else 0.0,
                'avg_safety_margin': np.mean(self.safety_margins) if self.safety_margins else 0.0
            }
        }


if __name__ == "__main__":
    # Test the strategy
    strategy = YuzuStrategy()
    
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