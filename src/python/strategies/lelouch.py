"""
Lelouch Strategy - Strategic Mastermind

This strategy implements a complex, strategic approach with psychological
elements and long-term planning.

Key characteristics:
- High strategic focus (0.8-0.9)
- Moderate risk tolerance (0.5-0.7)
- Cash-out threshold: 2.12x minimum (as specified)
- Complex decision-making with psychological elements
- Long-term planning and adaptation
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from ..game.math import win_probability, expected_value, payout_table_25_2, kelly_criterion_fraction, optimal_clicks


@dataclass
class LelouchConfig:
    """Configuration for Lelouch strategy."""
    strategic_focus: float = 0.85
    risk_tolerance: float = 0.6
    cash_out_threshold: float = 2.12  # Minimum 2.12x as specified
    max_clicks: int = 22
    learning_rate: float = 0.06
    confidence_threshold: float = 0.75
    psychological_weight: float = 0.4
    long_term_planning: float = 0.6
    adaptation_rate: float = 0.1


class LelouchStrategy:
    """
    Lelouch - Strategic Mastermind Strategy
    
    Implements a complex, strategic approach with psychological elements,
    long-term planning, and a minimum 2.12x cash-out threshold.
    """
    
    def __init__(self, config: Optional[LelouchConfig] = None):
        self.config = config or LelouchConfig()
        self.name = "Lelouch"
        self.description = "Strategic mastermind with psychological elements"
        
        # Strategy state
        self.current_strategic_focus = self.config.strategic_focus
        self.current_risk_tolerance = self.config.risk_tolerance
        self.clicks_made = 0
        self.current_payout = 1.0
        self.game_history = []
        
        # Performance tracking
        self.total_games = 0
        self.total_wins = 0
        self.total_reward = 0.0
        self.max_reward = 0.0
        
        # Strategic data
        self.strategic_plans = []
        self.psychological_profiles = []
        self.adaptation_history = []
        self.long_term_goals = []
        
        # Payout table
        self.payout_table = payout_table_25_2()
    
    def select_action(self, state: Dict[str, Any], valid_actions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Select action using Lelouch's strategic approach.
        
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
        
        # Calculate strategic action selection
        action = self._select_strategic_action(state, valid_actions)
        
        return action
    
    def _should_cash_out(self, state: Dict[str, Any]) -> bool:
        """
        Determine if we should cash out based on Lelouch's strategic criteria.
        
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
        
        # Strategic decision: consider long-term goals and psychological factors
        strategic_analysis = self._analyze_strategic_position(state)
        
        # Cash out if strategic analysis suggests it's optimal
        if strategic_analysis['should_cash_out']:
            return True
        
        # Additional psychological check
        psychological_factor = self._calculate_psychological_factor(state)
        if psychological_factor < 0.3:  # Low confidence
            return True
        
        return False
    
    def _select_strategic_action(self, state: Dict[str, Any], valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select action using strategic decision-making.
        
        Args:
            state: Current game state
            valid_actions: List of valid actions
        
        Returns:
            Selected action
        """
        # Calculate strategic scores for all actions
        action_scores = []
        
        for action in valid_actions:
            score = self._calculate_strategic_score(action, state)
            action_scores.append((action, score))
        
        # Sort by score (highest first)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply strategic focus: choose from top actions with strategic consideration
        if random.random() < self.current_strategic_focus:
            # Choose from top 25% of actions (most strategic)
            top_actions = action_scores[:max(1, len(action_scores) // 4)]
            selected_action = random.choice(top_actions)[0]
        else:
            # Choose from top 50% of actions (still strategic)
            top_actions = action_scores[:max(1, len(action_scores) // 2)]
            selected_action = random.choice(top_actions)[0]
        
        return selected_action
    
    def _calculate_strategic_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate strategic score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Strategic score (higher is better)
        """
        # Strategic expected value
        strategic_ev = self._calculate_strategic_ev(action, state)
        
        # Psychological score
        psychological_score = self._calculate_psychological_score(action, state)
        
        # Long-term planning score
        long_term_score = self._calculate_long_term_score(action, state)
        
        # Adaptation score
        adaptation_score = self._calculate_adaptation_score(action, state)
        
        # Combine scores with strategic weights
        total_score = (
            strategic_ev * 0.3 +
            psychological_score * 0.25 +
            long_term_score * 0.25 +
            adaptation_score * 0.2
        )
        
        return total_score
    
    def _calculate_strategic_ev(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate strategic expected value for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Strategic expected value
        """
        clicks_made = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 2)
        
        # Calculate expected value for next click
        next_clicks = clicks_made + 1
        win_prob = win_probability(next_clicks, board_size * board_size, mine_count)
        payout = self.payout_table.get(next_clicks, 1.0)
        
        expected_val = win_prob * payout - (1 - win_prob) * 1.0
        
        # Apply strategic adjustment
        strategic_ev = expected_val * self.current_strategic_focus
        
        # Normalize to 0-1
        return max(0.0, min(1.0, strategic_ev / 5.0))
    
    def _calculate_psychological_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate psychological score for an action.
        
        Args:
            action: Action to score
            state: Current game state
        
        Returns:
            Psychological score
        """
        # Psychological score based on action characteristics
        row, col = action
        board_size = state.get('board_size', 5)
        
        # Center actions are more psychologically significant
        center = board_size // 2
        distance_from_center = abs(row - center) + abs(col - center)
        max_distance = board_size - 1
        
        center_score = 1.0 - (distance_from_center / max_distance)
        
        # Psychological factor from game state
        psychological_factor = self._calculate_psychological_factor(state)
        
        # Combine psychological elements
        psychological_score = (center_score + psychological_factor) / 2
        
        return psychological_score
    
    def _calculate_long_term_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate long-term planning score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Long-term planning score
        """
        # Long-term score based on strategic position
        strategic_position = self._analyze_strategic_position(state)
        
        # Long-term goals consideration
        long_term_goals = self._get_long_term_goals(state)
        
        # Combine long-term factors
        long_term_score = (strategic_position['long_term_value'] + long_term_goals) / 2
        
        return long_term_score
    
    def _calculate_adaptation_score(self, action: Tuple[int, int], state: Dict[str, Any]) -> float:
        """
        Calculate adaptation score for an action.
        
        Args:
            action: Action to score
            state: Current game state
            
        Returns:
            Adaptation score
        """
        # Adaptation score based on recent performance
        if len(self.adaptation_history) < 5:
            return 0.5
        
        recent_performance = self.adaptation_history[-5:]
        avg_performance = np.mean(recent_performance)
        
        # Adaptation score based on performance trend
        if len(recent_performance) >= 3:
            trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
            adaptation_score = 0.5 + trend * 0.5
        else:
            adaptation_score = avg_performance
        
        return max(0.0, min(1.0, adaptation_score))
    
    def _analyze_strategic_position(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze strategic position and provide recommendations.
        
        Args:
            state: Current game state
            
        Returns:
            Strategic analysis dictionary
        """
        clicks_made = state.get('clicks_made', 0)
        current_payout = state.get('current_payout', 1.0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 2)
        
        # Calculate strategic metrics
        win_prob = self._calculate_win_probability(state)
        expected_val = self._calculate_expected_value_continue(state)
        
        # Strategic decision factors
        should_cash_out = False
        long_term_value = 0.0
        
        # Cash out if expected value is declining
        if expected_val < current_payout * 0.8:
            should_cash_out = True
        
        # Long-term value based on strategic position
        long_term_value = win_prob * expected_val
        
        return {
            'should_cash_out': should_cash_out,
            'long_term_value': long_term_value,
            'win_probability': win_prob,
            'expected_value': expected_val,
            'strategic_confidence': (win_prob + expected_val) / 2
        }
    
    def _calculate_psychological_factor(self, state: Dict[str, Any]) -> float:
        """
        Calculate psychological factor for current state.
        
        Args:
            state: Current game state
            
        Returns:
            Psychological factor (0-1)
        """
        clicks_made = state.get('clicks_made', 0)
        current_payout = state.get('current_payout', 1.0)
        
        # Psychological factor based on game progress
        progress_factor = clicks_made / 20.0  # Normalize to 0-1
        
        # Psychological factor based on payout
        payout_factor = min(current_payout / 5.0, 1.0)  # Normalize to 0-1
        
        # Combine psychological factors
        psychological_factor = (progress_factor + payout_factor) / 2
        
        return psychological_factor
    
    def _get_long_term_goals(self, state: Dict[str, Any]) -> float:
        """
        Get long-term goals score.
        
        Args:
            state: Current game state
            
        Returns:
            Long-term goals score
        """
        # Long-term goals based on strategic objectives
        current_payout = state.get('current_payout', 1.0)
        clicks_made = state.get('clicks_made', 0)
        
        # Goal: maximize payout while minimizing risk
        goal_score = min(current_payout / 10.0, 1.0)  # Normalize to 0-1
        
        return goal_score
    
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
        
        # Update strategic data
        if action is not None:
            strategic_analysis = self._analyze_strategic_position(state)
            self.strategic_plans.append(strategic_analysis)
            
            psychological_factor = self._calculate_psychological_factor(state)
            self.psychological_profiles.append(psychological_factor)
            
            # Update adaptation history
            self.adaptation_history.append(reward)
        
        # Keep only recent history (last 100 games)
        if len(self.strategic_plans) > 100:
            self.strategic_plans = self.strategic_plans[-100:]
        if len(self.psychological_profiles) > 100:
            self.psychological_profiles = self.psychological_profiles[-100:]
        if len(self.adaptation_history) > 100:
            self.adaptation_history = self.adaptation_history[-100:]
        
        # Adaptive learning: adjust strategic focus and risk tolerance
        self._adapt_strategy()
    
    def _adapt_strategy(self) -> None:
        """
        Adapt strategy parameters based on recent performance.
        """
        if len(self.adaptation_history) < 10:
            return
        
        recent_rewards = self.adaptation_history[-10:]
        recent_strategic_plans = self.strategic_plans[-10:] if self.strategic_plans else []
        recent_psychological = self.psychological_profiles[-10:] if self.psychological_profiles else []
        
        # Adjust strategic focus based on performance
        avg_reward = np.mean(recent_rewards)
        if avg_reward > 2.0:
            # Increase strategic focus if doing well
            self.current_strategic_focus = min(0.95, self.current_strategic_focus + 0.01)
        elif avg_reward < 1.0:
            # Decrease strategic focus if doing poorly
            self.current_strategic_focus = max(0.7, self.current_strategic_focus - 0.02)
        
        # Adjust risk tolerance based on strategic analysis
        if recent_strategic_plans:
            avg_strategic_confidence = np.mean([p['strategic_confidence'] for p in recent_strategic_plans])
            if avg_strategic_confidence > 0.7:
                self.current_risk_tolerance = min(0.8, self.current_risk_tolerance + 0.01)
            elif avg_strategic_confidence < 0.5:
                self.current_risk_tolerance = max(0.4, self.current_risk_tolerance - 0.01)
    
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
            'current_strategic_focus': self.current_strategic_focus,
            'current_risk_tolerance': self.current_risk_tolerance,
            'avg_strategic_confidence': np.mean([p['strategic_confidence'] for p in self.strategic_plans]) if self.strategic_plans else 0.0,
            'avg_psychological_factor': np.mean(self.psychological_profiles) if self.psychological_profiles else 0.0,
            'adaptation_history_length': len(self.adaptation_history)
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
                'strategic_focus': self.config.strategic_focus,
                'risk_tolerance': self.config.risk_tolerance,
                'cash_out_threshold': self.config.cash_out_threshold,
                'max_clicks': self.config.max_clicks,
                'learning_rate': self.config.learning_rate,
                'confidence_threshold': self.config.confidence_threshold,
                'psychological_weight': self.config.psychological_weight,
                'long_term_planning': self.config.long_term_planning,
                'adaptation_rate': self.config.adaptation_rate
            },
            'current_state': {
                'current_strategic_focus': self.current_strategic_focus,
                'current_risk_tolerance': self.current_risk_tolerance,
                'avg_strategic_confidence': np.mean([p['strategic_confidence'] for p in self.strategic_plans]) if self.strategic_plans else 0.0,
                'avg_psychological_factor': np.mean(self.psychological_profiles) if self.psychological_profiles else 0.0
            }
        }


if __name__ == "__main__":
    # Test the strategy
    strategy = LelouchStrategy()
    
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