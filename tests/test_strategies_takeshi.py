"""
Unit tests for Takeshi Kovacs Strategy

This module tests the aggressive risk-taker strategy implementation
with focus on the 2.12x minimum cash-out threshold.
"""

import unittest
import numpy as np
from src.python.strategies.takeshi import TakeshiKovacsStrategy, TakeshiConfig


class TestTakeshiStrategy(unittest.TestCase):
    """Test cases for Takeshi Kovacs strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = TakeshiKovacsStrategy()
        self.test_state = {
            'board_size': 5,
            'mine_count': 2,
            'clicks_made': 5,
            'current_payout': 2.5,
            'revealed_cells': [(0, 0), (0, 1), (1, 0), (2, 2), (3, 3)]
        }
        self.valid_actions = [(0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.name, "Takeshi Kovacs")
        self.assertEqual(self.strategy.description, "Aggressive risk-taker with high payout targets")
        self.assertGreater(self.strategy.config.aggression_level, 0.8)
        self.assertGreater(self.strategy.config.risk_tolerance, 0.7)
        self.assertEqual(self.strategy.config.cash_out_threshold, 2.12)
    
    def test_cash_out_threshold(self):
        """Test that strategy respects 2.12x minimum cash-out threshold."""
        # Test state with payout above threshold
        high_payout_state = self.test_state.copy()
        high_payout_state['current_payout'] = 2.5
        
        action = self.strategy.select_action(high_payout_state, self.valid_actions)
        self.assertIsNone(action, "Should cash out when payout is above 2.12x threshold")
    
    def test_cash_out_below_threshold(self):
        """Test that strategy continues when payout is below threshold."""
        # Test state with payout below threshold
        low_payout_state = self.test_state.copy()
        low_payout_state['current_payout'] = 1.5
        
        action = self.strategy.select_action(low_payout_state, self.valid_actions)
        self.assertIsNotNone(action, "Should continue when payout is below 2.12x threshold")
        self.assertIn(action, self.valid_actions)
    
    def test_action_selection(self):
        """Test action selection logic."""
        action = self.strategy.select_action(self.test_state, self.valid_actions)
        
        # Should return a valid action or None
        if action is not None:
            self.assertIn(action, self.valid_actions)
    
    def test_aggressive_behavior(self):
        """Test aggressive behavior characteristics."""
        # Test multiple action selections to see aggressive patterns
        actions = []
        for _ in range(10):
            action = self.strategy.select_action(self.test_state, self.valid_actions)
            if action is not None:
                actions.append(action)
        
        # Should have some actions (not all cash outs)
        self.assertGreater(len(actions), 0)
    
    def test_max_clicks_limit(self):
        """Test that strategy respects max clicks limit."""
        # Test state with max clicks reached
        max_clicks_state = self.test_state.copy()
        max_clicks_state['clicks_made'] = 20  # Max clicks for Takeshi
        
        action = self.strategy.select_action(max_clicks_state, self.valid_actions)
        self.assertIsNone(action, "Should cash out when max clicks reached")
    
    def test_strategy_update(self):
        """Test strategy update mechanism."""
        initial_games = self.strategy.total_games
        
        # Update strategy with a win
        self.strategy.update(self.test_state, (0, 2), 2.5, self.test_state, True)
        
        self.assertEqual(self.strategy.total_games, initial_games + 1)
        self.assertEqual(self.strategy.total_wins, 1)
        self.assertEqual(self.strategy.total_reward, 2.5)
    
    def test_strategy_statistics(self):
        """Test strategy statistics."""
        stats = self.strategy.get_statistics()
        
        # Test required statistics
        required_keys = ['name', 'description', 'total_games', 'total_wins', 'win_rate', 'total_reward', 'avg_reward', 'max_reward']
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Test initial values
        self.assertEqual(stats['total_games'], 0)
        self.assertEqual(stats['total_wins'], 0)
        self.assertEqual(stats['win_rate'], 0.0)
    
    def test_strategy_info(self):
        """Test strategy information."""
        info = self.strategy.get_strategy_info()
        
        # Test required information
        required_keys = ['name', 'description', 'config', 'current_state']
        for key in required_keys:
            self.assertIn(key, info)
        
        # Test configuration
        config = info['config']
        self.assertEqual(config['cash_out_threshold'], 2.12)
        self.assertGreater(config['aggression_level'], 0.8)
        self.assertGreater(config['risk_tolerance'], 0.7)
    
    def test_strategy_reset(self):
        """Test strategy reset functionality."""
        # Update strategy state
        self.strategy.update(self.test_state, (0, 2), 2.5, self.test_state, True)
        
        # Reset strategy
        self.strategy.reset()
        
        # Test reset values
        self.assertEqual(self.strategy.clicks_made, 0)
        self.assertEqual(self.strategy.current_payout, 1.0)
        self.assertEqual(len(self.strategy.game_history), 0)
    
    def test_adaptive_learning(self):
        """Test adaptive learning mechanism."""
        # Simulate poor performance
        for _ in range(10):
            self.strategy.update(self.test_state, (0, 2), 0.0, self.test_state, True)
        
        initial_aggression = self.strategy.current_aggression
        
        # Simulate good performance
        for _ in range(10):
            self.strategy.update(self.test_state, (0, 2), 3.0, self.test_state, True)
        
        # Aggression should have increased
        self.assertGreaterEqual(self.strategy.current_aggression, initial_aggression)
    
    def test_risk_tolerance_adaptation(self):
        """Test risk tolerance adaptation."""
        initial_risk_tolerance = self.strategy.current_risk_tolerance
        
        # Simulate good performance
        for _ in range(10):
            self.strategy.update(self.test_state, (0, 2), 3.0, self.test_state, True)
        
        # Risk tolerance should have increased
        self.assertGreaterEqual(self.strategy.current_risk_tolerance, initial_risk_tolerance)
    
    def test_configuration_override(self):
        """Test custom configuration."""
        custom_config = TakeshiConfig(
            aggression_level=0.9,
            risk_tolerance=0.8,
            cash_out_threshold=3.0,
            max_clicks=15
        )
        
        custom_strategy = TakeshiKovacsStrategy(custom_config)
        
        self.assertEqual(custom_strategy.config.aggression_level, 0.9)
        self.assertEqual(custom_strategy.config.risk_tolerance, 0.8)
        self.assertEqual(custom_strategy.config.cash_out_threshold, 3.0)
        self.assertEqual(custom_strategy.config.max_clicks, 15)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with empty valid actions
        action = self.strategy.select_action(self.test_state, [])
        self.assertIsNone(action)
        
        # Test with single valid action
        action = self.strategy.select_action(self.test_state, [(0, 2)])
        self.assertIn(action, [None, (0, 2)])
    
    def test_consistency_across_games(self):
        """Test consistency across multiple games."""
        # Test that strategy maintains consistent behavior
        actions_1 = []
        actions_2 = []
        
        for _ in range(5):
            action = self.strategy.select_action(self.test_state, self.valid_actions)
            if action is not None:
                actions_1.append(action)
        
        # Reset and test again
        self.strategy.reset()
        
        for _ in range(5):
            action = self.strategy.select_action(self.test_state, self.valid_actions)
            if action is not None:
                actions_2.append(action)
        
        # Should have similar behavior patterns
        self.assertGreater(len(actions_1), 0)
        self.assertGreater(len(actions_2), 0)


class TestTakeshiStrategyIntegration(unittest.TestCase):
    """Integration tests for Takeshi strategy."""
    
    def test_full_game_simulation(self):
        """Test full game simulation with Takeshi strategy."""
        strategy = TakeshiKovacsStrategy()
        
        # Simulate a complete game
        game_state = {
            'board_size': 5,
            'mine_count': 2,
            'clicks_made': 0,
            'current_payout': 1.0,
            'revealed_cells': []
        }
        
        valid_actions = [(r, c) for r in range(5) for c in range(5)]
        
        # Simulate game progression
        for click in range(20):  # Max 20 clicks
            action = strategy.select_action(game_state, valid_actions)
            
            if action is None:  # Cash out
                break
            
            # Simulate action result
            if action in valid_actions:
                valid_actions.remove(action)
                game_state['clicks_made'] += 1
                game_state['current_payout'] *= 1.1  # Simulate payout increase
                game_state['revealed_cells'].append(action)
                
                # Update strategy
                reward = game_state['current_payout'] if len(valid_actions) == 0 else 0.0
                done = len(valid_actions) == 0
                strategy.update(game_state, action, reward, game_state, done)
        
        # Test that strategy completed the game
        self.assertGreater(strategy.total_games, 0)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        strategy = TakeshiKovacsStrategy()
        
        # Simulate multiple games
        for game in range(10):
            # Simulate win
            strategy.update({}, (0, 0), 2.5, {}, True)
        
        for game in range(5):
            # Simulate loss
            strategy.update({}, (0, 0), 0.0, {}, True)
        
        stats = strategy.get_statistics()
        
        # Test calculated metrics
        self.assertEqual(stats['total_games'], 15)
        self.assertEqual(stats['total_wins'], 10)
        self.assertAlmostEqual(stats['win_rate'], 10/15, places=2)
        self.assertEqual(stats['total_reward'], 25.0)
        self.assertAlmostEqual(stats['avg_reward'], 25.0/15, places=2)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
