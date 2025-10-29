"""
Unit tests for the Mines Game Mathematics Module

This module tests all mathematical functions and calculations
for the Mines game probability and payout systems.
"""

import unittest
import numpy as np
from src.python.game.math import (
    win_probability, payout_table_25_2, expected_value, optimal_clicks,
    risk_reward_ratio, kelly_criterion_fraction, validate_probabilities,
    get_board_configurations, calculate_all_expected_values, run_math_validation
)


class TestMinesMath(unittest.TestCase):
    """Test cases for Mines game mathematics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tiles = 25
        self.mines = 2
        self.payout_table = payout_table_25_2()
    
    def test_win_probability_basic(self):
        """Test basic win probability calculations."""
        # Test valid inputs
        self.assertAlmostEqual(win_probability(1, 25, 2), 0.92, places=2)
        self.assertAlmostEqual(win_probability(23, 25, 2), 0.0, places=2)
        
        # Test edge cases
        self.assertEqual(win_probability(1, 25, 2), 23/25)
        self.assertEqual(win_probability(23, 25, 2), 0.0)
    
    def test_win_probability_invalid_inputs(self):
        """Test win probability with invalid inputs."""
        # Test invalid k values
        with self.assertRaises(ValueError):
            win_probability(0, 25, 2)
        with self.assertRaises(ValueError):
            win_probability(24, 25, 2)
        
        # Test invalid mine counts
        with self.assertRaises(ValueError):
            win_probability(1, 25, -1)
        with self.assertRaises(ValueError):
            win_probability(1, 25, 25)
    
    def test_payout_table_25_2(self):
        """Test payout table for 25 tiles, 2 mines."""
        table = payout_table_25_2()
        
        # Test table structure
        self.assertEqual(len(table), 23)  # 1 to 23 clicks
        self.assertEqual(table[1], 1.00)
        self.assertEqual(table[23], 12.00)
        
        # Test increasing payouts
        for i in range(1, 23):
            self.assertGreaterEqual(table[i+1], table[i])
    
    def test_expected_value_calculations(self):
        """Test expected value calculations."""
        # Test basic expected value
        ev_1 = expected_value(1, 25, 2)
        self.assertGreater(ev_1, 0)
        
        # Test expected value with custom payout table
        custom_table = {1: 2.0, 2: 3.0}
        ev_custom = expected_value(1, 25, 2, custom_table)
        self.assertGreater(ev_custom, ev_1)
    
    def test_optimal_clicks(self):
        """Test optimal clicks calculation."""
        optimal_k, max_ev = optimal_clicks(25, 2)
        
        # Test optimal clicks is valid
        self.assertGreaterEqual(optimal_k, 1)
        self.assertLessEqual(optimal_k, 23)
        
        # Test max expected value is positive
        self.assertGreater(max_ev, 0)
        
        # Test that optimal clicks gives maximum expected value
        all_ev = calculate_all_expected_values(25, 2)
        self.assertEqual(all_ev[optimal_k], max_ev)
    
    def test_risk_reward_ratio(self):
        """Test risk-reward ratio calculations."""
        # Test basic risk-reward ratio
        ratio_1 = risk_reward_ratio(1, 25, 2)
        self.assertGreater(ratio_1, 0)
        
        # Test ratio increases with more clicks
        ratio_5 = risk_reward_ratio(5, 25, 2)
        ratio_10 = risk_reward_ratio(10, 25, 2)
        self.assertGreater(ratio_10, ratio_5)
    
    def test_kelly_criterion_fraction(self):
        """Test Kelly Criterion fraction calculations."""
        # Test basic Kelly fraction
        kelly_1 = kelly_criterion_fraction(1, 25, 2)
        self.assertGreaterEqual(kelly_1, 0)
        self.assertLessEqual(kelly_1, 1)
        
        # Test Kelly fraction with custom payout table
        custom_table = {1: 2.0, 2: 3.0}
        kelly_custom = kelly_criterion_fraction(1, 25, 2, custom_table)
        self.assertGreaterEqual(kelly_custom, 0)
        self.assertLessEqual(kelly_custom, 1)
    
    def test_validate_probabilities(self):
        """Test probability validation."""
        # Test valid configuration
        self.assertTrue(validate_probabilities(25, 2))
        
        # Test invalid configurations
        self.assertFalse(validate_probabilities(25, 0))  # No mines
        self.assertFalse(validate_probabilities(25, 25))  # All mines
    
    def test_get_board_configurations(self):
        """Test board configuration retrieval."""
        configs = get_board_configurations()
        
        # Test configuration structure
        self.assertIn('25_2', configs)
        self.assertIn('49_5', configs)
        self.assertIn('64_8', configs)
        
        # Test configuration data
        config_25_2 = configs['25_2']
        self.assertEqual(config_25_2['tiles'], 25)
        self.assertEqual(config_25_2['mines'], 2)
        self.assertIn('description', config_25_2)
    
    def test_calculate_all_expected_values(self):
        """Test calculation of all expected values."""
        ev_dict = calculate_all_expected_values(25, 2)
        
        # Test dictionary structure
        self.assertEqual(len(ev_dict), 23)  # 1 to 23 clicks
        
        # Test expected values are calculated
        for k in range(1, 24):
            self.assertIn(k, ev_dict)
            self.assertIsInstance(ev_dict[k], float)
    
    def test_run_math_validation(self):
        """Test comprehensive math validation."""
        validation_results = run_math_validation()
        
        # Test all validations pass
        for test_name, passed in validation_results.items():
            self.assertTrue(passed, f"Validation failed: {test_name}")
    
    def test_probability_consistency(self):
        """Test probability consistency across different calculations."""
        # Test that win probabilities sum correctly
        total_prob = sum(win_probability(k, 25, 2) for k in range(1, 24))
        self.assertAlmostEqual(total_prob, 1.0, places=10)
    
    def test_expected_value_consistency(self):
        """Test expected value consistency."""
        # Test that expected values are reasonable
        all_ev = calculate_all_expected_values(25, 2)
        
        for k, ev in all_ev.items():
            self.assertIsInstance(ev, float)
            self.assertGreater(ev, -1.0)  # Should not lose more than the bet
            self.assertLess(ev, 10.0)  # Should not be unreasonably high
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test minimum valid configuration
        self.assertEqual(win_probability(1, 3, 1), 2/3)
        
        # Test maximum safe clicks
        self.assertEqual(win_probability(23, 25, 2), 0.0)
        
        # Test single mine configuration
        self.assertEqual(win_probability(1, 2, 1), 0.5)
    
    def test_payout_table_consistency(self):
        """Test payout table consistency."""
        table = payout_table_25_2()
        
        # Test that payouts are increasing
        for i in range(1, 23):
            self.assertGreaterEqual(table[i+1], table[i])
        
        # Test that payouts are reasonable
        for k, payout in table.items():
            self.assertGreater(payout, 0)
            self.assertLessEqual(payout, 20.0)  # Reasonable upper bound
    
    def test_mathematical_properties(self):
        """Test mathematical properties of the functions."""
        # Test that win probability decreases with more clicks
        prob_1 = win_probability(1, 25, 2)
        prob_5 = win_probability(5, 25, 2)
        prob_10 = win_probability(10, 25, 2)
        
        self.assertGreater(prob_1, prob_5)
        self.assertGreater(prob_5, prob_10)
        
        # Test that expected value has a maximum
        all_ev = calculate_all_expected_values(25, 2)
        ev_values = list(all_ev.values())
        max_ev = max(ev_values)
        
        # Should have a clear maximum
        self.assertGreater(max_ev, 0.5)


class TestMinesMathIntegration(unittest.TestCase):
    """Integration tests for Mines game mathematics."""
    
    def test_full_workflow(self):
        """Test complete mathematical workflow."""
        # Test complete workflow from probability to optimal strategy
        tiles = 25
        mines = 2
        
        # Calculate all probabilities
        probabilities = [win_probability(k, tiles, mines) for k in range(1, tiles - mines + 1)]
        
        # Calculate all expected values
        ev_dict = calculate_all_expected_values(tiles, mines)
        
        # Find optimal strategy
        optimal_k, max_ev = optimal_clicks(tiles, mines)
        
        # Test consistency
        self.assertEqual(len(probabilities), len(ev_dict))
        self.assertIn(optimal_k, ev_dict)
        self.assertEqual(ev_dict[optimal_k], max_ev)
        
        # Test that probabilities sum to 1
        self.assertAlmostEqual(sum(probabilities), 1.0, places=10)
    
    def test_different_configurations(self):
        """Test mathematics with different board configurations."""
        configs = get_board_configurations()
        
        for config_name, config in configs.items():
            tiles = config['tiles']
            mines = config['mines']
            
            # Test that configuration is valid
            self.assertTrue(validate_probabilities(tiles, mines))
            
            # Test that optimal clicks can be calculated
            optimal_k, max_ev = optimal_clicks(tiles, mines)
            self.assertGreaterEqual(optimal_k, 1)
            self.assertLessEqual(optimal_k, tiles - mines)
            self.assertGreater(max_ev, 0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
