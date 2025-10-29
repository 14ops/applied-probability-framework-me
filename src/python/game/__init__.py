"""
Mines Game Core Module

This module contains the core mathematical functions and utilities
for the Mines game probability calculations and payout systems.
"""

from .math import (
    win_probability,
    payout_table_25_2,
    payout_table_7x7,
    payout_table_8x8,
    expected_value,
    optimal_clicks,
    risk_reward_ratio,
    kelly_criterion_fraction,
    validate_probabilities,
    get_board_configurations,
    calculate_all_expected_values,
    run_math_validation
)

__all__ = [
    'win_probability',
    'payout_table_25_2',
    'payout_table_7x7',
    'payout_table_8x8',
    'expected_value',
    'optimal_clicks',
    'risk_reward_ratio',
    'kelly_criterion_fraction',
    'validate_probabilities',
    'get_board_configurations',
    'calculate_all_expected_values',
    'run_math_validation'
]