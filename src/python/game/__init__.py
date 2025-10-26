"""
Game Mathematics and Utilities Package

Provides mathematical foundations for probability games including:
- Win probability calculations
- Payout formulas
- Expected value computations
- Kelly criterion for optimal bet sizing
"""

from .math import (
    win_probability,
    win_probability_multiplicative,
    theoretical_fair_payout,
    actual_payout_with_house_edge,
    get_observed_payout,
    expected_value,
    kelly_criterion,
    generate_payout_table,
    print_payout_table,
    OBSERVED_PAYOUTS,
)

__all__ = [
    'win_probability',
    'win_probability_multiplicative',
    'theoretical_fair_payout',
    'actual_payout_with_house_edge',
    'get_observed_payout',
    'expected_value',
    'kelly_criterion',
    'generate_payout_table',
    'print_payout_table',
    'OBSERVED_PAYOUTS',
]

