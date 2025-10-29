"""
Mines Game Mathematics Module

This module provides the core mathematical functions for Mines game probability
calculations, payout tables, and expected value computations.

Based on the 25-tile, 2-mine configuration with validated probability calculations.
"""

from math import comb, factorial
from typing import Dict, List, Tuple, Optional
import numpy as np


def win_probability(k: int, tiles: int = 25, mines: int = 2) -> float:
    """
    Calculate the probability of winning with k safe clicks.
    
    Args:
        k: Number of safe clicks (1 to tiles - mines)
        tiles: Total number of tiles (default: 25)
        mines: Number of mines (default: 2)
    
    Returns:
        Probability of winning with k safe clicks
    
    Raises:
        ValueError: If k is invalid for the given configuration
    """
    if k < 1 or k > tiles - mines:
        raise ValueError(f"k must be between 1 and {tiles - mines} for {tiles} tiles with {mines} mines")
    
    if mines < 0 or mines >= tiles:
        raise ValueError(f"Invalid mine count: {mines} for {tiles} tiles")
    
    # Probability = C(tiles - mines, k) / C(tiles, k)
    # This is the probability of selecting k safe tiles out of (tiles - mines) safe tiles
    # when selecting k tiles out of all tiles
    
    safe_tiles = tiles - mines
    
    if k > safe_tiles:
        return 0.0
    
    # Calculate combinations
    numerator = comb(safe_tiles, k)
    denominator = comb(tiles, k)
    
    return numerator / denominator


def payout_table_25_2() -> Dict[int, float]:
    """
    Get the payout table for 25 tiles with 2 mines configuration.
    
    Returns:
        Dictionary mapping number of clicks to payout multiplier
    """
    return {
        1: 1.00,   # No risk, no reward
        2: 1.50,   # 1.5x for 2 clicks
        3: 2.00,   # 2x for 3 clicks
        4: 2.50,   # 2.5x for 4 clicks
        5: 3.00,   # 3x for 5 clicks
        6: 3.50,   # 3.5x for 6 clicks
        7: 4.00,   # 4x for 7 clicks
        8: 4.50,   # 4.5x for 8 clicks
        9: 5.00,   # 5x for 9 clicks
        10: 5.50,  # 5.5x for 10 clicks
        11: 6.00,  # 6x for 11 clicks
        12: 6.50,  # 6.5x for 12 clicks
        13: 7.00,  # 7x for 13 clicks
        14: 7.50,  # 7.5x for 14 clicks
        15: 8.00,  # 8x for 15 clicks
        16: 8.50,  # 8.5x for 16 clicks
        17: 9.00,  # 9x for 17 clicks
        18: 9.50,  # 9.5x for 18 clicks
        19: 10.00, # 10x for 19 clicks
        20: 10.50, # 10.5x for 20 clicks
        21: 11.00, # 11x for 21 clicks
        22: 11.50, # 11.5x for 22 clicks
        23: 12.00, # 12x for 23 clicks (maximum safe clicks)
    }


def payout_table_7x7() -> Dict[int, float]:
    """
    Get the payout table for 7x7 (49 tiles) configuration.
    
    Returns:
        Dictionary mapping number of clicks to payout multiplier
    """
    # Example payout structure for 7x7 with ~10% mine density
    return {
        k: 1.0 + (k * 0.15) for k in range(1, 48)  # 15% increase per click
    }


def payout_table_8x8() -> Dict[int, float]:
    """
    Get the payout table for 8x8 (64 tiles) configuration.
    
    Returns:
        Dictionary mapping number of clicks to payout multiplier
    """
    # Example payout structure for 8x8 with ~12% mine density
    return {
        k: 1.0 + (k * 0.12) for k in range(1, 57)  # 12% increase per click
    }


def expected_value(k: int, tiles: int = 25, mines: int = 2, 
                  payout_table: Optional[Dict[int, float]] = None) -> float:
    """
    Calculate the expected value for k clicks.
    
    Args:
        k: Number of clicks
        tiles: Total number of tiles
        mines: Number of mines
        payout_table: Custom payout table (uses default if None)
    
    Returns:
        Expected value (probability * payout - cost)
    """
    if payout_table is None:
        payout_table = payout_table_25_2()
    
    win_prob = win_probability(k, tiles, mines)
    payout = payout_table.get(k, 1.0)
    
    # Expected value = P(win) * payout - P(lose) * cost
    # Assuming cost = 1 (the bet amount)
    return win_prob * payout - (1 - win_prob) * 1.0


def optimal_clicks(tiles: int = 25, mines: int = 2, 
                  payout_table: Optional[Dict[int, float]] = None) -> Tuple[int, float]:
    """
    Find the optimal number of clicks that maximizes expected value.
    
    Args:
        tiles: Total number of tiles
        mines: Number of mines
        payout_table: Custom payout table (uses default if None)
    
    Returns:
        Tuple of (optimal_clicks, max_expected_value)
    """
    if payout_table is None:
        payout_table = payout_table_25_2()
    
    max_ev = float('-inf')
    optimal_k = 1
    
    for k in range(1, tiles - mines + 1):
        ev = expected_value(k, tiles, mines, payout_table)
        if ev > max_ev:
            max_ev = ev
            optimal_k = k
    
    return optimal_k, max_ev


def risk_reward_ratio(k: int, tiles: int = 25, mines: int = 2) -> float:
    """
    Calculate the risk-reward ratio for k clicks.
    
    Args:
        k: Number of clicks
        tiles: Total number of tiles
        mines: Number of mines
    
    Returns:
        Risk-reward ratio (higher is better)
    """
    win_prob = win_probability(k, tiles, mines)
    payout_table = payout_table_25_2()
    payout = payout_table.get(k, 1.0)
    
    # Risk-reward ratio = (P(win) * payout) / (P(lose) * cost)
    return (win_prob * payout) / ((1 - win_prob) * 1.0)


def kelly_criterion_fraction(k: int, tiles: int = 25, mines: int = 2,
                           payout_table: Optional[Dict[int, float]] = None) -> float:
    """
    Calculate the Kelly Criterion fraction for k clicks.
    
    Args:
        k: Number of clicks
        tiles: Total number of tiles
        mines: Number of mines
        payout_table: Custom payout table (uses default if None)
    
    Returns:
        Kelly Criterion fraction (0 to 1)
    """
    if payout_table is None:
        payout_table = payout_table_25_2()
    
    win_prob = win_probability(k, tiles, mines)
    payout = payout_table.get(k, 1.0)
    
    # Kelly fraction = (bp - q) / b
    # where b = payout - 1, p = win_prob, q = 1 - win_prob
    b = payout - 1
    p = win_prob
    q = 1 - win_prob
    
    if b <= 0:
        return 0.0
    
    kelly_fraction = (b * p - q) / b
    return max(0.0, min(1.0, kelly_fraction))  # Clamp between 0 and 1


def validate_probabilities(tiles: int = 25, mines: int = 2) -> bool:
    """
    Validate that all probabilities sum correctly.
    
    Args:
        tiles: Total number of tiles
        mines: Number of mines
    
    Returns:
        True if probabilities are valid
    """
    total_prob = 0.0
    
    for k in range(1, tiles - mines + 1):
        prob = win_probability(k, tiles, mines)
        total_prob += prob
    
    # Should sum to 1.0 (accounting for floating point precision)
    return abs(total_prob - 1.0) < 1e-10


def get_board_configurations() -> Dict[str, Dict[str, int]]:
    """
    Get available board configurations.
    
    Returns:
        Dictionary of configuration names to their parameters
    """
    return {
        '25_2': {'tiles': 25, 'mines': 2, 'description': '5x5 board, 2 mines (8% density)'},
        '49_5': {'tiles': 49, 'mines': 5, 'description': '7x7 board, 5 mines (10.2% density)'},
        '64_8': {'tiles': 64, 'mines': 8, 'description': '8x8 board, 8 mines (12.5% density)'},
        '100_15': {'tiles': 100, 'mines': 15, 'description': '10x10 board, 15 mines (15% density)'},
    }


def calculate_all_expected_values(tiles: int = 25, mines: int = 2,
                                payout_table: Optional[Dict[int, float]] = None) -> Dict[int, float]:
    """
    Calculate expected values for all possible click counts.
    
    Args:
        tiles: Total number of tiles
        mines: Number of mines
        payout_table: Custom payout table (uses default if None)
    
    Returns:
        Dictionary mapping click count to expected value
    """
    if payout_table is None:
        payout_table = payout_table_25_2()
    
    ev_dict = {}
    for k in range(1, tiles - mines + 1):
        ev_dict[k] = expected_value(k, tiles, mines, payout_table)
    
    return ev_dict


# Validation and testing functions
def run_math_validation() -> Dict[str, bool]:
    """
    Run comprehensive validation of all mathematical functions.
    
    Returns:
        Dictionary of validation results
    """
    results = {}
    
    # Test basic probability calculations
    try:
        prob_1 = win_probability(1, 25, 2)
        prob_23 = win_probability(23, 25, 2)
        results['basic_probabilities'] = 0 < prob_1 < 1 and 0 < prob_23 < 1
    except Exception:
        results['basic_probabilities'] = False
    
    # Test probability validation
    results['probability_validation'] = validate_probabilities(25, 2)
    
    # Test expected value calculations
    try:
        ev_1 = expected_value(1, 25, 2)
        ev_10 = expected_value(10, 25, 2)
        results['expected_values'] = isinstance(ev_1, float) and isinstance(ev_10, float)
    except Exception:
        results['expected_values'] = False
    
    # Test optimal clicks calculation
    try:
        optimal_k, max_ev = optimal_clicks(25, 2)
        results['optimal_clicks'] = 1 <= optimal_k <= 23 and max_ev > 0
    except Exception:
        results['optimal_clicks'] = False
    
    # Test Kelly Criterion
    try:
        kelly_1 = kelly_criterion_fraction(1, 25, 2)
        kelly_10 = kelly_criterion_fraction(10, 25, 2)
        results['kelly_criterion'] = 0 <= kelly_1 <= 1 and 0 <= kelly_10 <= 1
    except Exception:
        results['kelly_criterion'] = False
    
    return results


if __name__ == "__main__":
    # Run validation and print results
    print("Mines Game Mathematics Module Validation")
    print("=" * 50)
    
    validation_results = run_math_validation()
    
    for test_name, passed in validation_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
    
    print("\nExpected Values for 25 tiles, 2 mines:")
    print("-" * 40)
    ev_dict = calculate_all_expected_values(25, 2)
    for k in sorted(ev_dict.keys())[:10]:  # Show first 10
        print(f"k={k:2d}: EV={ev_dict[k]:.4f}")
    
    print("\nOptimal Strategy:")
    print("-" * 20)
    optimal_k, max_ev = optimal_clicks(25, 2)
    print(f"Optimal clicks: {optimal_k}")
    print(f"Maximum EV: {max_ev:.4f}")
    
    print("\nPayout Table (25 tiles, 2 mines):")
    print("-" * 35)
    payout_table = payout_table_25_2()
    for k in sorted(payout_table.keys())[:10]:  # Show first 10
        print(f"k={k:2d}: {payout_table[k]:.2f}x")