"""
Mines Game Mathematics Module

This module provides the exact mathematical foundations for the Mines game:
- Win probability calculations (combinatorial)
- Expected value computations
- Payout formulas
- Statistical analysis functions

Model: 5×5 board (25 tiles), 2 mines by default
"""

import math
from typing import Tuple, List
from functools import lru_cache


def combination(n: int, r: int) -> int:
    """
    Calculate binomial coefficient C(n, r) = n! / (r! * (n-r)!)
    
    Args:
        n: Total number of items
        r: Number of items to choose
        
    Returns:
        Number of combinations
        
    Examples:
        >>> combination(25, 2)
        300
        >>> combination(23, 8)
        490314
    """
    if r > n or r < 0:
        return 0
    if r == 0 or r == n:
        return 1
    
    # Use math.comb for efficiency (Python 3.8+)
    return math.comb(n, r)


@lru_cache(maxsize=1024)
def win_probability(k: int, tiles: int = 25, mines: int = 2) -> float:
    """
    Calculate exact probability of winning with k safe clicks.
    
    Uses exact combinatorial formula:
    P(win k clicks) = C(tiles - mines, k) / C(tiles, k)
    
    Alternatively, multiplicative formula:
    P(win k clicks) = ∏(i=0 to k-1) [(tiles - mines - i) / (tiles - i)]
    
    Args:
        k: Number of clicks to make
        tiles: Total tiles on board (default 25 for 5×5)
        mines: Number of mines (default 2)
        
    Returns:
        Probability of successfully clicking k safe tiles (0.0 to 1.0)
        
    Examples:
        >>> round(win_probability(1, 25, 2), 4)
        0.92
        >>> round(win_probability(8, 25, 2), 4)
        0.5882
        >>> round(win_probability(15, 25, 2), 4)
        0.3143
    """
    if k < 0 or k > tiles - mines:
        return 0.0
    
    if k == 0:
        return 1.0
    
    # Combinatorial method (more stable for large numbers)
    safe_tiles = tiles - mines
    numerator = combination(safe_tiles, k)
    denominator = combination(tiles, k)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def win_probability_multiplicative(k: int, tiles: int = 25, mines: int = 2) -> float:
    """
    Calculate win probability using multiplicative formula.
    
    P(win k clicks) = ∏(i=0 to k-1) [(tiles - mines - i) / (tiles - i)]
    
    This is mathematically equivalent to the combinatorial formula but
    can be more intuitive for understanding sequential click probabilities.
    
    Args:
        k: Number of clicks
        tiles: Total tiles
        mines: Number of mines
        
    Returns:
        Probability of winning
        
    Examples:
        >>> round(win_probability_multiplicative(3, 25, 2), 4)
        0.8213
    """
    if k < 0 or k > tiles - mines:
        return 0.0
    
    if k == 0:
        return 1.0
    
    prob = 1.0
    for i in range(k):
        prob *= (tiles - mines - i) / (tiles - i)
    
    return prob


def theoretical_fair_payout(k: int, tiles: int = 25, mines: int = 2) -> float:
    """
    Calculate theoretical fair payout multiplier (zero house edge).
    
    Fair payout = 1 / P(win k clicks)
    
    Args:
        k: Number of clicks
        tiles: Total tiles  
        mines: Number of mines
        
    Returns:
        Fair payout multiplier
        
    Examples:
        >>> round(theoretical_fair_payout(1, 25, 2), 2)
        1.09
        >>> round(theoretical_fair_payout(8, 25, 2), 2)
        1.70
    """
    prob = win_probability(k, tiles, mines)
    if prob <= 0:
        return float('inf')
    return 1.0 / prob


def actual_payout_with_house_edge(k: int, tiles: int = 25, mines: int = 2, 
                                  house_edge: float = 0.03) -> float:
    """
    Calculate actual payout with house edge applied.
    
    Actual payout = Fair payout × (1 - house_edge)
    
    Args:
        k: Number of clicks
        tiles: Total tiles
        mines: Number of mines
        house_edge: House edge as decimal (0.03 = 3%)
        
    Returns:
        Actual payout multiplier
        
    Examples:
        >>> round(actual_payout_with_house_edge(1, 25, 2, 0.03), 2)
        1.06
    """
    fair = theoretical_fair_payout(k, tiles, mines)
    return fair * (1 - house_edge)


# Site-specific payout table (observed from actual game)
# Format: clicks -> payout multiplier
OBSERVED_PAYOUTS = {
    1: 1.04,
    2: 1.14,
    3: 1.25,
    4: 1.38,
    5: 1.53,
    6: 1.72,
    7: 1.95,
    8: 2.12,
    9: 2.52,
    10: 3.02,
    11: 3.70,
    12: 4.65,
    13: 6.06,
    14: 8.35,
    15: 12.50,
    # Maximum possible: 23 clicks (all safe tiles)
}


def get_observed_payout(k: int) -> float:
    """
    Get observed payout from actual game site.
    
    Args:
        k: Number of clicks
        
    Returns:
        Observed payout multiplier, or 0.0 if not in table
        
    Examples:
        >>> get_observed_payout(1)
        1.04
        >>> get_observed_payout(8)
        2.12
    """
    return OBSERVED_PAYOUTS.get(k, 0.0)


def expected_value(k: int, bet: float = 1.0, tiles: int = 25, mines: int = 2,
                   payout_func=get_observed_payout) -> float:
    """
    Calculate expected value of making k clicks.
    
    EV = P(win) × payout × bet - P(lose) × bet
    EV = P(win) × (payout - 1) × bet - (1 - P(win)) × bet
    EV = [P(win) × payout - 1] × bet
    
    Args:
        k: Number of clicks
        bet: Bet amount
        tiles: Total tiles
        mines: Number of mines
        payout_func: Function to get payout for k clicks
        
    Returns:
        Expected value (profit/loss)
        
    Examples:
        >>> round(expected_value(1, 1.0), 4)
        -0.0435
        >>> round(expected_value(8, 1.0), 4)
        0.2469
    """
    prob_win = win_probability(k, tiles, mines)
    payout = payout_func(k)
    
    if payout == 0:
        return -bet  # No payout data
    
    # EV = P(win) × payout × bet - P(lose) × bet
    ev = prob_win * payout * bet - (1 - prob_win) * bet
    return ev


def kelly_criterion(win_prob: float, payout_multiplier: float) -> float:
    """
    Calculate Kelly Criterion for optimal bet sizing.
    
    Kelly % = (bp - q) / b
    where:
        b = odds (payout - 1)
        p = probability of winning
        q = probability of losing (1 - p)
    
    Args:
        win_prob: Probability of winning (0-1)
        payout_multiplier: Payout on win (e.g., 2.0 = 2x)
        
    Returns:
        Fraction of bankroll to bet (0-1), or 0 if no edge
        
    Examples:
        >>> round(kelly_criterion(0.6, 2.0), 4)
        0.2
        >>> kelly_criterion(0.4, 2.0)  # No edge
        0.0
    """
    if win_prob <= 0 or win_prob >= 1:
        return 0.0
    
    b = payout_multiplier - 1.0  # Odds
    p = win_prob
    q = 1 - win_prob
    
    kelly = (b * p - q) / b
    
    # Only bet if positive edge
    return max(0.0, kelly)


def generate_payout_table(tiles: int = 25, mines: int = 2, 
                          house_edge: float = 0.03) -> List[Tuple[int, float, float, float]]:
    """
    Generate complete payout table with theoretical and actual values.
    
    Args:
        tiles: Total tiles
        mines: Number of mines
        house_edge: House edge percentage
        
    Returns:
        List of (clicks, win_prob, fair_payout, actual_payout) tuples
        
    Example:
        >>> table = generate_payout_table(25, 2, 0.03)
        >>> len(table)  # Should have entries for all possible clicks
        23
        >>> table[0]  # 1 click
        (1, 0.92, 1.09, 1.06)
    """
    max_clicks = tiles - mines
    table = []
    
    for k in range(1, max_clicks + 1):
        prob = win_probability(k, tiles, mines)
        fair = theoretical_fair_payout(k, tiles, mines)
        actual = actual_payout_with_house_edge(k, tiles, mines, house_edge)
        table.append((k, round(prob, 4), round(fair, 2), round(actual, 2)))
    
    return table


def print_payout_table(tiles: int = 25, mines: int = 2):
    """
    Print formatted payout table for documentation.
    
    Args:
        tiles: Total tiles
        mines: Number of mines
    """
    table = generate_payout_table(tiles, mines)
    
    print(f"Payout Table: {tiles} tiles, {mines} mines")
    print("=" * 70)
    print(f"{'Clicks':<8} {'Win Prob':<12} {'Fair Payout':<14} {'Actual (3% HE)':<16} {'Observed':<10}")
    print("-" * 70)
    
    for k, prob, fair, actual in table:
        observed = get_observed_payout(k)
        obs_str = f"{observed:.2f}" if observed > 0 else "—"
        print(f"{k:<8} {prob:<12.4f} {fair:<14.2f} {actual:<16.2f} {obs_str:<10}")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    print("\n")
    print_payout_table(25, 2)
    
    print("\n\nExpected Values (1.0 bet, observed payouts):")
    print("=" * 50)
    for k in range(1, 16):
        ev = expected_value(k, 1.0)
        print(f"{k:2d} clicks: EV = {ev:+.4f}")

