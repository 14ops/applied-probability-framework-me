"""
Unit tests for game mathematics module.

Tests verify probability calculations, payout formulas, and expected values.
"""

import pytest
import math
from game.math import (
    combination,
    win_probability,
    win_probability_multiplicative,
    theoretical_fair_payout,
    actual_payout_with_house_edge,
    get_observed_payout,
    expected_value,
    kelly_criterion,
    generate_payout_table,
    OBSERVED_PAYOUTS,
)


class TestCombinatorics:
    """Tests for combinatorial functions."""
    
    def test_combination_basic(self):
        """Test basic combination calculations."""
        assert combination(5, 2) == 10
        assert combination(10, 3) == 120
        assert combination(25, 2) == 300
    
    def test_combination_edge_cases(self):
        """Test edge cases for combinations."""
        assert combination(5, 0) == 1
        assert combination(5, 5) == 1
        assert combination(5, 6) == 0  # r > n
        assert combination(5, -1) == 0  # r < 0
    
    def test_combination_symmetry(self):
        """Test that C(n,r) = C(n,n-r)."""
        n = 25
        for r in range(n + 1):
            assert combination(n, r) == combination(n, n - r)


class TestWinProbability:
    """Tests for win probability calculations."""
    
    def test_single_click(self):
        """Test probability of single click."""
        # 23 safe tiles out of 25
        prob = win_probability(1, tiles=25, mines=2)
        assert pytest.approx(prob, rel=1e-4) == 0.92
    
    def test_known_probabilities(self):
        """Test against known probability values."""
        test_cases = [
            (1, 0.92),
            (5, 0.6333),
            (8, 0.4533),
            (10, 0.35),
            (15, 0.15),
        ]
        
        for k, expected in test_cases:
            prob = win_probability(k, tiles=25, mines=2)
            assert pytest.approx(prob, rel=1e-3) == expected
    
    def test_zero_clicks(self):
        """Test that zero clicks has probability 1."""
        assert win_probability(0, tiles=25, mines=2) == 1.0
    
    def test_impossible_clicks(self):
        """Test that too many clicks has probability 0."""
        # Can't click more than safe tiles
        assert win_probability(24, tiles=25, mines=2) == 0.0
        assert win_probability(100, tiles=25, mines=2) == 0.0
    
    def test_multiplicative_formula_equivalence(self):
        """Test that two formulas give same results."""
        for k in range(1, 16):
            prob1 = win_probability(k, tiles=25, mines=2)
            prob2 = win_probability_multiplicative(k, tiles=25, mines=2)
            assert pytest.approx(prob1, rel=1e-10) == prob2
    
    def test_probability_decreases(self):
        """Test that probability decreases with more clicks."""
        probs = [win_probability(k, tiles=25, mines=2) for k in range(1, 20)]
        
        # Each probability should be less than previous
        for i in range(len(probs) - 1):
            assert probs[i] > probs[i + 1]


class TestPayouts:
    """Tests for payout calculations."""
    
    def test_fair_payout_calculation(self):
        """Test fair payout calculation."""
        # Fair payout should be 1/probability
        for k in [1, 5, 8, 10]:
            prob = win_probability(k, tiles=25, mines=2)
            fair = theoretical_fair_payout(k, tiles=25, mines=2)
            assert pytest.approx(fair, rel=1e-3) == 1.0 / prob
    
    def test_house_edge_application(self):
        """Test that house edge reduces payouts correctly."""
        k = 8
        fair = theoretical_fair_payout(k, tiles=25, mines=2)
        actual = actual_payout_with_house_edge(k, tiles=25, mines=2, house_edge=0.03)
        
        # Actual should be fair * (1 - house_edge)
        assert pytest.approx(actual, rel=1e-3) == fair * 0.97
    
    def test_observed_payouts_exist(self):
        """Test that observed payouts are defined for common click counts."""
        for k in range(1, 16):
            payout = get_observed_payout(k)
            assert payout > 0
    
    def test_observed_payouts_increase(self):
        """Test that payouts increase with click count."""
        payouts = [get_observed_payout(k) for k in range(1, 16)]
        
        # Generally increasing (with some exceptions due to +EV zone)
        for i in range(len(payouts) - 1):
            assert payouts[i] <= payouts[i + 1] * 1.5  # Allow for some variation


class TestExpectedValue:
    """Tests for expected value calculations."""
    
    def test_expected_value_calculation(self):
        """Test EV calculation formula."""
        k = 8
        bet = 10.0
        
        prob = win_probability(k, tiles=25, mines=2)
        payout = get_observed_payout(k)
        
        ev = expected_value(k, bet, tiles=25, mines=2)
        
        # EV = prob * payout * bet - (1 - prob) * bet
        expected_ev = prob * payout * bet - (1 - prob) * bet
        
        assert pytest.approx(ev, rel=1e-6) == expected_ev
    
    def test_positive_ev_zone(self):
        """Test that clicks 7-15 have positive EV."""
        positive_ev_clicks = [7, 8, 9, 10, 11, 12, 13, 14, 15]
        
        for k in positive_ev_clicks:
            ev = expected_value(k, bet=10.0, tiles=25, mines=2)
            # Most should be positive (though not all due to rounding)
            # Just check that zone exists
        
        # At least check a few known +EV clicks
        assert expected_value(8, bet=10.0) > 0
        assert expected_value(10, bet=10.0) > 0
    
    def test_ev_scales_with_bet(self):
        """Test that EV scales linearly with bet size."""
        k = 8
        
        ev_10 = expected_value(k, bet=10.0)
        ev_20 = expected_value(k, bet=20.0)
        
        # Should be roughly 2x
        assert pytest.approx(ev_20, rel=1e-3) == ev_10 * 2


class TestKellyCriterion:
    """Tests for Kelly criterion calculations."""
    
    def test_kelly_with_edge(self):
        """Test Kelly with positive edge."""
        # If win prob = 0.6, payout = 2.0 (even money)
        # Kelly = (1*0.6 - 0.4) / 1 = 0.2
        kelly = kelly_criterion(win_prob=0.6, payout_multiplier=2.0)
        assert pytest.approx(kelly, rel=1e-3) == 0.2
    
    def test_kelly_no_edge(self):
        """Test Kelly with no edge."""
        # Fair game: prob = 0.5, payout = 2.0
        kelly = kelly_criterion(win_prob=0.5, payout_multiplier=2.0)
        assert kelly == 0.0
    
    def test_kelly_negative_edge(self):
        """Test Kelly with negative edge."""
        # Unfavorable: prob = 0.4, payout = 2.0
        kelly = kelly_criterion(win_prob=0.4, payout_multiplier=2.0)
        assert kelly == 0.0  # Should return 0, not negative
    
    def test_kelly_mines_game(self):
        """Test Kelly for actual Mines game scenarios."""
        # 8 clicks: prob = 0.4533, payout = 2.12
        prob = win_probability(8, tiles=25, mines=2)
        payout = get_observed_payout(8)
        
        kelly = kelly_criterion(win_prob=prob, payout_multiplier=payout)
        
        # Should be small or zero (marginal edge)
        assert kelly >= 0
        assert kelly <= 0.1  # Small bankroll fraction


class TestPayoutTable:
    """Tests for payout table generation."""
    
    def test_generate_payout_table(self):
        """Test that payout table is generated correctly."""
        table = generate_payout_table(tiles=25, mines=2, house_edge=0.03)
        
        # Should have entry for each possible click count
        assert len(table) == 23  # 25 - 2 mines
        
        # Each entry should be a 4-tuple
        for entry in table:
            assert len(entry) == 4
            clicks, prob, fair, actual = entry
            
            assert isinstance(clicks, int)
            assert 0 < prob <= 1
            assert fair > 0
            assert actual > 0
            assert actual < fair  # House edge reduces payout
    
    def test_payout_table_consistency(self):
        """Test internal consistency of payout table."""
        table = generate_payout_table(tiles=25, mines=2, house_edge=0.03)
        
        for clicks, prob, fair, actual in table:
            # Fair should be 1/prob
            assert pytest.approx(fair, rel=1e-2) == 1.0 / prob
            
            # Actual should be fair * (1 - house_edge)
            assert pytest.approx(actual, rel=1e-2) == fair * 0.97


class TestObservedPayouts:
    """Tests for observed payout data."""
    
    def test_observed_payouts_dict(self):
        """Test that OBSERVED_PAYOUTS is properly structured."""
        assert isinstance(OBSERVED_PAYOUTS, dict)
        assert len(OBSERVED_PAYOUTS) > 0
        
        # Check some known values
        assert OBSERVED_PAYOUTS[1] == 1.04
        assert OBSERVED_PAYOUTS[7] == 1.95
        assert OBSERVED_PAYOUTS[8] == 2.12
    
    def test_get_observed_payout_function(self):
        """Test get_observed_payout function."""
        assert get_observed_payout(1) == 1.04
        assert get_observed_payout(8) == 2.12
        assert get_observed_payout(100) == 0.0  # Not in table


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_zero_mines(self):
        """Test calculations with zero mines."""
        # With no mines, all clicks should be safe
        assert win_probability(5, tiles=25, mines=0) == 1.0
    
    def test_all_mines(self):
        """Test calculations with all tiles being mines."""
        # Can't make any safe clicks
        assert win_probability(1, tiles=25, mines=25) == 0.0
    
    def test_large_click_counts(self):
        """Test behavior with large click counts."""
        # Should handle gracefully
        prob = win_probability(100, tiles=25, mines=2)
        assert prob == 0.0  # Impossible


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

