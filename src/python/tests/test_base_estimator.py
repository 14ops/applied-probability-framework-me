"""
Unit tests for base estimator classes.

Tests cover the BaseEstimator interface and concrete implementations
like BetaEstimator, ensuring correct probability estimation and
confidence intervals.
"""

import pytest
import numpy as np
from scipy import stats

from core.base_estimator import BaseEstimator, BetaEstimator


class TestBaseEstimator:
    """Tests for BaseEstimator abstract class."""
    
    def test_base_estimator_is_abstract(self):
        """BaseEstimator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEstimator()
    
    def test_subclass_must_implement_abstract_methods(self):
        """Subclasses must implement all abstract methods."""
        class IncompleteEstimator(BaseEstimator):
            pass
        
        with pytest.raises(TypeError):
            IncompleteEstimator()


class TestBetaEstimator:
    """Tests for BetaEstimator implementation."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        est = BetaEstimator()
        assert est.alpha == 1.0
        assert est.beta == 1.0
        assert est.name == "BetaEstimator"
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        est = BetaEstimator(alpha=5.0, beta=3.0, name="custom")
        assert est.alpha == 5.0
        assert est.beta == 3.0
        assert est.name == "custom"
    
    def test_initialization_invalid_alpha(self):
        """Test that negative alpha raises error."""
        with pytest.raises(ValueError, match="Beta parameters must be positive"):
            BetaEstimator(alpha=-1.0, beta=1.0)
    
    def test_initialization_invalid_beta(self):
        """Test that negative beta raises error."""
        with pytest.raises(ValueError, match="Beta parameters must be positive"):
            BetaEstimator(alpha=1.0, beta=-1.0)
    
    def test_initial_estimate_uniform_prior(self):
        """Test that uniform prior (1,1) gives 0.5 estimate."""
        est = BetaEstimator(alpha=1.0, beta=1.0)
        assert est.estimate() == 0.5
    
    def test_estimate_after_updates(self):
        """Test estimate updates correctly."""
        est = BetaEstimator(alpha=1.0, beta=1.0)
        
        # After 10 successes, 0 failures
        est.update(successes=10, failures=0)
        # Estimate should be (1+10)/(1+10+1+0) = 11/12 ≈ 0.917
        assert abs(est.estimate() - 11/12) < 1e-10
        
        # After 5 more successes, 5 failures
        est.update(successes=5, failures=5)
        # Now (11+5)/(11+5+1+5) = 16/22 ≈ 0.727
        assert abs(est.estimate() - 16/22) < 1e-10
    
    def test_update_with_wins_losses_alias(self):
        """Test that wins/losses aliases work."""
        est1 = BetaEstimator(alpha=1.0, beta=1.0)
        est2 = BetaEstimator(alpha=1.0, beta=1.0)
        
        est1.update(successes=5, failures=3)
        est2.update(wins=5, losses=3)
        
        assert est1.estimate() == est2.estimate()
    
    def test_variance_calculation(self):
        """Test variance formula."""
        est = BetaEstimator(alpha=2.0, beta=2.0)
        
        # Variance = αβ / [(α+β)²(α+β+1)]
        expected = (2*2) / ((2+2)**2 * (2+2+1))
        assert abs(est.variance() - expected) < 1e-10
    
    def test_std_calculation(self):
        """Test standard deviation."""
        est = BetaEstimator(alpha=2.0, beta=2.0)
        assert abs(est.std() - np.sqrt(est.variance())) < 1e-10
    
    def test_confidence_interval_default(self):
        """Test 95% confidence interval."""
        est = BetaEstimator(alpha=10.0, beta=5.0)
        lower, upper = est.confidence_interval()
        
        # Check that bounds are valid probabilities
        assert 0 <= lower <= 1
        assert 0 <= upper <= 1
        assert lower < upper
        
        # Verify against scipy
        expected_lower = stats.beta.ppf(0.025, 10.0, 5.0)
        expected_upper = stats.beta.ppf(0.975, 10.0, 5.0)
        assert abs(lower - expected_lower) < 1e-6
        assert abs(upper - expected_upper) < 1e-6
    
    def test_confidence_interval_custom(self):
        """Test custom confidence level."""
        est = BetaEstimator(alpha=10.0, beta=5.0)
        lower, upper = est.confidence_interval(confidence=0.90)
        
        expected_lower = stats.beta.ppf(0.05, 10.0, 5.0)
        expected_upper = stats.beta.ppf(0.95, 10.0, 5.0)
        assert abs(lower - expected_lower) < 1e-6
        assert abs(upper - expected_upper) < 1e-6
    
    def test_sample_returns_correct_shape(self):
        """Test sampling returns correct number of samples."""
        est = BetaEstimator(alpha=5.0, beta=3.0)
        
        samples_1 = est.sample(1)
        assert samples_1.shape == (1,)
        
        samples_100 = est.sample(100)
        assert samples_100.shape == (100,)
    
    def test_sample_distribution(self):
        """Test that samples follow Beta distribution."""
        est = BetaEstimator(alpha=10.0, beta=5.0)
        samples = est.sample(10000)
        
        # Sample mean should be close to theoretical mean
        theoretical_mean = 10.0 / (10.0 + 5.0)
        sample_mean = np.mean(samples)
        assert abs(sample_mean - theoretical_mean) < 0.01
        
        # All samples should be in [0, 1]
        assert np.all((samples >= 0) & (samples <= 1))
    
    def test_reset(self):
        """Test reset returns to initial state."""
        est = BetaEstimator(alpha=2.0, beta=3.0)
        initial_estimate = est.estimate()
        
        # Update
        est.update(successes=10, failures=5)
        assert est.estimate() != initial_estimate
        
        # Reset
        est.reset()
        assert est.estimate() == initial_estimate
        assert est.get_num_updates() == 0
    
    def test_get_metadata(self):
        """Test metadata contains all required fields."""
        est = BetaEstimator(alpha=5.0, beta=3.0, name="test")
        metadata = est.get_metadata()
        
        assert metadata['name'] == "test"
        assert metadata['type'] == "BetaEstimator"
        assert metadata['num_updates'] == 0
        assert 'estimate' in metadata
        assert 'confidence_interval' in metadata
        assert 'variance' in metadata
    
    def test_repr(self):
        """Test string representation."""
        est = BetaEstimator(alpha=5.0, beta=3.0)
        repr_str = repr(est)
        
        assert "BetaEstimator" in repr_str
        assert "estimate" in repr_str
        assert "CI" in repr_str
        assert "updates" in repr_str
    
    def test_update_increments_counter(self):
        """Test that updates increment the counter."""
        est = BetaEstimator()
        assert est.get_num_updates() == 0
        
        est.update(successes=1)
        assert est.get_num_updates() == 1
        
        est.update(failures=1)
        assert est.get_num_updates() == 2


class TestBetaEstimatorStatisticalProperties:
    """Statistical property tests for BetaEstimator."""
    
    def test_high_confidence_with_more_data(self):
        """Confidence interval should narrow with more data."""
        est = BetaEstimator(alpha=1.0, beta=1.0)
        
        ci_initial = est.confidence_interval()
        width_initial = ci_initial[1] - ci_initial[0]
        
        # Add substantial evidence
        est.update(successes=100, failures=100)
        ci_after = est.confidence_interval()
        width_after = ci_after[1] - ci_after[0]
        
        # Interval should be narrower
        assert width_after < width_initial
    
    def test_posterior_concentration(self):
        """Posterior should concentrate around true probability."""
        true_p = 0.7
        n_observations = 1000
        
        # Simulate observations from true probability
        np.random.seed(42)
        successes = np.random.binomial(n_observations, true_p)
        failures = n_observations - successes
        
        est = BetaEstimator(alpha=1.0, beta=1.0)
        est.update(successes=successes, failures=failures)
        
        # Estimate should be close to true probability
        assert abs(est.estimate() - true_p) < 0.05
        
        # True probability should be in confidence interval
        ci = est.confidence_interval()
        assert ci[0] <= true_p <= ci[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

