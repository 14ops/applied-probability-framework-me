from math import isfinite, sqrt
import numpy as np
from typing import Tuple, Optional, Union
from functools import lru_cache

try:
    from scipy.stats import beta
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class BetaEstimator:
    """
    Optimized Beta distribution estimator with vectorized operations and caching.
    
    Performance improvements:
    - Vectorized batch updates
    - Cached probability calculations
    - Memory-efficient storage
    - Fast confidence interval computation
    """
    
    def __init__(self, a=1.0, b=1.0):
        self.a = float(a)
        self.b = float(b)
        self._mean_cache = None
        self._ci_cache = {}
    
    def update(self, wins=0, losses=0, win=None):
        """Update the estimator with wins/losses or a single win/loss."""
        if win is not None:
            if win:
                self.a += 1
            else:
                self.b += 1
        else:
            self.a += wins
            self.b += losses
        
        # Invalidate caches
        self._mean_cache = None
        self._ci_cache.clear()
    
    def update_batch(self, wins_array: np.ndarray, losses_array: np.ndarray):
        """Vectorized batch update for multiple observations."""
        self.a += np.sum(wins_array)
        self.b += np.sum(losses_array)
        self._mean_cache = None
        self._ci_cache.clear()
    
    def mean(self):
        """Cached mean calculation."""
        if self._mean_cache is None:
            self._mean_cache = self.a / (self.a + self.b)
        return self._mean_cache
    
    @lru_cache(maxsize=128)
    def ci(self, alpha=0.05):
        """Compute confidence interval with caching."""
        if HAS_SCIPY:
            return beta.ppf(alpha / 2, self.a, self.b), beta.ppf(
                1 - alpha / 2, self.a, self.b
            )
        else:
            # Normal approximation for large samples
            mean = self.mean()
            n = self.a + self.b
            # Standard error of beta distribution
            std_err = sqrt(self.a * self.b / (n * n * (n + 1)))
            z = 1.96  # For 95% CI (alpha=0.05)
            return max(0, mean - z * std_err), min(1, mean + z * std_err)
    
    def variance(self):
        """Calculate variance of the beta distribution."""
        n = self.a + self.b
        return (self.a * self.b) / (n * n * (n + 1))
    
    def mode(self):
        """Calculate mode of the beta distribution."""
        if self.a > 1 and self.b > 1:
            return (self.a - 1) / (self.a + self.b - 2)
        elif self.a <= 1 and self.b > 1:
            return 0.0
        elif self.a > 1 and self.b <= 1:
            return 1.0
        else:
            return 0.5


class VectorizedBetaEstimator:
    """
    Vectorized beta estimator for multiple simultaneous estimates.
    
    Optimized for batch processing of multiple probability estimates.
    """
    
    def __init__(self, a_values: np.ndarray, b_values: np.ndarray):
        self.a = np.asarray(a_values, dtype=np.float64)
        self.b = np.asarray(b_values, dtype=np.float64)
        self._mean_cache = None
    
    def update_batch(self, wins: np.ndarray, losses: np.ndarray):
        """Vectorized update for all estimators."""
        self.a += wins
        self.b += losses
        self._mean_cache = None
    
    def means(self):
        """Vectorized mean calculation."""
        if self._mean_cache is None:
            self._mean_cache = self.a / (self.a + self.b)
        return self._mean_cache
    
    def variances(self):
        """Vectorized variance calculation."""
        n = self.a + self.b
        return (self.a * self.b) / (n * n * (n + 1))


def fractional_kelly_vectorized(p: np.ndarray, b: np.ndarray, f: float = 0.25) -> np.ndarray:
    """
    Vectorized fractional Kelly criterion calculation.
    
    Args:
        p: Array of win probabilities
        b: Array of net odds (payout - 1)
        f: Fractional Kelly factor
        
    Returns:
        Array of optimal bet fractions
    """
    # Vectorized Kelly calculation
    k = np.where(b > 0, (p * (b + 1) - 1) / b, 0.0)
    return np.maximum(0.0, f * k)


def fractional_kelly(p, b, f=0.25):
    """Single-value fractional Kelly criterion (optimized)."""
    if b <= 0:
        return 0.0
    k = (p * (b + 1) - 1) / b
    return max(0.0, f * k)


class ProbabilityCache:
    """
    High-performance probability calculation cache with LRU eviction.
    
    Optimized for repeated probability calculations in strategy evaluation.
    """
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, key: Tuple) -> Optional[float]:
        """Get cached probability value."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: Tuple, value: float):
        """Cache probability value."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        """Clear all cached values."""
        self.cache.clear()
        self.access_order.clear()


# Global probability cache for shared use across strategies
_probability_cache = ProbabilityCache()
