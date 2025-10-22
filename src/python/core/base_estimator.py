"""
Abstract base class for probability estimators.

This module defines the interface for Bayesian and frequentist estimators
that strategies use to make informed decisions under uncertainty.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np


class BaseEstimator(ABC):
    """
    Abstract base class for probability estimators.
    
    Estimators maintain beliefs about unknown probabilities (e.g., win rates,
    expected values) and update these beliefs as evidence is observed. They
    provide point estimates, confidence intervals, and enable decision-making
    under uncertainty.
    
    The framework supports both:
    - Bayesian estimators (Beta, Dirichlet, Gaussian processes)
    - Frequentist estimators (MLE with confidence intervals)
    
    Attributes:
        name: Identifier for the estimator type
    """
    
    def __init__(self, name: str = "BaseEstimator"):
        """
        Initialize the estimator.
        
        Args:
            name: Name for this estimator instance
        """
        self.name = name
        self._num_updates = 0
        
    @abstractmethod
    def estimate(self) -> float:
        """
        Get point estimate of the probability.
        
        Returns:
            Point estimate (e.g., posterior mean for Bayesian,
            MLE for frequentist)
        """
        pass
    
    @abstractmethod
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Get confidence/credible interval for the estimate.
        
        Args:
            confidence: Confidence level (default 95%)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Update the estimator with new observations.
        
        The exact signature depends on the estimator type.
        Common patterns:
            - Binary outcomes: update(successes=k, failures=n-k)
            - Continuous: update(observation=x)
            - Multi-armed bandit: update(arm=i, reward=r)
        """
        self._num_updates += 1
    
    def variance(self) -> float:
        """
        Get variance of the estimate.
        
        Returns:
            Variance of the estimated probability
            
        Note:
            Default implementation returns NaN. Subclasses should override.
        """
        return np.nan
    
    def std(self) -> float:
        """
        Get standard deviation of the estimate.
        
        Returns:
            Standard deviation (square root of variance)
        """
        return np.sqrt(self.variance())
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the posterior/sampling distribution.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Array of samples from the distribution
            
        Note:
            Useful for Thompson Sampling and uncertainty quantification.
            Default returns point estimates. Override for proper sampling.
        """
        return np.full(n_samples, self.estimate())
    
    def get_num_updates(self) -> int:
        """
        Get the number of times this estimator has been updated.
        
        Returns:
            Count of update calls
        """
        return self._num_updates
    
    def reset(self) -> None:
        """
        Reset the estimator to its prior/initial state.
        
        This is useful for starting fresh simulations while keeping
        the same estimator configuration.
        """
        self._num_updates = 0
        
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the estimator.
        
        Returns:
            Dictionary with estimator information
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'num_updates': self._num_updates,
            'estimate': self.estimate(),
            'confidence_interval': self.confidence_interval(),
            'variance': self.variance(),
        }
    
    def __repr__(self) -> str:
        """String representation."""
        est = self.estimate()
        ci = self.confidence_interval()
        return (f"{self.__class__.__name__}(estimate={est:.4f}, "
                f"CI=[{ci[0]:.4f}, {ci[1]:.4f}], "
                f"updates={self._num_updates})")


class BetaEstimator(BaseEstimator):
    """
    Bayesian estimator using Beta distribution (for Bernoulli processes).
    
    The Beta distribution is the conjugate prior for Bernoulli/Binomial
    likelihoods, making it ideal for estimating win probabilities.
    
    Prior: Beta(α, β)
    After observing k successes in n trials:
    Posterior: Beta(α + k, β + n - k)
    
    Attributes:
        alpha: Shape parameter (prior successes + 1)
        beta: Shape parameter (prior failures + 1)
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, name: str = "BetaEstimator"):
        """
        Initialize Beta estimator.
        
        Args:
            alpha: Prior parameter for successes (α > 0)
            beta: Prior parameter for failures (β > 0)
            name: Estimator name
            
        Note:
            alpha=1, beta=1 gives uniform prior (no prior knowledge)
            alpha=beta gives symmetric prior centered at 0.5
        """
        super().__init__(name)
        if alpha <= 0 or beta <= 0:
            raise ValueError("Beta parameters must be positive")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._initial_alpha = alpha
        self._initial_beta = beta
        
    def estimate(self) -> float:
        """
        Get posterior mean estimate.
        
        Returns:
            E[p] = α / (α + β)
        """
        return self.alpha / (self.alpha + self.beta)
    
    def variance(self) -> float:
        """
        Get posterior variance.
        
        Returns:
            Var[p] = αβ / [(α+β)²(α+β+1)]
        """
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
    
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Get credible interval for the probability.
        
        Args:
            confidence: Credible level (default 95%)
            
        Returns:
            (lower, upper) bounds of credible interval
        """
        from scipy import stats
        alpha_level = (1 - confidence) / 2
        lower = stats.beta.ppf(alpha_level, self.alpha, self.beta)
        upper = stats.beta.ppf(1 - alpha_level, self.alpha, self.beta)
        return (lower, upper)
    
    def update(self, successes: int = 0, failures: int = 0, 
               wins: Optional[int] = None, losses: Optional[int] = None) -> None:
        """
        Update with new observations.
        
        Args:
            successes: Number of successful outcomes
            failures: Number of failed outcomes
            wins: Alias for successes (for compatibility)
            losses: Alias for failures (for compatibility)
        """
        # Handle aliases
        if wins is not None:
            successes = wins
        if losses is not None:
            failures = losses
            
        self.alpha += successes
        self.beta += failures
        super().update()
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Draw samples from Beta posterior.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Samples from Beta(α, β)
        """
        return np.random.beta(self.alpha, self.beta, n_samples)
    
    def reset(self) -> None:
        """Reset to initial prior."""
        self.alpha = self._initial_alpha
        self.beta = self._initial_beta
        super().reset()

