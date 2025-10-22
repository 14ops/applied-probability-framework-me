"""
Advanced probability engine for Monte Carlo simulations.

This module provides sophisticated probability distributions, sampling methods,
and statistical analysis tools following Monte Carlo best practices.

Author: Shihab Belal
License: MIT
Version: 2.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from scipy.optimize import minimize
import warnings

# Suppress scipy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class DistributionType(Enum):
    """Enumeration of supported probability distributions."""
    UNIFORM = "uniform"
    NORMAL = "normal"
    BETA = "beta"
    GAMMA = "gamma"
    EXPONENTIAL = "exponential"
    POISSON = "poisson"
    BINOMIAL = "binomial"
    CUSTOM = "custom"


@dataclass
class DistributionConfig:
    """Configuration for probability distributions."""
    distribution_type: DistributionType
    parameters: Dict[str, float]
    bounds: Optional[Tuple[float, float]] = None
    custom_function: Optional[Callable] = None


class ProbabilityDistribution(ABC):
    """Abstract base class for probability distributions."""
    
    @abstractmethod
    def sample(self, size: int) -> np.ndarray:
        """Generate random samples from the distribution."""
        pass
    
    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute probability density function."""
        pass
    
    @abstractmethod
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Compute cumulative distribution function."""
        pass
    
    @abstractmethod
    def mean(self) -> float:
        """Compute distribution mean."""
        pass
    
    @abstractmethod
    def variance(self) -> float:
        """Compute distribution variance."""
        pass


class UniformDistribution(ProbabilityDistribution):
    """Uniform distribution implementation."""
    
    def __init__(self, low: float = 0.0, high: float = 1.0):
        self.low = low
        self.high = high
        self._dist = stats.uniform(loc=low, scale=high - low)
    
    def sample(self, size: int) -> np.ndarray:
        return self._dist.rvs(size=size)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist.pdf(x)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist.cdf(x)
    
    def mean(self) -> float:
        return (self.low + self.high) / 2
    
    def variance(self) -> float:
        return (self.high - self.low) ** 2 / 12


class NormalDistribution(ProbabilityDistribution):
    """Normal distribution implementation."""
    
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma
        self._dist = stats.norm(loc=mu, scale=sigma)
    
    def sample(self, size: int) -> np.ndarray:
        return self._dist.rvs(size=size)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist.pdf(x)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist.cdf(x)
    
    def mean(self) -> float:
        return self.mu
    
    def variance(self) -> float:
        return self.sigma ** 2


class BetaDistribution(ProbabilityDistribution):
    """Beta distribution implementation."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self._dist = stats.beta(a=alpha, b=beta)
    
    def sample(self, size: int) -> np.ndarray:
        return self._dist.rvs(size=size)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist.pdf(x)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist.cdf(x)
    
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    def variance(self) -> float:
        return (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))


class CustomDistribution(ProbabilityDistribution):
    """Custom distribution implementation using user-defined functions."""
    
    def __init__(self, sample_func: Callable, pdf_func: Optional[Callable] = None,
                 cdf_func: Optional[Callable] = None, mean: Optional[float] = None,
                 variance: Optional[float] = None):
        self.sample_func = sample_func
        self.pdf_func = pdf_func
        self.cdf_func = cdf_func
        self._mean = mean
        self._variance = variance
    
    def sample(self, size: int) -> np.ndarray:
        return self.sample_func(size)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        if self.pdf_func is None:
            raise NotImplementedError("PDF function not provided")
        return self.pdf_func(x)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        if self.cdf_func is None:
            raise NotImplementedError("CDF function not provided")
        return self.cdf_func(x)
    
    def mean(self) -> float:
        if self._mean is None:
            raise NotImplementedError("Mean not provided")
        return self._mean
    
    def variance(self) -> float:
        if self._variance is None:
            raise NotImplementedError("Variance not provided")
        return self._variance


class ProbabilityEngine:
    """
    Advanced probability engine for Monte Carlo simulations.
    
    This class provides comprehensive probability distribution management,
    sampling methods, and statistical analysis tools following Monte Carlo
    best practices for reproducibility and accuracy.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the probability engine.
        
        Args:
            random_seed: Seed for random number generation (None for random)
        """
        self.random_seed = random_seed
        self._distributions: Dict[str, ProbabilityDistribution] = {}
        self._sampling_history: List[Dict[str, Any]] = []
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def create_distribution(self, name: str, config: DistributionConfig) -> ProbabilityDistribution:
        """
        Create a probability distribution.
        
        Args:
            name: Name for the distribution
            config: Distribution configuration
            
        Returns:
            Created distribution object
        """
        if config.distribution_type == DistributionType.UNIFORM:
            params = config.parameters
            dist = UniformDistribution(
                low=params.get('low', 0.0),
                high=params.get('high', 1.0)
            )
        elif config.distribution_type == DistributionType.NORMAL:
            params = config.parameters
            dist = NormalDistribution(
                mu=params.get('mu', 0.0),
                sigma=params.get('sigma', 1.0)
            )
        elif config.distribution_type == DistributionType.BETA:
            params = config.parameters
            dist = BetaDistribution(
                alpha=params.get('alpha', 1.0),
                beta=params.get('beta', 1.0)
            )
        elif config.distribution_type == DistributionType.CUSTOM:
            if config.custom_function is None:
                raise ValueError("Custom function must be provided for custom distribution")
            dist = CustomDistribution(
                sample_func=config.custom_function,
                pdf_func=config.parameters.get('pdf_func'),
                cdf_func=config.parameters.get('cdf_func'),
                mean=config.parameters.get('mean'),
                variance=config.parameters.get('variance')
            )
        else:
            raise ValueError(f"Unsupported distribution type: {config.distribution_type}")
        
        self._distributions[name] = dist
        return dist
    
    def sample(self, distribution_name: str, size: int) -> np.ndarray:
        """
        Sample from a named distribution.
        
        Args:
            distribution_name: Name of the distribution
            size: Number of samples
            
        Returns:
            Array of samples
        """
        if distribution_name not in self._distributions:
            raise ValueError(f"Distribution '{distribution_name}' not found")
        
        samples = self._distributions[distribution_name].sample(size)
        
        # Record sampling for reproducibility
        self._sampling_history.append({
            'distribution': distribution_name,
            'size': size,
            'seed': self.random_seed,
            'timestamp': np.datetime64('now')
        })
        
        return samples
    
    def importance_sampling(self, target_dist: str, proposal_dist: str, 
                          size: int, weight_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform importance sampling.
        
        Args:
            target_dist: Name of target distribution
            proposal_dist: Name of proposal distribution
            size: Number of samples
            weight_func: Weight function for importance sampling
            
        Returns:
            Tuple of (samples, weights)
        """
        if target_dist not in self._distributions or proposal_dist not in self._distributions:
            raise ValueError("Both distributions must exist")
        
        # Sample from proposal distribution
        samples = self.sample(proposal_dist, size)
        
        # Calculate importance weights
        target_pdf = self._distributions[target_dist].pdf(samples)
        proposal_pdf = self._distributions[proposal_dist].pdf(samples)
        
        # Avoid division by zero
        weights = np.where(proposal_pdf > 1e-10, target_pdf / proposal_pdf, 0.0)
        weights = weights * weight_func(samples)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return samples, weights
    
    def monte_carlo_integration(self, func: Callable, distribution_name: str, 
                              size: int) -> Tuple[float, float]:
        """
        Perform Monte Carlo integration.
        
        Args:
            func: Function to integrate
            distribution_name: Name of sampling distribution
            size: Number of samples
            
        Returns:
            Tuple of (integral_estimate, standard_error)
        """
        samples = self.sample(distribution_name, size)
        function_values = func(samples)
        
        # Monte Carlo estimate
        integral_estimate = np.mean(function_values)
        
        # Standard error
        standard_error = np.std(function_values) / np.sqrt(size)
        
        return integral_estimate, standard_error
    
    def sensitivity_analysis(self, func: Callable, param_ranges: Dict[str, Tuple[float, float]],
                           num_samples: int = 1000) -> Dict[str, float]:
        """
        Perform sensitivity analysis using Monte Carlo methods.
        
        Args:
            func: Function to analyze
            param_ranges: Dictionary of parameter ranges
            num_samples: Number of samples for analysis
            
        Returns:
            Dictionary of sensitivity indices
        """
        param_names = list(param_ranges.keys())
        n_params = len(param_names)
        
        # Generate parameter samples
        param_samples = {}
        for param, (low, high) in param_ranges.items():
            param_samples[param] = np.random.uniform(low, high, num_samples)
        
        # Evaluate function
        function_values = []
        for i in range(num_samples):
            params = {param: param_samples[param][i] for param in param_names}
            function_values.append(func(**params))
        
        function_values = np.array(function_values)
        
        # Calculate sensitivity indices (Sobol indices)
        sensitivity_indices = {}
        total_variance = np.var(function_values)
        
        for param in param_names:
            # Calculate first-order sensitivity index
            param_values = param_samples[param]
            correlation = np.corrcoef(param_values, function_values)[0, 1]
            sensitivity_indices[f"{param}_first_order"] = correlation ** 2
            
            # Calculate total sensitivity index (simplified)
            sensitivity_indices[f"{param}_total"] = abs(correlation)
        
        return sensitivity_indices
    
    def bootstrap_confidence_interval(self, data: np.ndarray, statistic: Callable,
                                    confidence: float = 0.95, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.
        
        Args:
            data: Input data
            statistic: Statistic function to bootstrap
            confidence: Confidence level
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        bootstrap_statistics = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_statistics.append(statistic(bootstrap_sample))
        
        bootstrap_statistics = np.array(bootstrap_statistics)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_statistics, lower_percentile)
        upper_bound = np.percentile(bootstrap_statistics, upper_percentile)
        
        return lower_bound, upper_bound
    
    def get_distribution_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a distribution.
        
        Args:
            name: Distribution name
            
        Returns:
            Dictionary with distribution information
        """
        if name not in self._distributions:
            raise ValueError(f"Distribution '{name}' not found")
        
        dist = self._distributions[name]
        return {
            'name': name,
            'mean': dist.mean(),
            'variance': dist.variance(),
            'std': np.sqrt(dist.variance()),
            'type': type(dist).__name__
        }
    
    def get_sampling_history(self) -> List[Dict[str, Any]]:
        """Get history of all sampling operations."""
        return self._sampling_history.copy()
    
    def reset(self) -> None:
        """Reset the engine state."""
        self._distributions.clear()
        self._sampling_history.clear()
        if self.random_seed is not None:
            np.random.seed(self.random_seed)