"""
Advanced statistics engine for simulation analysis.

This module provides comprehensive statistical analysis tools for Monte Carlo
simulation results, including error estimation, convergence analysis, and
hypothesis testing.

Author: Shihab Belal
License: MIT
Version: 2.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from scipy.optimize import curve_fit
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class ConfidenceLevel(Enum):
    """Enumeration of common confidence levels."""
    P90 = 0.90
    P95 = 0.95
    P99 = 0.99
    P99_9 = 0.999


@dataclass
class StatisticalSummary:
    """Container for statistical summary information."""
    mean: float
    std: float
    variance: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    skewness: float
    kurtosis: float
    count: int
    confidence_interval: Tuple[float, float]
    standard_error: float


@dataclass
class ConvergenceAnalysis:
    """Container for convergence analysis results."""
    is_converged: bool
    convergence_rate: float
    required_samples: Optional[int]
    confidence_level: float
    error_bound: float


class StatisticsEngine:
    """
    Advanced statistics engine for simulation analysis.
    
    This class provides comprehensive statistical analysis tools following
    Monte Carlo best practices for error estimation and result validation.
    """
    
    def __init__(self, confidence_level: ConfidenceLevel = ConfidenceLevel.P95):
        """
        Initialize the statistics engine.
        
        Args:
            confidence_level: Default confidence level for analysis
        """
        self.confidence_level = confidence_level
        self._analysis_cache: Dict[str, Any] = {}
    
    def compute_summary_statistics(self, data: np.ndarray, 
                                 confidence_level: Optional[ConfidenceLevel] = None) -> StatisticalSummary:
        """
        Compute comprehensive summary statistics.
        
        Args:
            data: Input data array
            confidence_level: Confidence level for interval calculation
            
        Returns:
            Statistical summary object
        """
        if len(data) == 0:
            raise ValueError("Data array cannot be empty")
        
        conf_level = confidence_level or self.confidence_level
        alpha = 1 - conf_level.value
        
        # Basic statistics
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # Sample standard deviation
        variance = np.var(data, ddof=1)
        min_val = np.min(data)
        max_val = np.max(data)
        median = np.median(data)
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)
        
        # Higher moments
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        # Confidence interval
        n = len(data)
        standard_error = std / np.sqrt(n)
        t_value = stats.t.ppf(1 - alpha/2, n - 1)
        margin_of_error = t_value * standard_error
        
        confidence_interval = (mean - margin_of_error, mean + margin_of_error)
        
        return StatisticalSummary(
            mean=mean,
            std=std,
            variance=variance,
            min=min_val,
            max=max_val,
            median=median,
            q25=q25,
            q75=q75,
            skewness=skewness,
            kurtosis=kurtosis,
            count=n,
            confidence_interval=confidence_interval,
            standard_error=standard_error
        )
    
    def analyze_convergence(self, data: np.ndarray, 
                          target_error: float = 0.01,
                          confidence_level: Optional[ConfidenceLevel] = None) -> ConvergenceAnalysis:
        """
        Analyze convergence of Monte Carlo simulation.
        
        Args:
            data: Simulation results (should be cumulative)
            target_error: Target error bound
            confidence_level: Confidence level for analysis
            
        Returns:
            Convergence analysis results
        """
        if len(data) < 10:
            return ConvergenceAnalysis(
                is_converged=False,
                convergence_rate=0.0,
                required_samples=None,
                confidence_level=confidence_level.value if confidence_level else self.confidence_level.value,
                error_bound=float('inf')
            )
        
        conf_level = confidence_level or self.confidence_level
        alpha = 1 - conf_level.value
        
        # Calculate running statistics
        n_samples = len(data)
        running_means = np.cumsum(data) / np.arange(1, n_samples + 1)
        running_stds = np.array([
            np.std(data[:i+1], ddof=1) for i in range(n_samples)
        ])
        
        # Calculate error bounds
        t_values = np.array([
            stats.t.ppf(1 - alpha/2, i) for i in range(1, n_samples + 1)
        ])
        error_bounds = t_values * running_stds / np.sqrt(np.arange(1, n_samples + 1))
        
        # Check convergence
        is_converged = error_bounds[-1] <= target_error
        
        # Estimate convergence rate
        if n_samples > 1:
            # Fit exponential decay to error bounds
            x = np.arange(1, n_samples + 1)
            y = error_bounds
            
            # Avoid log(0) issues
            y_safe = np.maximum(y, 1e-10)
            log_y = np.log(y_safe)
            
            try:
                # Linear fit to log(error) vs log(n)
                log_x = np.log(x)
                coeffs = np.polyfit(log_x, log_y, 1)
                convergence_rate = -coeffs[0]  # Negative slope
            except:
                convergence_rate = 0.0
        else:
            convergence_rate = 0.0
        
        # Estimate required samples for convergence
        required_samples = None
        if not is_converged and convergence_rate > 0:
            # Solve: target_error = A * n^(-rate)
            # n = (A / target_error)^(1/rate)
            try:
                A = error_bounds[-1] * (n_samples ** convergence_rate)
                required_samples = int(np.ceil((A / target_error) ** (1 / convergence_rate)))
            except:
                required_samples = None
        
        return ConvergenceAnalysis(
            is_converged=is_converged,
            convergence_rate=convergence_rate,
            required_samples=required_samples,
            confidence_level=conf_level.value,
            error_bound=error_bounds[-1]
        )
    
    def bootstrap_analysis(self, data: np.ndarray, statistic: callable,
                         n_bootstrap: int = 1000,
                         confidence_level: Optional[ConfidenceLevel] = None) -> Dict[str, float]:
        """
        Perform bootstrap analysis on data.
        
        Args:
            data: Input data
            statistic: Statistic function to bootstrap
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with bootstrap results
        """
        if len(data) == 0:
            raise ValueError("Data array cannot be empty")
        
        conf_level = confidence_level or self.confidence_level
        alpha = 1 - conf_level.value
        
        # Generate bootstrap samples
        bootstrap_statistics = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_statistics.append(statistic(bootstrap_sample))
        
        bootstrap_statistics = np.array(bootstrap_statistics)
        
        # Calculate bootstrap statistics
        bootstrap_mean = np.mean(bootstrap_statistics)
        bootstrap_std = np.std(bootstrap_statistics, ddof=1)
        
        # Calculate confidence interval
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_statistics, lower_percentile)
        ci_upper = np.percentile(bootstrap_statistics, upper_percentile)
        
        # Bias estimation
        original_statistic = statistic(data)
        bias = bootstrap_mean - original_statistic
        
        return {
            'bootstrap_mean': bootstrap_mean,
            'bootstrap_std': bootstrap_std,
            'confidence_interval_lower': ci_lower,
            'confidence_interval_upper': ci_upper,
            'bias': bias,
            'bias_corrected_estimate': original_statistic - bias,
            'standard_error': bootstrap_std
        }
    
    def compare_distributions(self, data1: np.ndarray, data2: np.ndarray,
                            test_type: str = 'ks') -> Dict[str, Any]:
        """
        Compare two distributions using statistical tests.
        
        Args:
            data1: First dataset
            data2: Second dataset
            test_type: Type of test ('ks', 'ttest', 'mannwhitney')
            
        Returns:
            Dictionary with test results
        """
        if len(data1) == 0 or len(data2) == 0:
            raise ValueError("Both datasets must be non-empty")
        
        results = {}
        
        if test_type == 'ks':
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(data1, data2)
            results = {
                'test': 'Kolmogorov-Smirnov',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        elif test_type == 'ttest':
            # Two-sample t-test
            statistic, p_value = stats.ttest_ind(data1, data2)
            results = {
                'test': 'Two-sample t-test',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        elif test_type == 'mannwhitney':
            # Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            results = {
                'test': 'Mann-Whitney U',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        # Add effect size (Cohen's d for t-test)
        if test_type == 'ttest':
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                 (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                (len(data1) + len(data2) - 2))
            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
            results['effect_size'] = cohens_d
        
        return results
    
    def autocorrelation_analysis(self, data: np.ndarray, max_lag: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze autocorrelation in time series data.
        
        Args:
            data: Time series data
            max_lag: Maximum lag to analyze (default: len(data)//4)
            
        Returns:
            Dictionary with autocorrelation analysis
        """
        if len(data) < 2:
            raise ValueError("Data must have at least 2 points")
        
        if max_lag is None:
            max_lag = len(data) // 4
        
        max_lag = min(max_lag, len(data) - 1)
        
        # Calculate autocorrelation
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Calculate partial autocorrelation (simplified)
        partial_autocorr = []
        for lag in range(1, min(max_lag + 1, len(data))):
            if lag == 1:
                partial_autocorr.append(autocorr[1])
            else:
                # Simplified partial autocorrelation calculation
                partial_autocorr.append(autocorr[lag])
        
        # Find significant lags
        significant_lags = []
        for i, corr in enumerate(autocorr[1:max_lag+1], 1):
            if abs(corr) > 2 / np.sqrt(len(data)):  # Rough significance threshold
                significant_lags.append(i)
        
        return {
            'autocorrelation': autocorr[:max_lag+1].tolist(),
            'partial_autocorrelation': partial_autocorr,
            'significant_lags': significant_lags,
            'max_autocorr': np.max(np.abs(autocorr[1:max_lag+1])),
            'is_stationary': len(significant_lags) == 0
        }
    
    def detect_outliers(self, data: np.ndarray, method: str = 'iqr') -> Dict[str, Any]:
        """
        Detect outliers in data.
        
        Args:
            data: Input data
            method: Detection method ('iqr', 'zscore', 'modified_zscore')
            
        Returns:
            Dictionary with outlier information
        """
        if len(data) == 0:
            raise ValueError("Data array cannot be empty")
        
        outliers = np.array([], dtype=bool)
        
        if method == 'iqr':
            # Interquartile Range method
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            outliers = z_scores > 3
        
        elif method == 'modified_zscore':
            # Modified Z-score method
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = np.abs(modified_z_scores) > 3.5
        
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        outlier_indices = np.where(outliers)[0]
        outlier_values = data[outliers]
        
        return {
            'outlier_indices': outlier_indices.tolist(),
            'outlier_values': outlier_values.tolist(),
            'num_outliers': len(outlier_indices),
            'outlier_percentage': len(outlier_indices) / len(data) * 100,
            'method': method
        }
    
    def calculate_effect_size(self, data1: np.ndarray, data2: np.ndarray) -> Dict[str, float]:
        """
        Calculate effect size between two datasets.
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            Dictionary with effect size measures
        """
        if len(data1) == 0 or len(data2) == 0:
            raise ValueError("Both datasets must be non-empty")
        
        # Cohen's d
        pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                             (len(data2) - 1) * np.var(data2, ddof=1)) / 
                            (len(data1) + len(data2) - 2))
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
        
        # Hedges' g (bias-corrected Cohen's d)
        correction_factor = 1 - (3 / (4 * (len(data1) + len(data2)) - 9))
        hedges_g = cohens_d * correction_factor
        
        # Glass's delta
        glass_delta = (np.mean(data1) - np.mean(data2)) / np.std(data2, ddof=1)
        
        return {
            'cohens_d': cohens_d,
            'hedges_g': hedges_g,
            'glass_delta': glass_delta,
            'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d))
        }
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def clear_cache(self) -> None:
        """Clear analysis cache."""
        self._analysis_cache.clear()