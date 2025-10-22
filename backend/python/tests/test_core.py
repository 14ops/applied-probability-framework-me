"""
Comprehensive test suite for core framework components.

This module provides extensive unit tests for all core framework components,
following pytest best practices and ensuring high code coverage.

Author: Shihab Belal
License: MIT
Version: 2.0.0
"""

import pytest
import numpy as np
from typing import Dict, Any, List
import tempfile
import os
from unittest.mock import Mock, patch

# Import core modules
from src.core.base import (
    BaseGameModel, BaseStrategy, BaseSimulator,
    SimulationConfig, SimulationResult, SimulationStatus,
    GameInitializationError, SimulationError, StrategyError
)
from src.core.probability_engine import (
    ProbabilityEngine, DistributionType, DistributionConfig,
    UniformDistribution, NormalDistribution, BetaDistribution
)
from src.core.statistics import (
    StatisticsEngine, ConfidenceLevel, StatisticalSummary,
    ConvergenceAnalysis
)


class TestSimulationConfig:
    """Test cases for SimulationConfig class."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = SimulationConfig()
        assert config.num_simulations == 1000
        assert config.max_iterations == 100
        assert config.random_seed is None
        assert config.parallel is False
        assert config.num_workers == 4
        assert config.timeout is None
        assert config.output_level == 1
        assert config.save_intermediate is False
        assert config.output_file is None
    
    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        config = SimulationConfig(
            num_simulations=500,
            max_iterations=50,
            random_seed=42,
            parallel=True,
            num_workers=8,
            timeout=300.0,
            output_level=2,
            save_intermediate=True,
            output_file="test_output.json"
        )
        assert config.num_simulations == 500
        assert config.max_iterations == 50
        assert config.random_seed == 42
        assert config.parallel is True
        assert config.num_workers == 8
        assert config.timeout == 300.0
        assert config.output_level == 2
        assert config.save_intermediate is True
        assert config.output_file == "test_output.json"
    
    def test_validation_errors(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="num_simulations must be positive"):
            SimulationConfig(num_simulations=0)
        
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            SimulationConfig(max_iterations=-1)
        
        with pytest.raises(ValueError, match="num_workers must be positive"):
            SimulationConfig(num_workers=0)
        
        with pytest.raises(ValueError, match="timeout must be positive"):
            SimulationConfig(timeout=-1.0)


class TestSimulationResult:
    """Test cases for SimulationResult class."""
    
    def test_default_initialization(self):
        """Test default result initialization."""
        result = SimulationResult()
        assert result.results == []
        assert result.statistics == {}
        assert isinstance(result.config, SimulationConfig)
        assert result.execution_time == 0.0
        assert result.status == SimulationStatus.PENDING
        assert result.metadata == {}
        assert isinstance(result.timestamp, type(result.timestamp))
    
    def test_add_result(self):
        """Test adding results."""
        result = SimulationResult()
        result.add_result("test_result_1")
        result.add_result("test_result_2")
        assert len(result.results) == 2
        assert result.results == ["test_result_1", "test_result_2"]
    
    def test_get_summary(self):
        """Test getting result summary."""
        result = SimulationResult()
        result.add_result("test_result")
        result.execution_time = 1.5
        result.status = SimulationStatus.COMPLETED
        result.statistics = {"mean": 0.5, "std": 0.1}
        
        summary = result.get_summary()
        assert summary["num_results"] == 1
        assert summary["execution_time"] == 1.5
        assert summary["status"] == "completed"
        assert summary["statistics"] == {"mean": 0.5, "std": 0.1}
        assert "timestamp" in summary


class TestProbabilityEngine:
    """Test cases for ProbabilityEngine class."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = ProbabilityEngine(random_seed=42)
        assert engine.random_seed == 42
        assert len(engine._distributions) == 0
        assert len(engine._sampling_history) == 0
    
    def test_create_uniform_distribution(self):
        """Test creating uniform distribution."""
        engine = ProbabilityEngine()
        config = DistributionConfig(
            distribution_type=DistributionType.UNIFORM,
            parameters={"low": 0.0, "high": 1.0}
        )
        dist = engine.create_distribution("uniform", config)
        assert isinstance(dist, UniformDistribution)
        assert dist.low == 0.0
        assert dist.high == 1.0
    
    def test_create_normal_distribution(self):
        """Test creating normal distribution."""
        engine = ProbabilityEngine()
        config = DistributionConfig(
            distribution_type=DistributionType.NORMAL,
            parameters={"mu": 0.0, "sigma": 1.0}
        )
        dist = engine.create_distribution("normal", config)
        assert isinstance(dist, NormalDistribution)
        assert dist.mu == 0.0
        assert dist.sigma == 1.0
    
    def test_create_beta_distribution(self):
        """Test creating beta distribution."""
        engine = ProbabilityEngine()
        config = DistributionConfig(
            distribution_type=DistributionType.BETA,
            parameters={"alpha": 2.0, "beta": 3.0}
        )
        dist = engine.create_distribution("beta", config)
        assert isinstance(dist, BetaDistribution)
        assert dist.alpha == 2.0
        assert dist.beta == 3.0
    
    def test_sampling(self):
        """Test sampling from distributions."""
        engine = ProbabilityEngine(random_seed=42)
        config = DistributionConfig(
            distribution_type=DistributionType.UNIFORM,
            parameters={"low": 0.0, "high": 1.0}
        )
        engine.create_distribution("uniform", config)
        
        samples = engine.sample("uniform", 1000)
        assert len(samples) == 1000
        assert all(0.0 <= x <= 1.0 for x in samples)
        assert len(engine._sampling_history) == 1
    
    def test_sampling_nonexistent_distribution(self):
        """Test sampling from non-existent distribution."""
        engine = ProbabilityEngine()
        with pytest.raises(ValueError, match="Distribution 'nonexistent' not found"):
            engine.sample("nonexistent", 100)
    
    def test_monte_carlo_integration(self):
        """Test Monte Carlo integration."""
        engine = ProbabilityEngine(random_seed=42)
        config = DistributionConfig(
            distribution_type=DistributionType.UNIFORM,
            parameters={"low": 0.0, "high": 1.0}
        )
        engine.create_distribution("uniform", config)
        
        def func(x):
            return x ** 2
        
        integral, error = engine.monte_carlo_integration(func, "uniform", 10000)
        assert isinstance(integral, float)
        assert isinstance(error, float)
        assert error > 0  # Should have some error estimate
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        engine = ProbabilityEngine(random_seed=42)
        
        def func(x, y):
            return x + 2 * y
        
        param_ranges = {
            "x": (0.0, 1.0),
            "y": (0.0, 1.0)
        }
        
        sensitivity = engine.sensitivity_analysis(func, param_ranges, 1000)
        assert "x_first_order" in sensitivity
        assert "y_first_order" in sensitivity
        assert "x_total" in sensitivity
        assert "y_total" in sensitivity
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval."""
        engine = ProbabilityEngine(random_seed=42)
        data = np.random.normal(0, 1, 100)
        
        def mean_statistic(x):
            return np.mean(x)
        
        lower, upper = engine.bootstrap_confidence_interval(
            data, mean_statistic, confidence=0.95, n_bootstrap=1000
        )
        assert lower < upper
        assert isinstance(lower, float)
        assert isinstance(upper, float)
    
    def test_reset(self):
        """Test engine reset."""
        engine = ProbabilityEngine(random_seed=42)
        config = DistributionConfig(
            distribution_type=DistributionType.UNIFORM,
            parameters={"low": 0.0, "high": 1.0}
        )
        engine.create_distribution("uniform", config)
        engine.sample("uniform", 100)
        
        engine.reset()
        assert len(engine._distributions) == 0
        assert len(engine._sampling_history) == 0


class TestStatisticsEngine:
    """Test cases for StatisticsEngine class."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = StatisticsEngine()
        assert engine.confidence_level == ConfidenceLevel.P95
        assert len(engine._analysis_cache) == 0
    
    def test_compute_summary_statistics(self):
        """Test summary statistics computation."""
        engine = StatisticsEngine()
        data = np.random.normal(0, 1, 1000)
        
        summary = engine.compute_summary_statistics(data)
        assert isinstance(summary, StatisticalSummary)
        assert summary.count == 1000
        assert summary.mean == pytest.approx(0.0, abs=0.1)
        assert summary.std == pytest.approx(1.0, abs=0.1)
        assert summary.min < summary.max
        assert summary.q25 < summary.median < summary.q75
    
    def test_compute_summary_statistics_empty_data(self):
        """Test summary statistics with empty data."""
        engine = StatisticsEngine()
        with pytest.raises(ValueError, match="Data array cannot be empty"):
            engine.compute_summary_statistics(np.array([]))
    
    def test_analyze_convergence(self):
        """Test convergence analysis."""
        engine = StatisticsEngine()
        # Create data that converges
        data = np.cumsum(np.random.normal(0, 1, 1000)) / np.arange(1, 1001)
        
        analysis = engine.analyze_convergence(data, target_error=0.01)
        assert isinstance(analysis, ConvergenceAnalysis)
        assert isinstance(analysis.is_converged, bool)
        assert analysis.convergence_rate >= 0
        assert analysis.confidence_level == 0.95
    
    def test_analyze_convergence_insufficient_data(self):
        """Test convergence analysis with insufficient data."""
        engine = StatisticsEngine()
        data = np.array([1, 2, 3])
        
        analysis = engine.analyze_convergence(data)
        assert not analysis.is_converged
        assert analysis.convergence_rate == 0.0
        assert analysis.required_samples is None
    
    def test_bootstrap_analysis(self):
        """Test bootstrap analysis."""
        engine = StatisticsEngine()
        data = np.random.normal(0, 1, 100)
        
        def mean_statistic(x):
            return np.mean(x)
        
        results = engine.bootstrap_analysis(data, mean_statistic, n_bootstrap=1000)
        assert "bootstrap_mean" in results
        assert "bootstrap_std" in results
        assert "confidence_interval_lower" in results
        assert "confidence_interval_upper" in results
        assert "bias" in results
        assert "standard_error" in results
    
    def test_compare_distributions_ks(self):
        """Test distribution comparison with KS test."""
        engine = StatisticsEngine()
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(0, 1, 100)
        
        results = engine.compare_distributions(data1, data2, test_type='ks')
        assert results['test'] == 'Kolmogorov-Smirnov'
        assert 'statistic' in results
        assert 'p_value' in results
        assert 'significant' in results
    
    def test_compare_distributions_ttest(self):
        """Test distribution comparison with t-test."""
        engine = StatisticsEngine()
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(1, 1, 100)
        
        results = engine.compare_distributions(data1, data2, test_type='ttest')
        assert results['test'] == 'Two-sample t-test'
        assert 'statistic' in results
        assert 'p_value' in results
        assert 'significant' in results
        assert 'effect_size' in results
    
    def test_compare_distributions_mannwhitney(self):
        """Test distribution comparison with Mann-Whitney test."""
        engine = StatisticsEngine()
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(1, 1, 100)
        
        results = engine.compare_distributions(data1, data2, test_type='mannwhitney')
        assert results['test'] == 'Mann-Whitney U'
        assert 'statistic' in results
        assert 'p_value' in results
        assert 'significant' in results
    
    def test_compare_distributions_invalid_test(self):
        """Test distribution comparison with invalid test type."""
        engine = StatisticsEngine()
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(1, 1, 100)
        
        with pytest.raises(ValueError, match="Unsupported test type"):
            engine.compare_distributions(data1, data2, test_type='invalid')
    
    def test_autocorrelation_analysis(self):
        """Test autocorrelation analysis."""
        engine = StatisticsEngine()
        data = np.random.normal(0, 1, 100)
        
        results = engine.autocorrelation_analysis(data, max_lag=10)
        assert 'autocorrelation' in results
        assert 'partial_autocorrelation' in results
        assert 'significant_lags' in results
        assert 'max_autocorr' in results
        assert 'is_stationary' in results
        assert len(results['autocorrelation']) == 11  # 0 to 10 lags
    
    def test_detect_outliers_iqr(self):
        """Test outlier detection with IQR method."""
        engine = StatisticsEngine()
        data = np.array([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        
        results = engine.detect_outliers(data, method='iqr')
        assert 'outlier_indices' in results
        assert 'outlier_values' in results
        assert 'num_outliers' in results
        assert 'outlier_percentage' in results
        assert 'method' in results
        assert results['method'] == 'iqr'
        assert results['num_outliers'] > 0
    
    def test_detect_outliers_zscore(self):
        """Test outlier detection with Z-score method."""
        engine = StatisticsEngine()
        data = np.random.normal(0, 1, 100)
        data[0] = 10  # Add outlier
        
        results = engine.detect_outliers(data, method='zscore')
        assert results['method'] == 'zscore'
        assert results['num_outliers'] > 0
    
    def test_detect_outliers_modified_zscore(self):
        """Test outlier detection with modified Z-score method."""
        engine = StatisticsEngine()
        data = np.random.normal(0, 1, 100)
        data[0] = 10  # Add outlier
        
        results = engine.detect_outliers(data, method='modified_zscore')
        assert results['method'] == 'modified_zscore'
        assert results['num_outliers'] > 0
    
    def test_detect_outliers_invalid_method(self):
        """Test outlier detection with invalid method."""
        engine = StatisticsEngine()
        data = np.random.normal(0, 1, 100)
        
        with pytest.raises(ValueError, match="Unsupported outlier detection method"):
            engine.detect_outliers(data, method='invalid')
    
    def test_calculate_effect_size(self):
        """Test effect size calculation."""
        engine = StatisticsEngine()
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(1, 1, 100)
        
        results = engine.calculate_effect_size(data1, data2)
        assert 'cohens_d' in results
        assert 'hedges_g' in results
        assert 'glass_delta' in results
        assert 'effect_size_interpretation' in results
        assert results['effect_size_interpretation'] in ['negligible', 'small', 'medium', 'large']
    
    def test_clear_cache(self):
        """Test cache clearing."""
        engine = StatisticsEngine()
        engine._analysis_cache['test'] = 'value'
        engine.clear_cache()
        assert len(engine._analysis_cache) == 0


class TestUniformDistribution:
    """Test cases for UniformDistribution class."""
    
    def test_initialization(self):
        """Test uniform distribution initialization."""
        dist = UniformDistribution(low=0.0, high=1.0)
        assert dist.low == 0.0
        assert dist.high == 1.0
    
    def test_sample(self):
        """Test uniform distribution sampling."""
        dist = UniformDistribution(low=0.0, high=1.0)
        samples = dist.sample(1000)
        assert len(samples) == 1000
        assert all(0.0 <= x <= 1.0 for x in samples)
    
    def test_pdf(self):
        """Test uniform distribution PDF."""
        dist = UniformDistribution(low=0.0, high=1.0)
        x = np.array([0.0, 0.5, 1.0])
        pdf_values = dist.pdf(x)
        assert all(pdf_values == 1.0)
    
    def test_cdf(self):
        """Test uniform distribution CDF."""
        dist = UniformDistribution(low=0.0, high=1.0)
        x = np.array([0.0, 0.5, 1.0])
        cdf_values = dist.cdf(x)
        expected = np.array([0.0, 0.5, 1.0])
        assert np.allclose(cdf_values, expected)
    
    def test_mean(self):
        """Test uniform distribution mean."""
        dist = UniformDistribution(low=0.0, high=1.0)
        assert dist.mean() == 0.5
    
    def test_variance(self):
        """Test uniform distribution variance."""
        dist = UniformDistribution(low=0.0, high=1.0)
        assert dist.variance() == 1.0 / 12.0


class TestNormalDistribution:
    """Test cases for NormalDistribution class."""
    
    def test_initialization(self):
        """Test normal distribution initialization."""
        dist = NormalDistribution(mu=0.0, sigma=1.0)
        assert dist.mu == 0.0
        assert dist.sigma == 1.0
    
    def test_sample(self):
        """Test normal distribution sampling."""
        dist = NormalDistribution(mu=0.0, sigma=1.0)
        samples = dist.sample(1000)
        assert len(samples) == 1000
        assert np.mean(samples) == pytest.approx(0.0, abs=0.1)
        assert np.std(samples) == pytest.approx(1.0, abs=0.1)
    
    def test_mean(self):
        """Test normal distribution mean."""
        dist = NormalDistribution(mu=5.0, sigma=2.0)
        assert dist.mean() == 5.0
    
    def test_variance(self):
        """Test normal distribution variance."""
        dist = NormalDistribution(mu=0.0, sigma=3.0)
        assert dist.variance() == 9.0


class TestBetaDistribution:
    """Test cases for BetaDistribution class."""
    
    def test_initialization(self):
        """Test beta distribution initialization."""
        dist = BetaDistribution(alpha=2.0, beta=3.0)
        assert dist.alpha == 2.0
        assert dist.beta == 3.0
    
    def test_sample(self):
        """Test beta distribution sampling."""
        dist = BetaDistribution(alpha=2.0, beta=3.0)
        samples = dist.sample(1000)
        assert len(samples) == 1000
        assert all(0.0 <= x <= 1.0 for x in samples)
    
    def test_mean(self):
        """Test beta distribution mean."""
        dist = BetaDistribution(alpha=2.0, beta=3.0)
        expected_mean = 2.0 / (2.0 + 3.0)
        assert dist.mean() == expected_mean
    
    def test_variance(self):
        """Test beta distribution variance."""
        dist = BetaDistribution(alpha=2.0, beta=3.0)
        expected_var = (2.0 * 3.0) / ((2.0 + 3.0) ** 2 * (2.0 + 3.0 + 1))
        assert dist.variance() == expected_var


# Integration tests
class TestIntegration:
    """Integration tests for core components."""
    
    def test_probability_engine_with_statistics_engine(self):
        """Test integration between probability and statistics engines."""
        prob_engine = ProbabilityEngine(random_seed=42)
        stats_engine = StatisticsEngine()
        
        # Create and sample from distribution
        config = DistributionConfig(
            distribution_type=DistributionType.NORMAL,
            parameters={"mu": 0.0, "sigma": 1.0}
        )
        prob_engine.create_distribution("normal", config)
        samples = prob_engine.sample("normal", 1000)
        
        # Analyze samples
        summary = stats_engine.compute_summary_statistics(samples)
        assert summary.count == 1000
        assert summary.mean == pytest.approx(0.0, abs=0.1)
        assert summary.std == pytest.approx(1.0, abs=0.1)
    
    def test_full_simulation_workflow(self):
        """Test complete simulation workflow."""
        # This would test the full workflow once we implement
        # the concrete game model and simulator classes
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])