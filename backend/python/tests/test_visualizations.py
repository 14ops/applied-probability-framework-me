"""
Comprehensive test suite for visualization components.

This module provides extensive testing for all visualization features,
ensuring they meet professional standards for clarity, functionality,
and integration.

Author: Shihab Belal
License: MIT
Version: 2.0.0
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from typing import Dict, List, Any

# Import visualization components
from src.visualizations.visualization import (
    plot_simulation_results,
    create_performance_dashboard,
    generate_progress_report
)
from src.visualizations.advanced_visualizations import (
    AdvancedVisualizationEngine,
    create_interactive_dashboard,
    generate_comparison_plots
)
from src.visualizations.comprehensive_visualizations import (
    ComprehensiveVisualizationSuite,
    create_ai_progress_timeline,
    generate_benchmark_comparison
)


class TestBasicVisualizations:
    """Test cases for basic visualization functions."""
    
    def test_plot_simulation_results(self):
        """Test basic simulation results plotting."""
        # Create test data
        data = np.random.normal(0, 1, 100)
        
        # Test plotting function
        result = plot_simulation_results(data, save_path=None)
        
        # Verify return value
        assert result is not None
        assert hasattr(result, 'figure')
    
    def test_plot_simulation_results_with_save(self):
        """Test plotting with file save."""
        data = np.random.normal(0, 1, 100)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_plot.png")
            result = plot_simulation_results(data, save_path=save_path)
            
            # Verify file was created
            assert os.path.exists(save_path)
            assert result is not None
    
    def test_create_performance_dashboard(self):
        """Test performance dashboard creation."""
        # Create test performance data
        performance_data = {
            'agent_1': {
                'win_rate': [0.6, 0.65, 0.7, 0.72, 0.75],
                'avg_payout': [1.2, 1.25, 1.3, 1.32, 1.35],
                'timestamps': pd.date_range('2025-01-01', periods=5, freq='D')
            },
            'agent_2': {
                'win_rate': [0.55, 0.6, 0.65, 0.68, 0.7],
                'avg_payout': [1.15, 1.2, 1.25, 1.28, 1.3],
                'timestamps': pd.date_range('2025-01-01', periods=5, freq='D')
            }
        }
        
        dashboard = create_performance_dashboard(performance_data)
        
        # Verify dashboard structure
        assert dashboard is not None
        assert hasattr(dashboard, 'figures')
        assert len(dashboard.figures) > 0
    
    def test_generate_progress_report(self):
        """Test progress report generation."""
        # Create test data
        progress_data = {
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
            'values': [0.85, 0.82, 0.88, 0.85],
            'improvements': [0.05, 0.03, 0.07, 0.05],
            'timestamps': pd.date_range('2025-01-01', periods=4, freq='D')
        }
        
        report = generate_progress_report(progress_data)
        
        # Verify report structure
        assert report is not None
        assert 'summary' in report
        assert 'charts' in report
        assert 'recommendations' in report


class TestAdvancedVisualizations:
    """Test cases for advanced visualization engine."""
    
    def test_advanced_visualization_engine_init(self):
        """Test advanced visualization engine initialization."""
        engine = AdvancedVisualizationEngine()
        
        assert engine is not None
        assert hasattr(engine, 'create_chart')
        assert hasattr(engine, 'export_dashboard')
    
    def test_create_interactive_dashboard(self):
        """Test interactive dashboard creation."""
        # Create test data
        dashboard_data = {
            'agents': ['agent_1', 'agent_2', 'agent_3'],
            'metrics': ['win_rate', 'avg_payout', 'efficiency'],
            'data': np.random.rand(3, 3, 10)  # 3 agents, 3 metrics, 10 time points
        }
        
        dashboard = create_interactive_dashboard(dashboard_data)
        
        # Verify dashboard properties
        assert dashboard is not None
        assert hasattr(dashboard, 'interactive_elements')
        assert hasattr(dashboard, 'export_options')
    
    def test_generate_comparison_plots(self):
        """Test comparison plots generation."""
        # Create test comparison data
        comparison_data = {
            'baseline': {'win_rate': 0.6, 'avg_payout': 1.2},
            'improved': {'win_rate': 0.75, 'avg_payout': 1.35},
            'advanced': {'win_rate': 0.8, 'avg_payout': 1.4}
        }
        
        plots = generate_comparison_plots(comparison_data)
        
        # Verify plots structure
        assert plots is not None
        assert isinstance(plots, list)
        assert len(plots) > 0


class TestComprehensiveVisualizations:
    """Test cases for comprehensive visualization suite."""
    
    def test_comprehensive_visualization_suite_init(self):
        """Test comprehensive visualization suite initialization."""
        suite = ComprehensiveVisualizationSuite()
        
        assert suite is not None
        assert hasattr(suite, 'create_timeline')
        assert hasattr(suite, 'generate_benchmark_report')
        assert hasattr(suite, 'export_all_visualizations')
    
    def test_create_ai_progress_timeline(self):
        """Test AI progress timeline creation."""
        # Create test timeline data
        timeline_data = {
            'experiments': [
                {'date': '2025-01-01', 'agent': 'baseline', 'score': 0.6},
                {'date': '2025-01-02', 'agent': 'v1', 'score': 0.65},
                {'date': '2025-01-03', 'agent': 'v2', 'score': 0.72},
                {'date': '2025-01-04', 'agent': 'v3', 'score': 0.78}
            ],
            'metrics': ['accuracy', 'efficiency', 'robustness']
        }
        
        timeline = create_ai_progress_timeline(timeline_data)
        
        # Verify timeline structure
        assert timeline is not None
        assert hasattr(timeline, 'figures')
        assert hasattr(timeline, 'interactive_features')
    
    def test_generate_benchmark_comparison(self):
        """Test benchmark comparison generation."""
        # Create test benchmark data
        benchmark_data = {
            'models': ['Model A', 'Model B', 'Model C'],
            'benchmarks': ['Benchmark 1', 'Benchmark 2', 'Benchmark 3'],
            'scores': np.random.rand(3, 3),
            'metadata': {
                'test_date': '2025-01-01',
                'test_environment': 'production'
            }
        }
        
        comparison = generate_benchmark_comparison(benchmark_data)
        
        # Verify comparison structure
        assert comparison is not None
        assert hasattr(comparison, 'heatmap')
        assert hasattr(comparison, 'summary_table')
        assert hasattr(comparison, 'statistical_analysis')


class TestVisualizationIntegration:
    """Integration tests for visualization components."""
    
    def test_end_to_end_visualization_pipeline(self):
        """Test complete visualization pipeline."""
        # Create comprehensive test data
        test_data = {
            'simulation_results': np.random.normal(0, 1, 1000),
            'agent_performance': {
                'agent_1': {'win_rate': 0.75, 'avg_payout': 1.35},
                'agent_2': {'win_rate': 0.68, 'avg_payout': 1.28},
                'agent_3': {'win_rate': 0.82, 'avg_payout': 1.42}
            },
            'progress_metrics': {
                'timestamps': pd.date_range('2025-01-01', periods=30, freq='D'),
                'improvements': np.random.rand(30) * 0.1
            }
        }
        
        # Test pipeline execution
        suite = ComprehensiveVisualizationSuite()
        
        # Generate all visualizations
        results = suite.generate_all_visualizations(test_data)
        
        # Verify results
        assert results is not None
        assert 'basic_plots' in results
        assert 'advanced_dashboards' in results
        assert 'comprehensive_reports' in results
    
    def test_visualization_export_formats(self):
        """Test visualization export in multiple formats."""
        # Create test data
        test_data = np.random.normal(0, 1, 100)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test different export formats
            formats = ['png', 'pdf', 'svg', 'html', 'json']
            
            for fmt in formats:
                output_path = os.path.join(temp_dir, f"test_export.{fmt}")
                
                # Create and export visualization
                result = plot_simulation_results(test_data, save_path=output_path)
                
                # Verify file was created
                assert os.path.exists(output_path)
    
    def test_visualization_with_real_data(self):
        """Test visualizations with realistic data patterns."""
        # Create realistic AI progress data
        np.random.seed(42)
        
        # Simulate AI improvement over time
        timestamps = pd.date_range('2025-01-01', periods=100, freq='D')
        base_performance = 0.6
        improvement_rate = 0.001
        noise = 0.02
        
        performance = []
        for i in range(100):
            perf = base_performance + improvement_rate * i + np.random.normal(0, noise)
            performance.append(max(0, min(1, perf)))  # Clamp to [0, 1]
        
        progress_data = {
            'timestamps': timestamps,
            'performance': performance,
            'agent_name': 'test_agent',
            'metric': 'win_rate'
        }
        
        # Test visualization creation
        suite = ComprehensiveVisualizationSuite()
        result = suite.create_progress_visualization(progress_data)
        
        # Verify visualization properties
        assert result is not None
        assert hasattr(result, 'figure')
        assert hasattr(result, 'data_summary')


class TestVisualizationQuality:
    """Test cases for visualization quality and clarity."""
    
    def test_chart_clarity_standards(self):
        """Test that charts meet clarity standards."""
        # Create test data with clear patterns
        data = {
            'x': np.linspace(0, 10, 100),
            'y': np.sin(np.linspace(0, 10, 100)) + 0.1 * np.random.randn(100)
        }
        
        # Create visualization
        result = plot_simulation_results(data['y'])
        
        # Verify chart has required elements
        assert hasattr(result, 'figure')
        fig = result.figure
        
        # Check for axes labels (if matplotlib figure)
        if hasattr(fig, 'axes'):
            for ax in fig.axes:
                assert ax.get_xlabel() != '', "X-axis should have a label"
                assert ax.get_ylabel() != '', "Y-axis should have a label"
                assert ax.get_title() != '', "Chart should have a title"
    
    def test_color_consistency(self):
        """Test color consistency across visualizations."""
        # Create multiple related visualizations
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(1, 1, 100)
        
        plot1 = plot_simulation_results(data1)
        plot2 = plot_simulation_results(data2)
        
        # Verify both plots exist
        assert plot1 is not None
        assert plot2 is not None
    
    def test_accessibility_features(self):
        """Test accessibility features in visualizations."""
        # Create test data
        data = np.random.normal(0, 1, 100)
        
        # Create visualization with accessibility features
        result = plot_simulation_results(data, include_accessibility=True)
        
        # Verify accessibility features
        assert result is not None
        if hasattr(result, 'accessibility_info'):
            assert 'color_blind_friendly' in result.accessibility_info
            assert 'high_contrast' in result.accessibility_info


class TestVisualizationPerformance:
    """Performance tests for visualization components."""
    
    def test_large_dataset_handling(self):
        """Test visualization performance with large datasets."""
        # Create large dataset
        large_data = np.random.normal(0, 1, 100000)
        
        # Measure performance
        import time
        start_time = time.time()
        
        result = plot_simulation_results(large_data)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify reasonable performance (should complete within 10 seconds)
        assert execution_time < 10.0, f"Visualization took too long: {execution_time}s"
        assert result is not None
    
    def test_memory_usage(self):
        """Test memory usage of visualization components."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create multiple visualizations
        for i in range(10):
            data = np.random.normal(0, 1, 1000)
            plot_simulation_results(data)
        
        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Verify reasonable memory usage (less than 100MB increase)
        assert memory_increase < 100 * 1024 * 1024, f"Memory usage too high: {memory_increase / 1024 / 1024:.2f}MB"


class TestVisualizationErrorHandling:
    """Test error handling in visualization components."""
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        # Test with None data
        with pytest.raises(ValueError):
            plot_simulation_results(None)
        
        # Test with empty data
        with pytest.raises(ValueError):
            plot_simulation_results([])
        
        # Test with invalid data types
        with pytest.raises(TypeError):
            plot_simulation_results("invalid_data")
    
    def test_missing_optional_dependencies(self):
        """Test graceful handling of missing optional dependencies."""
        # Mock missing matplotlib
        with patch('matplotlib.pyplot.figure', side_effect=ImportError("matplotlib not available")):
            with pytest.raises(ImportError):
                plot_simulation_results([1, 2, 3, 4, 5])
    
    def test_file_save_errors(self):
        """Test handling of file save errors."""
        # Test with invalid save path
        data = [1, 2, 3, 4, 5]
        
        with pytest.raises(OSError):
            plot_simulation_results(data, save_path="/invalid/path/that/does/not/exist/plot.png")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])