"""
Integration tests for the framework.

Tests the full pipeline from configuration to simulation results.
"""

import pytest
import json

from core.config import FrameworkConfig, SimulationConfig, GameConfig, StrategyConfig
from core.parallel_engine import ParallelSimulationEngine
from core.reproducibility import ExperimentLogger, set_global_seed


class TestEndToEndSimulation:
    """End-to-end integration tests."""
    
    def test_simple_sequential_simulation(self, simple_simulator, simple_strategy, temp_dir):
        """Test running a simple sequential simulation."""
        config = SimulationConfig(
            num_simulations=10,
            seed=42,
            parallel=False,
            save_results=False
        )
        
        engine = ParallelSimulationEngine(config, n_jobs=1)
        results = engine.run_batch(simple_simulator, simple_strategy, show_progress=False)
        
        assert results.num_runs == 10
        assert results.mean_reward >= 0
        assert results.std_reward >= 0
        assert 0 <= results.success_rate <= 1
        assert results.total_duration > 0
    
    def test_parallel_simulation(self, simple_simulator, simple_strategy):
        """Test parallel simulation execution."""
        config = SimulationConfig(
            num_simulations=20,
            seed=42,
            parallel=True,
            n_jobs=2,
            save_results=False
        )
        
        engine = ParallelSimulationEngine(config, n_jobs=2)
        results = engine.run_batch(simple_simulator, simple_strategy, show_progress=False)
        
        assert results.num_runs == 20
        assert len(results.all_results) == 20
    
    def test_reproducibility_with_same_seed(self, simple_simulator, simple_strategy):
        """Test that same seed produces same results."""
        config = SimulationConfig(
            num_simulations=5,
            seed=12345,
            parallel=False,
            save_results=False
        )
        
        engine = ParallelSimulationEngine(config)
        
        results1 = engine.run_batch(simple_simulator, simple_strategy, show_progress=False)
        results2 = engine.run_batch(simple_simulator, simple_strategy, show_progress=False)
        
        # Should get same mean reward
        assert abs(results1.mean_reward - results2.mean_reward) < 1e-10
    
    def test_experiment_logging(self, simple_simulator, simple_strategy, temp_dir):
        """Test full experiment with logging."""
        logger = ExperimentLogger(str(temp_dir), "test_experiment")
        
        config = {
            'simulation': {'num_runs': 5, 'seed': 42},
            'game': {'type': 'test'}
        }
        
        logger.start_experiment(config, seed=42, tags=['test'], description="Test run")
        
        # Run simulation
        sim_config = SimulationConfig(
            num_simulations=5,
            seed=42,
            parallel=False,
            save_results=False
        )
        
        engine = ParallelSimulationEngine(sim_config)
        results = engine.run_batch(simple_simulator, simple_strategy, show_progress=False)
        
        # Log results
        results_dict = {
            'mean_reward': results.mean_reward,
            'success_rate': results.success_rate,
            'individual_runs': []
        }
        logger.log_results(results_dict)
        
        logger.end_experiment()
        
        # Verify outputs exist
        exp_dir = logger.get_experiment_dir()
        assert (exp_dir / "metadata.json").exists()
        assert (exp_dir / "config.json").exists()
        assert (exp_dir / "results.json").exists()
        assert (exp_dir / "summary.json").exists()


class TestConfigurationPipeline:
    """Test configuration loading and validation."""
    
    def test_config_to_simulation(self, temp_dir):
        """Test full pipeline from config file to simulation."""
        # Create config
        config = FrameworkConfig(
            simulation=SimulationConfig(num_simulations=5, seed=42, parallel=False),
            game=GameConfig(board_size=5, mine_count=3),
            strategies=[
                StrategyConfig(name="test_strategy")
            ]
        )
        
        # Save to file
        config_path = temp_dir / "test_config.json"
        config.to_json(str(config_path))
        
        # Load from file
        loaded_config = FrameworkConfig.from_json(str(config_path))
        
        # Validate
        assert loaded_config.validate()
        assert loaded_config.simulation.num_simulations == 5
        assert loaded_config.game.mine_count == 3


class TestReproducibility:
    """Test reproducibility features."""
    
    def test_set_global_seed(self):
        """Test that set_global_seed affects random generation."""
        import numpy as np
        
        set_global_seed(12345)
        val1 = np.random.rand()
        
        set_global_seed(12345)
        val2 = np.random.rand()
        
        assert val1 == val2
    
    def test_experiment_metadata_creation(self):
        """Test experiment metadata creation."""
        from core.reproducibility import ExperimentMetadata
        
        config = {'param1': 'value1', 'param2': 42}
        metadata = ExperimentMetadata.create(
            config=config,
            seed=123,
            tags=['test', 'integration'],
            description="Test experiment"
        )
        
        assert metadata.seed == 123
        assert metadata.tags == ['test', 'integration']
        assert metadata.description == "Test experiment"
        assert len(metadata.config_hash) == 16
        assert len(metadata.experiment_id) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

