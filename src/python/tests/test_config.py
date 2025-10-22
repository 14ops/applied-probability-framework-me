"""
Unit tests for configuration system.

Tests configuration dataclasses, validation, and serialization.
"""

import pytest
import json

from core.config import (
    EstimatorConfig, StrategyConfig, SimulationConfig,
    GameConfig, FrameworkConfig, create_default_config
)


class TestEstimatorConfig:
    """Tests for EstimatorConfig."""
    
    def test_default_initialization(self):
        """Test default values."""
        config = EstimatorConfig()
        assert config.estimator_type == "beta"
        assert config.alpha == 1.0
        assert config.beta == 1.0
        assert config.confidence_level == 0.95
    
    def test_custom_initialization(self):
        """Test custom values."""
        config = EstimatorConfig(
            estimator_type="gaussian",
            alpha=2.0,
            beta=3.0,
            confidence_level=0.99
        )
        assert config.estimator_type == "gaussian"
        assert config.alpha == 2.0
        assert config.beta == 3.0
        assert config.confidence_level == 0.99
    
    def test_validation_negative_alpha(self):
        """Test that negative alpha raises error."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            EstimatorConfig(alpha=-1.0)
    
    def test_validation_invalid_confidence(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="confidence_level must be in"):
            EstimatorConfig(confidence_level=1.5)
    
    def test_validation_unknown_type(self):
        """Test that unknown estimator type raises error."""
        with pytest.raises(ValueError, match="Unknown estimator_type"):
            EstimatorConfig(estimator_type="invalid")


class TestStrategyConfig:
    """Tests for StrategyConfig."""
    
    def test_default_initialization(self):
        """Test default values."""
        config = StrategyConfig()
        assert config.strategy_type == "basic"
        assert config.risk_tolerance == 0.5
        assert config.kelly_fraction == 0.25
        assert isinstance(config.estimator_config, EstimatorConfig)
    
    def test_validation_risk_tolerance(self):
        """Test risk tolerance bounds."""
        with pytest.raises(ValueError, match="risk_tolerance must be in"):
            StrategyConfig(risk_tolerance=1.5)
    
    def test_validation_kelly_fraction(self):
        """Test kelly fraction bounds."""
        with pytest.raises(ValueError, match="kelly_fraction must be in"):
            StrategyConfig(kelly_fraction=0.0)


class TestSimulationConfig:
    """Tests for SimulationConfig."""
    
    def test_default_initialization(self):
        """Test default values."""
        config = SimulationConfig()
        assert config.num_simulations == 10000
        assert config.parallel == False
        assert config.convergence_check == True
    
    def test_validation_num_simulations(self):
        """Test num_simulations must be positive."""
        with pytest.raises(ValueError, match="num_simulations must be positive"):
            SimulationConfig(num_simulations=0)
    
    def test_validation_convergence_window(self):
        """Test convergence window must be >= 2."""
        with pytest.raises(ValueError, match="convergence_window must be >= 2"):
            SimulationConfig(convergence_window=1)


class TestGameConfig:
    """Tests for GameConfig."""
    
    def test_default_initialization(self):
        """Test default values."""
        config = GameConfig()
        assert config.game_type == "mines"
        assert config.board_size == 5
        assert config.mine_count == 3
    
    def test_validation_mine_count_vs_board_size(self):
        """Test that mine count can't exceed board size."""
        with pytest.raises(ValueError, match="mine_count.*must be less than total cells"):
            GameConfig(board_size=3, mine_count=10)
    
    def test_validation_initial_bankroll(self):
        """Test that bankroll must be positive."""
        with pytest.raises(ValueError, match="initial_bankroll must be positive"):
            GameConfig(initial_bankroll=-100.0)


class TestFrameworkConfig:
    """Tests for FrameworkConfig."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = create_default_config()
        assert isinstance(config, FrameworkConfig)
        assert isinstance(config.simulation, SimulationConfig)
        assert isinstance(config.game, GameConfig)
        assert len(config.strategies) == 3  # Conservative, balanced, aggressive
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = create_default_config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'simulation' in config_dict
        assert 'game' in config_dict
        assert 'strategies' in config_dict
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        original = create_default_config()
        config_dict = original.to_dict()
        
        reconstructed = FrameworkConfig.from_dict(config_dict)
        
        assert reconstructed.simulation.num_simulations == original.simulation.num_simulations
        assert reconstructed.game.board_size == original.game.board_size
        assert len(reconstructed.strategies) == len(original.strategies)
    
    def test_save_and_load_json(self, temp_dir):
        """Test JSON serialization."""
        config = create_default_config()
        json_path = temp_dir / "config.json"
        
        # Save
        config.to_json(str(json_path))
        assert json_path.exists()
        
        # Load
        loaded_config = FrameworkConfig.from_json(str(json_path))
        
        assert loaded_config.simulation.seed == config.simulation.seed
        assert loaded_config.game.mine_count == config.game.mine_count
    
    def test_validate_unique_strategy_names(self):
        """Test that strategy names must be unique."""
        config = FrameworkConfig(
            strategies=[
                StrategyConfig(name="strategy1"),
                StrategyConfig(name="strategy1"),  # Duplicate
            ]
        )
        
        with pytest.raises(ValueError, match="Strategy names must be unique"):
            config.validate()
    
    def test_json_roundtrip(self, temp_dir):
        """Test that config survives JSON round-trip."""
        original = FrameworkConfig(
            simulation=SimulationConfig(num_simulations=500, seed=123),
            game=GameConfig(board_size=7, mine_count=5),
            strategies=[
                StrategyConfig(
                    name="test_strategy",
                    kelly_fraction=0.3,
                    estimator_config=EstimatorConfig(alpha=2.0, beta=3.0)
                )
            ]
        )
        
        json_path = temp_dir / "roundtrip.json"
        original.to_json(str(json_path))
        loaded = FrameworkConfig.from_json(str(json_path))
        
        # Check all key values
        assert loaded.simulation.num_simulations == 500
        assert loaded.simulation.seed == 123
        assert loaded.game.board_size == 7
        assert loaded.game.mine_count == 5
        assert len(loaded.strategies) == 1
        assert loaded.strategies[0].name == "test_strategy"
        assert loaded.strategies[0].kelly_fraction == 0.3
        assert loaded.strategies[0].estimator_config.alpha == 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

