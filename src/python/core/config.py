"""
Configuration system for the Applied Probability Framework.

This module provides dataclass-based configuration with validation,
serialization, and type safety. Uses Python 3.7+ dataclasses for
clean configuration management without external dependencies.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List
import json
from pathlib import Path


@dataclass
class EstimatorConfig:
    """
    Configuration for probability estimators.
    
    Attributes:
        estimator_type: Type of estimator ('beta', 'gaussian', 'dirichlet')
        alpha: First shape parameter (for Beta) or prior mean
        beta: Second shape parameter (for Beta) or prior variance
        confidence_level: Confidence level for intervals (0.0 to 1.0)
        name: Optional name for the estimator
    """
    estimator_type: str = "beta"
    alpha: float = 1.0
    beta: float = 1.0
    confidence_level: float = 0.95
    name: str = "default_estimator"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if not 0 < self.confidence_level < 1:
            raise ValueError(f"confidence_level must be in (0,1), got {self.confidence_level}")
        if self.estimator_type not in ['beta', 'gaussian', 'dirichlet', 'mle']:
            raise ValueError(f"Unknown estimator_type: {self.estimator_type}")


@dataclass
class StrategyConfig:
    """
    Configuration for decision-making strategies.
    
    Attributes:
        strategy_type: Type of strategy ('basic', 'kelly', 'conservative', etc.)
        risk_tolerance: Risk tolerance parameter (0.0 = risk-averse, 1.0 = risk-neutral)
        kelly_fraction: Fraction of Kelly bet to use (0.0 to 1.0)
        min_confidence_threshold: Minimum confidence before taking action
        estimator_config: Configuration for the strategy's estimator
        name: Strategy name
        params: Additional strategy-specific parameters
    """
    strategy_type: str = "basic"
    risk_tolerance: float = 0.5
    kelly_fraction: float = 0.25
    min_confidence_threshold: float = 0.1
    estimator_config: EstimatorConfig = field(default_factory=EstimatorConfig)
    name: str = "default_strategy"
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 <= self.risk_tolerance <= 1:
            raise ValueError(f"risk_tolerance must be in [0,1], got {self.risk_tolerance}")
        if not 0 < self.kelly_fraction <= 1:
            raise ValueError(f"kelly_fraction must be in (0,1], got {self.kelly_fraction}")
        if not 0 <= self.min_confidence_threshold <= 1:
            raise ValueError(f"min_confidence_threshold must be in [0,1], got {self.min_confidence_threshold}")


@dataclass
class SimulationConfig:
    """
    Configuration for Monte Carlo simulations.
    
    Attributes:
        num_simulations: Number of Monte Carlo runs
        num_episodes_per_sim: Episodes per simulation run
        seed: Random seed for reproducibility
        parallel: Whether to run simulations in parallel
        n_jobs: Number of parallel jobs (-1 for all cores)
        convergence_check: Whether to check for convergence
        convergence_window: Window size for convergence detection
        convergence_tolerance: Tolerance for declaring convergence
        save_results: Whether to save results to disk
        output_dir: Directory for output files
        log_level: Logging verbosity ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    num_simulations: int = 10000
    num_episodes_per_sim: int = 1
    seed: Optional[int] = None
    parallel: bool = False
    n_jobs: int = -1
    convergence_check: bool = True
    convergence_window: int = 100
    convergence_tolerance: float = 0.001
    save_results: bool = True
    output_dir: str = "results"
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations must be positive, got {self.num_simulations}")
        if self.num_episodes_per_sim < 1:
            raise ValueError(f"num_episodes_per_sim must be positive, got {self.num_episodes_per_sim}")
        if self.convergence_window < 2:
            raise ValueError(f"convergence_window must be >= 2, got {self.convergence_window}")
        if self.convergence_tolerance <= 0:
            raise ValueError(f"convergence_tolerance must be positive, got {self.convergence_tolerance}")
        if self.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            raise ValueError(f"Invalid log_level: {self.log_level}")


@dataclass
class GameConfig:
    """
    Configuration for game-specific parameters.
    
    Attributes:
        game_type: Type of game ('mines', 'custom')
        board_size: Size of game board (for grid games)
        mine_count: Number of mines (for Mines game)
        initial_bankroll: Starting bankroll
        initial_bet_size: Initial bet size
        params: Additional game-specific parameters
    """
    game_type: str = "mines"
    board_size: int = 5
    mine_count: int = 3
    initial_bankroll: float = 1000.0
    initial_bet_size: float = 10.0
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.board_size < 1:
            raise ValueError(f"board_size must be positive, got {self.board_size}")
        if self.mine_count < 0:
            raise ValueError(f"mine_count must be non-negative, got {self.mine_count}")
        if self.mine_count >= self.board_size ** 2:
            raise ValueError(f"mine_count ({self.mine_count}) must be less than total cells ({self.board_size**2})")
        if self.initial_bankroll <= 0:
            raise ValueError(f"initial_bankroll must be positive, got {self.initial_bankroll}")
        if self.initial_bet_size <= 0:
            raise ValueError(f"initial_bet_size must be positive, got {self.initial_bet_size}")


@dataclass
class FrameworkConfig:
    """
    Master configuration for the entire framework.
    
    This combines all sub-configurations into a single validated config object.
    
    Attributes:
        simulation: Simulation parameters
        game: Game parameters
        strategies: List of strategy configurations
        meta_strategy: Configuration for meta-controller (if used)
    """
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    game: GameConfig = field(default_factory=GameConfig)
    strategies: List[StrategyConfig] = field(default_factory=list)
    meta_strategy: Optional[StrategyConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)
    
    def to_json(self, filepath: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FrameworkConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
            
        Returns:
            FrameworkConfig instance
        """
        # Handle nested configs
        sim_config = SimulationConfig(**config_dict.get('simulation', {}))
        game_config = GameConfig(**config_dict.get('game', {}))
        
        strategies = []
        for strat_dict in config_dict.get('strategies', []):
            est_dict = strat_dict.pop('estimator_config', {})
            est_config = EstimatorConfig(**est_dict)
            strat_dict['estimator_config'] = est_config
            strategies.append(StrategyConfig(**strat_dict))
        
        meta_strategy = None
        if 'meta_strategy' in config_dict and config_dict['meta_strategy']:
            meta_dict = config_dict['meta_strategy']
            est_dict = meta_dict.pop('estimator_config', {})
            est_config = EstimatorConfig(**est_dict)
            meta_dict['estimator_config'] = est_config
            meta_strategy = StrategyConfig(**meta_dict)
        
        return cls(
            simulation=sim_config,
            game=game_config,
            strategies=strategies,
            meta_strategy=meta_strategy
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> 'FrameworkConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            FrameworkConfig instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def validate(self) -> bool:
        """
        Validate entire configuration.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Dataclass __post_init__ already validates individual configs
        # Additional cross-validation can go here
        
        if self.strategies:
            if len(self.strategies) != len(set(s.name for s in self.strategies)):
                raise ValueError("Strategy names must be unique")
        
        return True


def load_config(filepath: str) -> FrameworkConfig:
    """
    Load framework configuration from file.
    
    Args:
        filepath: Path to configuration file (JSON)
        
    Returns:
        Validated FrameworkConfig instance
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If configuration is invalid
    """
    return FrameworkConfig.from_json(filepath)


def create_default_config() -> FrameworkConfig:
    """
    Create a default configuration.
    
    Returns:
        FrameworkConfig with sensible defaults
    """
    return FrameworkConfig(
        simulation=SimulationConfig(
            num_simulations=1000,
            seed=42,
            parallel=True
        ),
        game=GameConfig(
            board_size=5,
            mine_count=3,
            initial_bankroll=1000.0
        ),
        strategies=[
            StrategyConfig(
                name="conservative",
                strategy_type="kelly",
                kelly_fraction=0.1,
                risk_tolerance=0.3
            ),
            StrategyConfig(
                name="balanced",
                strategy_type="kelly",
                kelly_fraction=0.25,
                risk_tolerance=0.5
            ),
            StrategyConfig(
                name="aggressive",
                strategy_type="kelly",
                kelly_fraction=0.5,
                risk_tolerance=0.8
            ),
        ]
    )

