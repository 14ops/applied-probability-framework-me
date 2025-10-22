"""
Abstract base classes and interfaces for the Applied Probability Framework.

This module defines the core abstractions that enable extensibility and
modularity in the framework. All game models, strategies, and simulators
should inherit from these base classes.

Author: Shihab Belal
License: MIT
Version: 2.0.0
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Protocol, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime

# Type variables for generic classes
T = TypeVar('T')
GameState = TypeVar('GameState')
Action = TypeVar('Action')
Reward = TypeVar('Reward', bound=Union[int, float])


class SimulationStatus(Enum):
    """Status enumeration for simulation states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SimulationConfig:
    """
    Configuration class for simulation parameters.
    
    This class encapsulates all configurable parameters for running
    simulations, providing type safety and validation.
    
    Attributes:
        num_simulations: Number of simulation runs to execute
        max_iterations: Maximum iterations per simulation
        random_seed: Seed for random number generation (None for random)
        parallel: Whether to use parallel processing
        num_workers: Number of worker processes for parallel execution
        timeout: Maximum time allowed for simulation (seconds)
        output_level: Level of output verbosity
        save_intermediate: Whether to save intermediate results
        output_file: Path to save results (optional)
    """
    num_simulations: int = 1000
    max_iterations: int = 100
    random_seed: Optional[int] = None
    parallel: bool = False
    num_workers: int = 4
    timeout: Optional[float] = None
    output_level: int = 1
    save_intermediate: bool = False
    output_file: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass
class SimulationResult:
    """
    Container for simulation results and metadata.
    
    This class provides a structured way to store and access simulation
    results, including statistical measures and metadata.
    
    Attributes:
        results: Raw simulation results
        statistics: Computed statistical measures
        config: Configuration used for simulation
        execution_time: Time taken to complete simulation
        status: Final status of simulation
        metadata: Additional metadata about the simulation
        timestamp: When the simulation was completed
    """
    results: List[Any] = field(default_factory=list)
    statistics: Dict[str, float] = field(default_factory=dict)
    config: SimulationConfig = field(default_factory=SimulationConfig)
    execution_time: float = 0.0
    status: SimulationStatus = SimulationStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_result(self, result: Any) -> None:
        """Add a single result to the results list."""
        self.results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the simulation results."""
        return {
            'num_results': len(self.results),
            'execution_time': self.execution_time,
            'status': self.status.value,
            'statistics': self.statistics,
            'timestamp': self.timestamp.isoformat()
        }


class BaseGameModel(ABC, Generic[GameState, Action, Reward]):
    """
    Abstract base class for game models.
    
    This class defines the interface that all game models must implement
    to be compatible with the framework. It provides type safety and
    ensures consistent behavior across different game implementations.
    
    Type Parameters:
        GameState: Type representing the game state
        Action: Type representing possible actions
        Reward: Type representing rewards (must be numeric)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the game model.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._state: Optional[GameState] = None
        self._is_initialized = False
    
    @abstractmethod
    def initialize(self) -> GameState:
        """
        Initialize the game state.
        
        Returns:
            Initial game state
            
        Raises:
            GameInitializationError: If initialization fails
        """
        pass
    
    @abstractmethod
    def get_available_actions(self, state: GameState) -> List[Action]:
        """
        Get available actions for a given state.
        
        Args:
            state: Current game state
            
        Returns:
            List of available actions
        """
        pass
    
    @abstractmethod
    def apply_action(self, state: GameState, action: Action) -> tuple[GameState, Reward, bool]:
        """
        Apply an action to the current state.
        
        Args:
            state: Current game state
            action: Action to apply
            
        Returns:
            Tuple of (new_state, reward, is_terminal)
        """
        pass
    
    @abstractmethod
    def is_terminal(self, state: GameState) -> bool:
        """
        Check if the current state is terminal.
        
        Args:
            state: Game state to check
            
        Returns:
            True if state is terminal, False otherwise
        """
        pass
    
    @abstractmethod
    def get_reward(self, state: GameState) -> Reward:
        """
        Get the reward for a given state.
        
        Args:
            state: Game state
            
        Returns:
            Reward value
        """
        pass
    
    def reset(self) -> GameState:
        """
        Reset the game to initial state.
        
        Returns:
            Initial game state
        """
        self._state = self.initialize()
        self._is_initialized = True
        return self._state
    
    @property
    def state(self) -> Optional[GameState]:
        """Get current game state."""
        return self._state
    
    @property
    def is_initialized(self) -> bool:
        """Check if game is initialized."""
        return self._is_initialized


class BaseStrategy(ABC, Generic[GameState, Action]):
    """
    Abstract base class for game strategies.
    
    This class defines the interface that all strategies must implement
    to be compatible with the framework. Strategies can be simple rule-based
    approaches or complex AI algorithms.
    
    Type Parameters:
        GameState: Type representing the game state
        Action: Type representing possible actions
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the strategy.
        
        Args:
            name: Human-readable name for the strategy
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._performance_metrics: Dict[str, float] = {}
    
    @abstractmethod
    def select_action(self, state: GameState, available_actions: List[Action]) -> Action:
        """
        Select an action based on the current state.
        
        Args:
            state: Current game state
            available_actions: List of available actions
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, state: GameState, action: Action, reward: float, 
               next_state: GameState) -> None:
        """
        Update strategy based on experience.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
        """
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for this strategy.
        
        Returns:
            Dictionary of performance metrics
        """
        return self._performance_metrics.copy()
    
    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Update performance metrics.
        
        Args:
            metrics: Dictionary of metric updates
        """
        self._performance_metrics.update(metrics)
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics to zero."""
        self._performance_metrics.clear()


class BaseSimulator(ABC, Generic[GameState, Action, Reward]):
    """
    Abstract base class for simulation engines.
    
    This class defines the interface for simulation engines that can
    run Monte Carlo simulations with different game models and strategies.
    
    Type Parameters:
        GameState: Type representing the game state
        Action: Type representing possible actions
        Reward: Type representing rewards
    """
    
    def __init__(self, config: SimulationConfig) -> None:
        """
        Initialize the simulator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self._results: List[SimulationResult] = []
    
    @abstractmethod
    def run_simulation(self, game_model: BaseGameModel[GameState, Action, Reward],
                      strategy: BaseStrategy[GameState, Action]) -> SimulationResult:
        """
        Run a single simulation.
        
        Args:
            game_model: Game model to simulate
            strategy: Strategy to use
            
        Returns:
            Simulation result
        """
        pass
    
    @abstractmethod
    def run_batch_simulations(self, game_model: BaseGameModel[GameState, Action, Reward],
                             strategy: BaseStrategy[GameState, Action]) -> SimulationResult:
        """
        Run multiple simulations.
        
        Args:
            game_model: Game model to simulate
            strategy: Strategy to use
            
        Returns:
            Aggregated simulation results
        """
        pass
    
    def get_results(self) -> List[SimulationResult]:
        """
        Get all simulation results.
        
        Returns:
            List of simulation results
        """
        return self._results.copy()
    
    def clear_results(self) -> None:
        """Clear all stored results."""
        self._results.clear()


class GameInitializationError(Exception):
    """Exception raised when game initialization fails."""
    pass


class SimulationError(Exception):
    """Exception raised when simulation fails."""
    pass


class StrategyError(Exception):
    """Exception raised when strategy execution fails."""
    pass