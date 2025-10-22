"""
Abstract base class for simulation environments.

This module defines the interface that all simulation environments must implement,
enabling the framework to work with arbitrary games and stochastic processes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, List
import numpy as np


class BaseSimulator(ABC):
    """
    Abstract base class for all simulation environments.
    
    This class defines the core interface for simulation environments in the framework.
    Subclasses must implement methods for initialization, state queries, and action execution.
    
    The design follows the OpenAI Gym interface pattern for broad compatibility with
    reinforcement learning and Monte Carlo simulation approaches.
    
    Attributes:
        seed: Random seed for reproducibility (optional)
        config: Configuration dictionary for the simulator
        state: Current state of the simulation
    """
    
    def __init__(self, seed: Optional[int] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the simulator.
        
        Args:
            seed: Random seed for reproducibility. If None, uses system entropy.
            config: Configuration dictionary with simulation parameters.
                   Specific parameters depend on the simulation type.
        """
        self.seed = seed
        self.config = config or {}
        self.rng = np.random.default_rng(seed)
        self.state: Optional[Any] = None
        
    @abstractmethod
    def reset(self) -> Any:
        """
        Reset the simulator to initial state.
        
        Returns:
            initial_state: The initial state after reset
            
        Note:
            This method must be called before running a new simulation episode.
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Execute one step of the simulation.
        
        Args:
            action: The action to take in the current state
            
        Returns:
            A tuple containing:
                - next_state: The resulting state after the action
                - reward: Immediate reward received
                - done: Whether the episode has terminated
                - info: Additional diagnostic information
                
        Raises:
            ValueError: If action is invalid for current state
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self) -> List[Any]:
        """
        Get list of valid actions in the current state.
        
        Returns:
            List of valid actions that can be taken
        """
        pass
    
    @abstractmethod
    def get_state_representation(self) -> np.ndarray:
        """
        Get numerical representation of current state.
        
        Returns:
            Numpy array representing the current state, suitable for
            machine learning models and statistical analysis.
        """
        pass
    
    def get_state(self) -> Any:
        """
        Get the current state of the simulation.
        
        Returns:
            The current state object
        """
        return self.state
    
    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for reproducibility.
        
        Args:
            seed: The random seed to use
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the simulator.
        
        Returns:
            Dictionary containing simulator metadata including:
                - name: Simulator name
                - version: Version string
                - state_space: Description of state space
                - action_space: Description of action space
        """
        return {
            'name': self.__class__.__name__,
            'version': '1.0.0',
            'seed': self.seed,
            'config': self.config,
        }

