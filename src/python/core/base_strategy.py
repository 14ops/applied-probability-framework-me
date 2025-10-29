"""
Abstract base class for decision-making strategies.

This module defines the interface for strategies that can be plugged into
the framework to make decisions within simulations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class BaseStrategy(ABC):
    """
    Abstract base class for all strategies.
    
    A strategy encapsulates the decision-making logic for a simulation.
    It observes the state and decides what action to take. Strategies can be:
    - Rule-based (heuristic)
    - Learning-based (using estimators/models)
    - Optimal (derived from theory like Kelly Criterion)
    - Adaptive (meta-strategies that switch between sub-strategies)
    
    Attributes:
        name: Human-readable name for the strategy
        config: Configuration dictionary for strategy parameters
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Name identifier for this strategy
            config: Configuration parameters specific to this strategy
        """
        self.name = name
        self.config = config or {}
        self._statistics: Dict[str, Any] = {
            'games_played': 0,
            'total_reward': 0.0,
            'wins': 0,
            'losses': 0,
        }
        
    @abstractmethod
    def select_action(self, state: Any, valid_actions: Any, 
                     uncertainty_vector: Optional[List[Tuple[float, float]]] = None) -> Any:
        """
        Select an action given the current state.
        
        Args:
            state: Current state of the environment
            valid_actions: List or set of valid actions in this state
            uncertainty_vector: Optional list of (mean, variance) tuples per cell
                               for Bayesian uncertainty information
            
        Returns:
            The selected action (must be in valid_actions)
            
        Raises:
            ValueError: If no valid actions are available
        """
        pass
    
    def update(self, state: Any, action: Any, reward: float, 
               next_state: Any, done: bool) -> None:
        """
        Update strategy based on observed transition (optional).
        
        This method is called after each step to allow learning strategies
        to update their internal models. Non-learning strategies can ignore this.
        
        Args:
            state: The state where action was taken
            action: The action that was taken
            reward: The reward received
            next_state: The resulting state
            done: Whether the episode terminated
        """
        # Update statistics
        self._statistics['games_played'] += 1 if done else 0
        self._statistics['total_reward'] += reward
        if done and reward > 0:
            self._statistics['wins'] += 1
        elif done and reward <= 0:
            self._statistics['losses'] += 1
    
    def reset(self) -> None:
        """
        Reset strategy state at the beginning of a new episode.
        
        This is called before each simulation run. Strategies should reset
        any episode-specific state but preserve learned parameters.
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics for this strategy.
        
        Returns:
            Dictionary containing performance metrics:
                - games_played: Total number of complete games
                - total_reward: Cumulative reward
                - wins: Number of winning games
                - losses: Number of losing games
                - win_rate: Proportion of games won
                - avg_reward: Average reward per game
        """
        stats = self._statistics.copy()
        if stats['games_played'] > 0:
            stats['win_rate'] = stats['wins'] / stats['games_played']
            stats['avg_reward'] = stats['total_reward'] / stats['games_played']
        else:
            stats['win_rate'] = 0.0
            stats['avg_reward'] = 0.0
        return stats
    
    def save(self, filepath: str) -> None:
        """
        Save strategy parameters to disk.
        
        Args:
            filepath: Path where strategy should be saved
        """
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'name': self.name,
                'config': self.config,
                'statistics': self._statistics,
            }, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """
        Load strategy parameters from disk.
        
        Args:
            filepath: Path to saved strategy file
        """
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.name = data.get('name', self.name)
            self.config = data.get('config', self.config)
            self._statistics = data.get('statistics', self._statistics)
    
    def __repr__(self) -> str:
        """String representation of the strategy."""
        stats = self.get_statistics()
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"win_rate={stats['win_rate']:.2%}, "
                f"games_played={stats['games_played']})")

