"""
Plugin system for extensible framework architecture.

This module implements a plugin discovery and loading system that allows users
to register custom simulators, strategies, and estimators without modifying
core framework code.
"""

from typing import Dict, Type, List, Any, Callable
import importlib
import inspect
from pathlib import Path
import sys

from .base_simulator import BaseSimulator
from .base_strategy import BaseStrategy
from .base_estimator import BaseEstimator


class PluginRegistry:
    """
    Central registry for plugins (simulators, strategies, estimators).
    
    This implements the Registry pattern to allow dynamic discovery and
    instantiation of plugins. Users can register plugins programmatically
    or through automatic discovery.
    
    Attributes:
        simulators: Registered simulator classes
        strategies: Registered strategy classes
        estimators: Registered estimator classes
    """
    
    def __init__(self):
        """Initialize empty registries."""
        self._simulators: Dict[str, Type[BaseSimulator]] = {}
        self._strategies: Dict[str, Type[BaseStrategy]] = {}
        self._estimators: Dict[str, Type[BaseEstimator]] = {}
        self._factories: Dict[str, Callable] = {}
        
    def register_simulator(self, name: str, simulator_class: Type[BaseSimulator]) -> None:
        """
        Register a simulator class.
        
        Args:
            name: Unique identifier for the simulator
            simulator_class: Class inheriting from BaseSimulator
            
        Raises:
            TypeError: If class doesn't inherit from BaseSimulator
            ValueError: If name already registered
        """
        if not issubclass(simulator_class, BaseSimulator):
            raise TypeError(f"{simulator_class} must inherit from BaseSimulator")
        if name in self._simulators:
            raise ValueError(f"Simulator '{name}' already registered")
        self._simulators[name] = simulator_class
        
    def register_strategy(self, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a strategy class.
        
        Args:
            name: Unique identifier for the strategy
            strategy_class: Class inheriting from BaseStrategy
            
        Raises:
            TypeError: If class doesn't inherit from BaseStrategy
            ValueError: If name already registered
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError(f"{strategy_class} must inherit from BaseStrategy")
        if name in self._strategies:
            raise ValueError(f"Strategy '{name}' already registered")
        self._strategies[name] = strategy_class
        
    def register_estimator(self, name: str, estimator_class: Type[BaseEstimator]) -> None:
        """
        Register an estimator class.
        
        Args:
            name: Unique identifier for the estimator
            estimator_class: Class inheriting from BaseEstimator
            
        Raises:
            TypeError: If class doesn't inherit from BaseEstimator
            ValueError: If name already registered
        """
        if not issubclass(estimator_class, BaseEstimator):
            raise TypeError(f"{estimator_class} must inherit from BaseEstimator")
        if name in self._estimators:
            raise ValueError(f"Estimator '{name}' already registered")
        self._estimators[name] = estimator_class
        
    def register_factory(self, name: str, factory_func: Callable) -> None:
        """
        Register a factory function for complex object creation.
        
        Args:
            name: Unique identifier for the factory
            factory_func: Callable that returns configured instances
        """
        if name in self._factories:
            raise ValueError(f"Factory '{name}' already registered")
        self._factories[name] = factory_func
        
    def get_simulator(self, name: str) -> Type[BaseSimulator]:
        """
        Get registered simulator class by name.
        
        Args:
            name: Simulator identifier
            
        Returns:
            Simulator class
            
        Raises:
            KeyError: If simulator not found
        """
        if name not in self._simulators:
            raise KeyError(f"Simulator '{name}' not found. Available: {list(self._simulators.keys())}")
        return self._simulators[name]
    
    def get_strategy(self, name: str) -> Type[BaseStrategy]:
        """
        Get registered strategy class by name.
        
        Args:
            name: Strategy identifier
            
        Returns:
            Strategy class
            
        Raises:
            KeyError: If strategy not found
        """
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' not found. Available: {list(self._strategies.keys())}")
        return self._strategies[name]
    
    def get_estimator(self, name: str) -> Type[BaseEstimator]:
        """
        Get registered estimator class by name.
        
        Args:
            name: Estimator identifier
            
        Returns:
            Estimator class
            
        Raises:
            KeyError: If estimator not found
        """
        if name not in self._estimators:
            raise KeyError(f"Estimator '{name}' not found. Available: {list(self._estimators.keys())}")
        return self._estimators[name]
    
    def get_factory(self, name: str) -> Callable:
        """
        Get registered factory function.
        
        Args:
            name: Factory identifier
            
        Returns:
            Factory function
        """
        if name not in self._factories:
            raise KeyError(f"Factory '{name}' not found. Available: {list(self._factories.keys())}")
        return self._factories[name]
    
    def list_simulators(self) -> List[str]:
        """List all registered simulator names."""
        return list(self._simulators.keys())
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy names."""
        return list(self._strategies.keys())
    
    def list_estimators(self) -> List[str]:
        """List all registered estimator names."""
        return list(self._estimators.keys())
    
    def discover_plugins(self, plugin_dir: str) -> int:
        """
        Automatically discover and register plugins from a directory.
        
        Searches for Python modules in plugin_dir and registers any classes
        that inherit from base classes and have a 'register' attribute set to True.
        
        Args:
            plugin_dir: Path to directory containing plugin modules
            
        Returns:
            Number of plugins registered
            
        Example plugin structure:
            # my_plugin.py
            from core.base_strategy import BaseStrategy
            
            class MyStrategy(BaseStrategy):
                register = True  # Enable auto-discovery
                plugin_name = "my_strategy"  # Optional custom name
                
                def select_action(self, state, valid_actions):
                    ...
        """
        plugin_path = Path(plugin_dir)
        if not plugin_path.exists():
            return 0
        
        # Add plugin directory to sys.path temporarily
        sys.path.insert(0, str(plugin_path.parent))
        
        count = 0
        try:
            for py_file in plugin_path.glob("*.py"):
                if py_file.stem.startswith("_"):
                    continue
                    
                try:
                    # Import the module
                    module_name = py_file.stem
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find classes with register=True
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if not getattr(obj, 'register', False):
                            continue
                        
                        plugin_name = getattr(obj, 'plugin_name', name.lower())
                        
                        # Register based on base class
                        if issubclass(obj, BaseSimulator) and obj != BaseSimulator:
                            self.register_simulator(plugin_name, obj)
                            count += 1
                        elif issubclass(obj, BaseStrategy) and obj != BaseStrategy:
                            self.register_strategy(plugin_name, obj)
                            count += 1
                        elif issubclass(obj, BaseEstimator) and obj != BaseEstimator:
                            self.register_estimator(plugin_name, obj)
                            count += 1
                            
                except Exception as e:
                    print(f"Warning: Failed to load plugin from {py_file}: {e}")
                    
        finally:
            # Remove from sys.path
            sys.path.pop(0)
        
        return count
    
    def create_simulator(self, name: str, **kwargs) -> BaseSimulator:
        """
        Create simulator instance by name.
        
        Args:
            name: Simulator identifier
            **kwargs: Arguments passed to simulator constructor
            
        Returns:
            Configured simulator instance
        """
        simulator_class = self.get_simulator(name)
        return simulator_class(**kwargs)
    
    def create_strategy(self, name: str, **kwargs) -> BaseStrategy:
        """
        Create strategy instance by name.
        
        Args:
            name: Strategy identifier
            **kwargs: Arguments passed to strategy constructor
            
        Returns:
            Configured strategy instance
        """
        strategy_class = self.get_strategy(name)
        return strategy_class(**kwargs)
    
    def create_estimator(self, name: str, **kwargs) -> BaseEstimator:
        """
        Create estimator instance by name.
        
        Args:
            name: Estimator identifier
            **kwargs: Arguments passed to estimator constructor
            
        Returns:
            Configured estimator instance
        """
        estimator_class = self.get_estimator(name)
        return estimator_class(**kwargs)


# Global registry instance
_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """
    Get the global plugin registry.
    
    Returns:
        Global PluginRegistry instance
    """
    return _registry


def register_simulator(name: str):
    """
    Decorator to register a simulator class.
    
    Example:
        @register_simulator("my_sim")
        class MySimulator(BaseSimulator):
            ...
    """
    def decorator(cls):
        _registry.register_simulator(name, cls)
        return cls
    return decorator


def register_strategy(name: str):
    """
    Decorator to register a strategy class.
    
    Example:
        @register_strategy("my_strategy")
        class MyStrategy(BaseStrategy):
            ...
    """
    def decorator(cls):
        _registry.register_strategy(name, cls)
        return cls
    return decorator


def register_estimator(name: str):
    """
    Decorator to register an estimator class.
    
    Example:
        @register_estimator("my_estimator")
        class MyEstimator(BaseEstimator):
            ...
    """
    def decorator(cls):
        _registry.register_estimator(name, cls)
        return cls
    return decorator

