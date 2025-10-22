"""
Pytest configuration and fixtures for the test suite.

This module provides reusable fixtures for testing simulators,
strategies, and other framework components.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from core.base_simulator import BaseSimulator
from core.base_strategy import BaseStrategy
from core.base_estimator import BaseEstimator, BetaEstimator
from core.config import (
    FrameworkConfig, SimulationConfig, GameConfig,
    StrategyConfig, EstimatorConfig, create_default_config
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def default_config():
    """Provide a default configuration."""
    return create_default_config()


@pytest.fixture
def beta_estimator():
    """Provide a Beta estimator with default parameters."""
    return BetaEstimator(alpha=1.0, beta=1.0, name="test_estimator")


@pytest.fixture
def seed():
    """Provide a fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def simple_simulator(seed):
    """
    Provide a simple test simulator.
    
    This is a minimal implementation of BaseSimulator for testing.
    """
    class SimpleSimulator(BaseSimulator):
        def __init__(self, seed=None, config=None):
            super().__init__(seed, config)
            self.steps_taken = 0
            self.max_steps = config.get('max_steps', 10) if config else 10
            
        def reset(self):
            self.state = {'position': 0, 'done': False}
            self.steps_taken = 0
            return self.state
        
        def step(self, action):
            if self.state['done']:
                return self.state, 0.0, True, {}
            
            self.steps_taken += 1
            
            if action == 'forward':
                self.state['position'] += 1
                reward = 1.0
            elif action == 'backward':
                self.state['position'] -= 1
                reward = -0.5
            else:
                reward = 0.0
            
            done = self.steps_taken >= self.max_steps
            self.state['done'] = done
            
            return self.state, reward, done, {'steps': self.steps_taken}
        
        def get_valid_actions(self):
            if self.state and not self.state['done']:
                return ['forward', 'backward', 'stay']
            return []
        
        def get_state_representation(self):
            if self.state:
                return np.array([self.state['position'], self.steps_taken])
            return np.array([0, 0])
    
    return SimpleSimulator(seed=seed)


@pytest.fixture
def simple_strategy():
    """
    Provide a simple test strategy.
    
    This always chooses the 'forward' action if available.
    """
    class SimpleStrategy(BaseStrategy):
        def __init__(self, name="simple", config=None):
            super().__init__(name, config)
        
        def select_action(self, state, valid_actions):
            if not valid_actions:
                raise ValueError("No valid actions available")
            return 'forward' if 'forward' in valid_actions else valid_actions[0]
    
    return SimpleStrategy()


@pytest.fixture
def sample_simulation_results():
    """Provide sample simulation results for testing."""
    return {
        'num_runs': 100,
        'mean_reward': 5.5,
        'std_reward': 2.3,
        'confidence_interval': (4.9, 6.1),
        'success_rate': 0.75,
        'total_duration': 12.5,
        'convergence_achieved': True,
        'individual_runs': [
            {
                'run_id': i,
                'seed': 42 + i,
                'reward': 5.0 + np.random.randn(),
                'steps': 10,
                'success': True,
                'duration': 0.1
            }
            for i in range(100)
        ]
    }


@pytest.fixture(autouse=True)
def reset_random_state(seed):
    """Reset random state before each test."""
    np.random.seed(seed)
    import random
    random.seed(seed)


@pytest.fixture
def estimator_config():
    """Provide estimator configuration."""
    return EstimatorConfig(
        estimator_type="beta",
        alpha=2.0,
        beta=2.0,
        confidence_level=0.95,
        name="test_estimator"
    )


@pytest.fixture
def strategy_config(estimator_config):
    """Provide strategy configuration."""
    return StrategyConfig(
        strategy_type="kelly",
        risk_tolerance=0.5,
        kelly_fraction=0.25,
        min_confidence_threshold=0.1,
        estimator_config=estimator_config,
        name="test_strategy"
    )


@pytest.fixture
def simulation_config(seed):
    """Provide simulation configuration."""
    return SimulationConfig(
        num_simulations=100,
        num_episodes_per_sim=1,
        seed=seed,
        parallel=False,
        convergence_check=True,
        save_results=False
    )


@pytest.fixture
def game_config():
    """Provide game configuration."""
    return GameConfig(
        game_type="test",
        board_size=5,
        mine_count=3,
        initial_bankroll=1000.0,
        initial_bet_size=10.0
    )

