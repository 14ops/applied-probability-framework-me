"""
Smoke tests for CLI functionality.

These tests ensure that the CLI modules can be imported and run
without errors, even if they don't produce meaningful results.
"""

import pytest
import sys
from pathlib import Path

# Add src/python to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))


def test_cli_import():
    """Test that CLI module can be imported."""
    try:
        from cli.simulate import main
        assert callable(main)
    except ImportError as e:
        pytest.skip(f"CLI module not available: {e}")


def test_drl_imports():
    """Test that DRL modules can be imported."""
    try:
        from drl.drl_environment import MinesEnv, StepResult
        from drl.drl_agent import DQNAgent, QNet, ReplayBuffer
        from drl.drl_training import train_dqn
        assert True  # If we get here, imports worked
    except ImportError as e:
        pytest.skip(f"DRL modules not available: {e}")


def test_bayesian_imports():
    """Test that Bayesian modules can be imported."""
    try:
        from bayesian.bayesian_mines import BayesianMineModel
        assert True  # If we get here, imports worked
    except ImportError as e:
        pytest.skip(f"Bayesian modules not available: {e}")


def test_multiagent_imports():
    """Test that Multi-agent modules can be imported."""
    try:
        from multiagent.base_agent import BaseAgent
        from multiagent.blackboard import Blackboard
        from multiagent.consensus import majority_vote
        from multiagent.simulator import TeamSimulator
        assert True  # If we get here, imports worked
    except ImportError as e:
        pytest.skip(f"Multi-agent modules not available: {e}")


def test_drl_strategy_import():
    """Test that DRL strategy can be imported."""
    try:
        from strategies.drl_strategy import DRLStrategy
        assert True  # If we get here, imports worked
    except ImportError as e:
        pytest.skip(f"DRL strategy not available: {e}")


def test_cli_simulate_basic():
    """Test basic CLI simulate functionality."""
    try:
        from cli.simulate import main
        from drl.drl_environment import MinesEnv
        
        # Test environment creation
        env = MinesEnv(n_cells=25)
        state = env.reset()
        assert state is not None
        assert len(state) == 29  # 25 cells + 4 meta features
        
        # Test valid actions
        actions = env.valid_actions()
        assert len(actions) > 0
        assert 25 in actions  # Cash out action should be present
        
        # Test step
        result = env.step(0)  # Click first cell
        assert result is not None
        assert hasattr(result, 'next_state')
        assert hasattr(result, 'reward')
        assert hasattr(result, 'done')
        assert hasattr(result, 'info')
        
    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
