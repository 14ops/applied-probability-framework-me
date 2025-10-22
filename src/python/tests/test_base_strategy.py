"""
Unit tests for base strategy classes.

Tests the BaseStrategy interface and common strategy functionality.
"""

import pytest
import numpy as np

from core.base_strategy import BaseStrategy


class TestBaseStrategy:
    """Tests for BaseStrategy abstract class and interface."""
    
    def test_base_strategy_is_abstract(self):
        """BaseStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStrategy("test")
    
    def test_simple_strategy_works(self, simple_strategy):
        """Test that fixture strategy works correctly."""
        assert simple_strategy.name == "simple"
        
        state = {'position': 0}
        valid_actions = ['forward', 'backward', 'stay']
        action = simple_strategy.select_action(state, valid_actions)
        
        assert action == 'forward'
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        class TestStrategy(BaseStrategy):
            def select_action(self, state, valid_actions):
                return valid_actions[0]
        
        strategy = TestStrategy(name="test", config={'param': 1})
        assert strategy.name == "test"
        assert strategy.config == {'param': 1}
    
    def test_statistics_initialization(self):
        """Test that statistics are initialized."""
        class TestStrategy(BaseStrategy):
            def select_action(self, state, valid_actions):
                return valid_actions[0]
        
        strategy = TestStrategy("test")
        stats = strategy.get_statistics()
        
        assert stats['games_played'] == 0
        assert stats['total_reward'] == 0.0
        assert stats['wins'] == 0
        assert stats['losses'] == 0
        assert stats['win_rate'] == 0.0
        assert stats['avg_reward'] == 0.0
    
    def test_update_increments_statistics(self):
        """Test that update() modifies statistics."""
        class TestStrategy(BaseStrategy):
            def select_action(self, state, valid_actions):
                return valid_actions[0]
        
        strategy = TestStrategy("test")
        
        # Simulate a winning game
        strategy.update(state={'pos': 0}, action='forward', 
                       reward=10.0, next_state={'pos': 1}, done=True)
        
        stats = strategy.get_statistics()
        assert stats['games_played'] == 1
        assert stats['total_reward'] == 10.0
        assert stats['wins'] == 1
        assert stats['losses'] == 0
    
    def test_update_multiple_episodes(self):
        """Test statistics over multiple episodes."""
        class TestStrategy(BaseStrategy):
            def select_action(self, state, valid_actions):
                return valid_actions[0]
        
        strategy = TestStrategy("test")
        
        # Game 1: Win
        strategy.update({}, 'a', 5.0, {}, done=True)
        
        # Game 2: Loss
        strategy.update({}, 'a', -2.0, {}, done=True)
        
        # Game 3: Win
        strategy.update({}, 'a', 8.0, {}, done=True)
        
        stats = strategy.get_statistics()
        assert stats['games_played'] == 3
        assert stats['total_reward'] == 11.0
        assert stats['wins'] == 2
        assert stats['losses'] == 1
        assert abs(stats['win_rate'] - 2/3) < 1e-10
        assert abs(stats['avg_reward'] - 11.0/3) < 1e-10
    
    def test_update_mid_episode(self):
        """Test that mid-episode updates don't increment game counter."""
        class TestStrategy(BaseStrategy):
            def select_action(self, state, valid_actions):
                return valid_actions[0]
        
        strategy = TestStrategy("test")
        
        # Multiple steps in same episode
        strategy.update({}, 'a', 1.0, {}, done=False)
        strategy.update({}, 'a', 2.0, {}, done=False)
        strategy.update({}, 'a', 3.0, {}, done=True)
        
        stats = strategy.get_statistics()
        assert stats['games_played'] == 1
        assert stats['total_reward'] == 6.0
    
    def test_reset_method(self):
        """Test that reset() is callable."""
        class TestStrategy(BaseStrategy):
            def __init__(self, name, config=None):
                super().__init__(name, config)
                self.episode_data = []
            
            def select_action(self, state, valid_actions):
                return valid_actions[0]
            
            def reset(self):
                super().reset()
                self.episode_data = []
        
        strategy = TestStrategy("test")
        strategy.episode_data = [1, 2, 3]
        
        strategy.reset()
        assert strategy.episode_data == []
    
    def test_save_and_load(self, temp_dir):
        """Test saving and loading strategy."""
        class TestStrategy(BaseStrategy):
            def select_action(self, state, valid_actions):
                return valid_actions[0]
        
        strategy = TestStrategy("test_save", config={'param': 42})
        strategy.update({}, 'a', 10.0, {}, done=True)
        
        save_path = temp_dir / "strategy.json"
        strategy.save(str(save_path))
        
        # Create new strategy and load
        new_strategy = TestStrategy("temp")
        new_strategy.load(str(save_path))
        
        assert new_strategy.name == "test_save"
        assert new_strategy.config == {'param': 42}
        assert new_strategy.get_statistics()['total_reward'] == 10.0
    
    def test_repr(self):
        """Test string representation."""
        class TestStrategy(BaseStrategy):
            def select_action(self, state, valid_actions):
                return valid_actions[0]
        
        strategy = TestStrategy("test_repr")
        strategy.update({}, 'a', 10.0, {}, done=True)
        
        repr_str = repr(strategy)
        assert "TestStrategy" in repr_str
        assert "test_repr" in repr_str
        assert "win_rate" in repr_str
        assert "games_played" in repr_str


class TestStrategyWithSimulator:
    """Integration tests with simple simulator."""
    
    def test_strategy_simulation_loop(self, simple_simulator, simple_strategy):
        """Test full simulation loop with strategy."""
        state = simple_simulator.reset()
        simple_strategy.reset()
        
        total_reward = 0
        done = False
        
        while not done:
            valid_actions = simple_simulator.get_valid_actions()
            if not valid_actions:
                break
            
            action = simple_strategy.select_action(state, valid_actions)
            next_state, reward, done, info = simple_simulator.step(action)
            
            simple_strategy.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
        
        # Should complete successfully
        assert done
        assert total_reward > 0  # All forward moves give positive reward
        
        stats = simple_strategy.get_statistics()
        assert stats['games_played'] == 1
        assert stats['wins'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

