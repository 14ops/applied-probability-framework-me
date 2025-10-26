"""
Unit tests for character strategies.

Tests verify correct behavior of Takeshi, Yuzu, Aoi, Kazuya, and Lelouch strategies.
"""

import pytest
import random
from strategies import (
    TakeshiStrategy,
    YuzuStrategy,
    AoiStrategy,
    KazuyaStrategy,
    LelouchStrategy,
)
from strategies.base import StrategyBase


class TestTakeshiStrategy:
    """Tests for Takeshi's aggressive doubling strategy."""
    
    def setup_method(self):
        """Setup for each test."""
        self.rng = random.Random(42)
        self.strategy = TakeshiStrategy(
            config={
                'base_bet': 10.0,
                'max_bet': 100.0,
                'target_clicks': 8,
                'max_doubles': 2,
                'initial_bankroll': 1000.0
            },
            rng=self.rng
        )
    
    def test_initialization(self):
        """Test strategy initializes correctly."""
        assert self.strategy.name == "Takeshi Kovacs"
        assert self.strategy.target_clicks == 8
        assert self.strategy.base_bet == 10.0
        assert self.strategy.current_bet == 10.0
        assert self.strategy.consecutive_losses == 0
    
    def test_doubling_on_loss(self):
        """Test that bet doubles after loss."""
        initial_bet = self.strategy._calculate_bet_amount()
        assert initial_bet == 10.0
        
        # Simulate loss
        result = {'win': False, 'profit': -10.0, 'final_bankroll': 990.0, 'bet_amount': 10.0}
        self.strategy.on_result(result)
        
        # Should double
        next_bet = self.strategy._calculate_bet_amount()
        assert next_bet == 20.0
        assert self.strategy.consecutive_losses == 1
    
    def test_tranquility_mode_after_max_doubles(self):
        """Test tranquility mode activates after max doubles."""
        # Simulate 3 consecutive losses
        for i in range(3):
            bet = self.strategy._calculate_bet_amount()
            result = {'win': False, 'profit': -bet, 'final_bankroll': self.strategy.bankroll - bet, 'bet_amount': bet}
            self.strategy.on_result(result)
        
        # Should be in tranquility mode
        assert self.strategy.in_tranquility_mode
        assert self.strategy._calculate_bet_amount() == 10.0  # Back to base
    
    def test_reset_on_win(self):
        """Test that win resets doubling counter."""
        # Simulate loss then win
        result_loss = {'win': False, 'profit': -10.0, 'final_bankroll': 990.0, 'bet_amount': 10.0}
        self.strategy.on_result(result_loss)
        
        assert self.strategy.consecutive_losses == 1
        
        result_win = {'win': True, 'payout': 2.12, 'profit': 11.20, 'final_bankroll': 1001.20, 'bet_amount': 10.0}
        self.strategy.on_result(result_win)
        
        assert self.strategy.consecutive_losses == 0
        assert self.strategy.current_bet == 10.0
    
    def test_target_clicks(self):
        """Test that strategy targets correct number of clicks."""
        state = {
            'game_active': False,
            'revealed': [[False]*5 for _ in range(5)],
            'board_size': 5,
            'mine_count': 2,
            'clicks_made': 0,
            'bankroll': 1000.0
        }
        
        decision = self.strategy.decide(state)
        assert decision['action'] == 'bet'
        assert decision['max_clicks'] == 8


class TestYuzuStrategy:
    """Tests for Yuzu's controlled chaos strategy."""
    
    def setup_method(self):
        """Setup for each test."""
        self.rng = random.Random(42)
        self.strategy = YuzuStrategy(
            config={
                'base_bet': 10.0,
                'max_bet': 50.0,
                'target_clicks': 7,
                'chaos_factor': 0.3,
                'initial_bankroll': 1000.0
            },
            rng=self.rng
        )
    
    def test_initialization(self):
        """Test strategy initializes correctly."""
        assert self.strategy.name == "Yuzu"
        assert self.strategy.target_clicks == 7  # Fixed at 7
        assert self.strategy.chaos_factor == 0.3
    
    def test_fixed_target_clicks(self):
        """Test that Yuzu always targets 7 clicks."""
        state = {
            'game_active': False,
            'revealed': [[False]*5 for _ in range(5)],
            'board_size': 5,
            'mine_count': 2,
            'clicks_made': 0,
            'bankroll': 1000.0
        }
        
        decision = self.strategy.decide(state)
        assert decision['max_clicks'] == 7
        
        # Even after wins/losses, should stay at 7
        result_win = {'win': True, 'payout': 1.95, 'profit': 9.50, 'final_bankroll': 1009.50, 'bet_amount': 10.0}
        self.strategy.on_result(result_win)
        
        decision = self.strategy.decide(state)
        assert decision['max_clicks'] == 7
    
    def test_momentum_increases_on_win(self):
        """Test momentum increases on win."""
        initial_momentum = self.strategy.momentum
        
        result_win = {'win': True, 'payout': 1.95, 'profit': 9.50, 'final_bankroll': 1009.50, 'bet_amount': 10.0}
        self.strategy.on_result(result_win)
        
        assert self.strategy.momentum > initial_momentum
    
    def test_bet_variation(self):
        """Test that bets vary due to chaos factor."""
        bets = [self.strategy._calculate_bet_amount() for _ in range(10)]
        
        # Should have some variation
        assert len(set(bets)) > 1  # Not all the same


class TestAoiStrategy:
    """Tests for Aoi's cooperative sync strategy."""
    
    def setup_method(self):
        """Setup for each test."""
        self.rng = random.Random(42)
        self.strategy = AoiStrategy(
            config={
                'base_bet': 10.0,
                'max_bet': 80.0,
                'min_target_clicks': 6,
                'max_target_clicks': 7,
                'sync_threshold': 3,
                'initial_bankroll': 1000.0
            },
            rng=self.rng
        )
    
    def test_initialization(self):
        """Test strategy initializes correctly."""
        assert self.strategy.name == "Aoi"
        assert self.strategy.min_target_clicks == 6
        assert self.strategy.max_target_clicks == 7
        assert self.strategy.sync_threshold == 3
    
    def test_sync_activates_after_losses(self):
        """Test sync mode activates after threshold losses."""
        assert not self.strategy.in_sync_mode
        
        # Simulate 3 consecutive losses
        for i in range(3):
            result = {'win': False, 'profit': -10.0, 'final_bankroll': self.strategy.bankroll - 10.0, 'bet_amount': 10.0}
            self.strategy.on_result(result)
        
        # Should activate sync
        assert self.strategy.in_sync_mode
        assert self.strategy.sync_rounds_remaining > 0
    
    def test_conservative_bet_during_sync(self):
        """Test that bets are more conservative during sync."""
        normal_bet = self.strategy._calculate_bet_amount()
        
        # Trigger sync
        self.strategy._activate_sync()
        
        sync_bet = self.strategy._calculate_bet_amount()
        
        # Should be lower during sync
        assert sync_bet <= normal_bet


class TestKazuyaStrategy:
    """Tests for Kazuya's dagger strike strategy."""
    
    def setup_method(self):
        """Setup for each test."""
        self.rng = random.Random(42)
        self.strategy = KazuyaStrategy(
            config={
                'base_bet': 10.0,
                'normal_target': 5,
                'dagger_target': 10,
                'dagger_interval': 6,
                'initial_bankroll': 1000.0
            },
            rng=self.rng
        )
    
    def test_initialization(self):
        """Test strategy initializes correctly."""
        assert self.strategy.name == "Kazuya"
        assert self.strategy.normal_target == 5
        assert self.strategy.dagger_target == 10
        assert self.strategy.dagger_interval == 6
    
    def test_flat_betting(self):
        """Test that Kazuya always bets base amount."""
        bet1 = self.strategy._calculate_bet_amount()
        
        # After win
        result = {'win': True, 'payout': 1.53, 'profit': 5.30, 'final_bankroll': 1005.30, 'bet_amount': 10.0}
        self.strategy.on_result(result)
        bet2 = self.strategy._calculate_bet_amount()
        
        # After loss
        result = {'win': False, 'profit': -10.0, 'final_bankroll': 995.30, 'bet_amount': 10.0}
        self.strategy.on_result(result)
        bet3 = self.strategy._calculate_bet_amount()
        
        # All should be base bet
        assert bet1 == self.strategy.base_bet
        assert bet2 == self.strategy.base_bet
        assert bet3 == self.strategy.base_bet
    
    def test_dagger_strike_timing(self):
        """Test that dagger strike occurs at correct interval."""
        targets = []
        
        for i in range(10):
            target = self.strategy._determine_target()
            targets.append(target)
            
            # Simulate game
            result = {'win': True, 'payout': 1.5, 'profit': 5.0, 'final_bankroll': self.strategy.bankroll + 5.0, 'bet_amount': 10.0}
            self.strategy.on_result(result)
        
        # Should have dagger strikes at specific intervals
        assert targets[0] == 10  # First game is dagger (rounds_until_dagger starts at interval)
        assert targets[6] == 10  # After 6 more games


class TestLelouchStrategy:
    """Tests for Lelouch's strategic streak-based strategy."""
    
    def setup_method(self):
        """Setup for each test."""
        self.rng = random.Random(42)
        self.strategy = LelouchStrategy(
            config={
                'base_bet': 10.0,
                'safety_bet': 2.50,
                'max_bet': 100.0,
                'min_target': 6,
                'max_target': 8,
                'streak_multiplier': 1.5,
                'initial_bankroll': 1000.0
            },
            rng=self.rng
        )
    
    def test_initialization(self):
        """Test strategy initializes correctly."""
        assert self.strategy.name == "Lelouch vi Britannia"
        assert self.strategy.safety_bet == 2.50
        assert self.strategy.streak_multiplier == 1.5
    
    def test_escalation_on_win_streak(self):
        """Test bet escalation on win streaks."""
        initial_bet = self.strategy._calculate_strategic_bet()
        
        # Simulate 2 wins
        for i in range(2):
            result = {'win': True, 'payout': 1.95, 'profit': 9.50, 'final_bankroll': self.strategy.bankroll + 9.50, 'bet_amount': 10.0}
            self.strategy.on_result(result)
        
        # Should increase bet
        streak_bet = self.strategy._calculate_strategic_bet()
        assert streak_bet > initial_bet
    
    def test_safety_mode_after_big_loss(self):
        """Test safety mode activates after big bet loss."""
        # Simulate win streak to get big bet
        for i in range(3):
            result = {'win': True, 'payout': 1.95, 'profit': 9.50, 'final_bankroll': self.strategy.bankroll + 9.50, 'bet_amount': 10.0}
            self.strategy.on_result(result)
        
        big_bet = self.strategy._calculate_strategic_bet()
        
        # Now lose with big bet
        result_loss = {'win': False, 'profit': -big_bet, 'final_bankroll': self.strategy.bankroll - big_bet, 'bet_amount': big_bet}
        self.strategy.last_bet_was_big = True
        self.strategy.on_result(result_loss)
        
        # Should enter safety mode
        assert self.strategy.in_safety_mode
        
        # Next bet should be safety bet
        safety_bet = self.strategy._calculate_strategic_bet()
        assert safety_bet == self.strategy.safety_bet
    
    def test_confidence_adjustment(self):
        """Test confidence adjusts based on results."""
        initial_confidence = self.strategy.confidence
        
        # Win increases confidence
        result_win = {'win': True, 'payout': 1.95, 'profit': 9.50, 'final_bankroll': 1009.50, 'bet_amount': 10.0}
        self.strategy.on_result(result_win)
        assert self.strategy.confidence >= initial_confidence
        
        # Loss decreases confidence
        result_loss = {'win': False, 'profit': -10.0, 'final_bankroll': 999.50, 'bet_amount': 10.0}
        self.strategy.on_result(result_loss)
        self.strategy.on_result(result_loss)
        assert self.strategy.confidence < 1.0


class TestStrategyInterface:
    """Tests for StrategyBase interface compliance."""
    
    @pytest.mark.parametrize("strategy_class", [
        TakeshiStrategy,
        YuzuStrategy,
        AoiStrategy,
        KazuyaStrategy,
        LelouchStrategy,
    ])
    def test_implements_strategy_base(self, strategy_class):
        """Test that all strategies inherit from StrategyBase."""
        strategy = strategy_class()
        assert isinstance(strategy, StrategyBase)
    
    @pytest.mark.parametrize("strategy_class", [
        TakeshiStrategy,
        YuzuStrategy,
        AoiStrategy,
        KazuyaStrategy,
        LelouchStrategy,
    ])
    def test_has_required_methods(self, strategy_class):
        """Test that all strategies implement required methods."""
        strategy = strategy_class()
        
        assert hasattr(strategy, 'decide')
        assert callable(strategy.decide)
        
        assert hasattr(strategy, 'on_result')
        assert callable(strategy.on_result)
        
        assert hasattr(strategy, 'reset')
        assert callable(strategy.reset)
    
    @pytest.mark.parametrize("strategy_class", [
        TakeshiStrategy,
        YuzuStrategy,
        AoiStrategy,
        KazuyaStrategy,
        LelouchStrategy,
    ])
    def test_serialization(self, strategy_class):
        """Test that strategies can be serialized and deserialized."""
        strategy = strategy_class()
        
        # Serialize
        data = strategy.serialize()
        assert isinstance(data, dict)
        assert 'name' in data
        
        # Deserialize
        new_strategy = strategy_class()
        new_strategy.deserialize(data)
        
        # Should have same state
        assert new_strategy.name == strategy.name


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

