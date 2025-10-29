"""
Bayesian Adaptive Model for Dynamic Threshold Optimization

This module implements a comprehensive Bayesian framework for adaptive threshold
optimization in the Applied Probability Framework. It uses probabilistic
graphical models to dynamically adjust strategy parameters based on observed
performance and changing game conditions.

Features:
- Dynamic threshold adaptation using Bayesian inference
- Probabilistic graphical models for parameter relationships
- Multi-armed bandit optimization for strategy selection
- Real-time parameter updates based on performance feedback
- Uncertainty quantification for risk management
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import json
from datetime import datetime

try:
    import pgmpy
    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    HAS_PGMPY = True
except ImportError:
    HAS_PGMPY = False

from bayesian import BetaEstimator, VectorizedBetaEstimator


@dataclass
class AdaptiveThreshold:
    """Adaptive threshold with uncertainty quantification."""
    current_value: float
    confidence_interval: Tuple[float, float]
    update_count: int
    performance_history: deque
    uncertainty: float


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    win_rate: float
    avg_reward: float
    risk_score: float
    efficiency: float
    consistency: float
    last_updated: datetime


class BayesianAdaptiveModel:
    """
    Bayesian adaptive model for dynamic threshold optimization.
    
    This class implements a sophisticated Bayesian framework that:
    1. Models the relationship between game conditions and optimal thresholds
    2. Uses multi-armed bandit algorithms for strategy selection
    3. Provides uncertainty quantification for risk management
    4. Adapts parameters in real-time based on performance feedback
    """
    
    def __init__(self, 
                 strategy_names: List[str],
                 initial_thresholds: Optional[Dict[str, float]] = None,
                 learning_rate: float = 0.1,
                 confidence_level: float = 0.95):
        self.strategy_names = strategy_names
        self.learning_rate = learning_rate
        self.confidence_level = confidence_level
        
        # Initialize adaptive thresholds
        self.thresholds = self._initialize_thresholds(initial_thresholds)
        
        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.strategy_performance = {}
        
        # Bayesian estimators for each strategy
        self.bayesian_estimators = {}
        self._initialize_bayesian_estimators()
        
        # Multi-armed bandit for strategy selection
        self.bandit_estimators = {}
        self._initialize_bandit_estimators()
        
        # Probabilistic graphical model (if pgmpy available)
        self.bayesian_network = None
        if HAS_PGMPY:
            self._build_bayesian_network()
        
        # Context features for adaptation
        self.context_features = {
            'game_progress': 0.0,
            'risk_level': 0.0,
            'payout_multiplier': 1.0,
            'time_of_day': 0.0,
            'recent_volatility': 0.0
        }
    
    def _initialize_thresholds(self, initial_thresholds: Optional[Dict[str, float]]) -> Dict[str, AdaptiveThreshold]:
        """Initialize adaptive thresholds for each strategy."""
        thresholds = {}
        
        default_thresholds = {
            'aggression_threshold': 0.5,
            'risk_tolerance': 0.3,
            'win_target': 500.0,
            'stop_loss': 100.0,
            'exploration_rate': 0.1,
            'exploitation_threshold': 0.7
        }
        
        for strategy in self.strategy_names:
            thresholds[strategy] = {}
            for param_name, default_value in default_thresholds.items():
                initial_value = initial_thresholds.get(f"{strategy}_{param_name}", default_value) if initial_thresholds else default_value
                
                thresholds[strategy][param_name] = AdaptiveThreshold(
                    current_value=initial_value,
                    confidence_interval=(initial_value * 0.8, initial_value * 1.2),
                    update_count=0,
                    performance_history=deque(maxlen=100),
                    uncertainty=0.1
                )
        
        return thresholds
    
    def _initialize_bayesian_estimators(self):
        """Initialize Bayesian estimators for performance tracking."""
        for strategy in self.strategy_names:
            self.bayesian_estimators[strategy] = {
                'win_rate': BetaEstimator(1.0, 1.0),
                'reward_rate': BetaEstimator(1.0, 1.0),
                'risk_score': BetaEstimator(1.0, 1.0),
                'efficiency': BetaEstimator(1.0, 1.0)
            }
    
    def _initialize_bandit_estimators(self):
        """Initialize multi-armed bandit estimators for strategy selection."""
        for strategy in self.strategy_names:
            self.bandit_estimators[strategy] = BetaEstimator(1.0, 1.0)
    
    def _build_bayesian_network(self):
        """Build probabilistic graphical model for parameter relationships."""
        if not HAS_PGMPY:
            return
        
        # Define the Bayesian network structure
        model = BayesianNetwork([
            ('GameContext', 'StrategySelection'),
            ('GameContext', 'ThresholdAdaptation'),
            ('StrategyPerformance', 'ThresholdAdaptation'),
            ('ThresholdAdaptation', 'StrategySelection'),
            ('RiskLevel', 'ThresholdAdaptation'),
            ('RiskLevel', 'StrategySelection')
        ])
        
        # Define conditional probability distributions
        # GameContext -> StrategySelection
        game_context_cpd = TabularCPD(
            variable='StrategySelection',
            variable_card=len(self.strategy_names),
            values=[[0.25, 0.25, 0.25, 0.25]],  # Equal probability initially
            evidence=['GameContext'],
            evidence_card=[4]  # Low, Medium, High, Extreme risk
        )
        
        # Add more sophisticated CPDs based on domain knowledge
        # This is a simplified example - in practice, these would be learned from data
        
        self.bayesian_network = model
    
    def update_performance(self, 
                          strategy_name: str,
                          win: bool,
                          reward: float,
                          risk_score: float,
                          context: Dict[str, Any]):
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            win: Whether the game was won
            reward: Reward received
            risk_score: Risk score of the action taken
            context: Additional context information
        """
        # Update Bayesian estimators
        self.bayesian_estimators[strategy_name]['win_rate'].update(
            wins=1 if win else 0,
            losses=1 if not win else 0
        )
        
        # Normalize reward for Bayesian update
        normalized_reward = max(0, min(1, (reward + 10) / 20))  # Scale to [0, 1]
        self.bayesian_estimators[strategy_name]['reward_rate'].update(
            wins=1 if normalized_reward > 0.5 else 0,
            losses=1 if normalized_reward <= 0.5 else 0
        )
        
        # Update risk score estimator
        normalized_risk = max(0, min(1, risk_score))
        self.bayesian_estimators[strategy_name]['risk_score'].update(
            wins=1 if normalized_risk > 0.5 else 0,
            losses=1 if normalized_risk <= 0.5 else 0
        )
        
        # Update efficiency estimator
        efficiency = reward / max(1, context.get('steps_taken', 1))
        normalized_efficiency = max(0, min(1, (efficiency + 5) / 10))
        self.bayesian_estimators[strategy_name]['efficiency'].update(
            wins=1 if normalized_efficiency > 0.5 else 0,
            losses=1 if normalized_efficiency <= 0.5 else 0
        )
        
        # Update bandit estimator
        self.bandit_estimators[strategy_name].update(
            wins=1 if win else 0,
            losses=1 if not win else 0
        )
        
        # Update performance history
        self.performance_history[strategy_name].append({
            'win': win,
            'reward': reward,
            'risk_score': risk_score,
            'context': context,
            'timestamp': datetime.now()
        })
        
        # Update context features
        self._update_context_features(context)
        
        # Trigger threshold adaptation
        self._adapt_thresholds(strategy_name)
    
    def _update_context_features(self, context: Dict[str, Any]):
        """Update context features for adaptation."""
        self.context_features.update({
            'game_progress': context.get('game_progress', 0.0),
            'risk_level': context.get('risk_level', 0.0),
            'payout_multiplier': context.get('payout_multiplier', 1.0),
            'time_of_day': context.get('time_of_day', 0.0),
            'recent_volatility': context.get('recent_volatility', 0.0)
        })
    
    def _adapt_thresholds(self, strategy_name: str):
        """Adapt thresholds based on recent performance."""
        if strategy_name not in self.thresholds:
            return
        
        # Get recent performance
        recent_performance = list(self.performance_history[strategy_name])[-10:]  # Last 10 games
        if len(recent_performance) < 5:
            return
        
        # Calculate performance metrics
        recent_wins = sum(1 for p in recent_performance if p['win'])
        recent_rewards = [p['reward'] for p in recent_performance]
        recent_risks = [p['risk_score'] for p in recent_performance]
        
        win_rate = recent_wins / len(recent_performance)
        avg_reward = np.mean(recent_rewards)
        avg_risk = np.mean(recent_risks)
        
        # Adapt thresholds based on performance
        for param_name, threshold in self.thresholds[strategy_name].items():
            old_value = threshold.current_value
            
            # Calculate adaptation based on parameter type
            if 'aggression' in param_name.lower():
                # Increase aggression if winning with low risk
                if win_rate > 0.6 and avg_risk < 0.4:
                    new_value = old_value * (1 + self.learning_rate)
                elif win_rate < 0.4 or avg_risk > 0.6:
                    new_value = old_value * (1 - self.learning_rate)
                else:
                    new_value = old_value
                    
            elif 'risk' in param_name.lower():
                # Adjust risk tolerance based on performance
                if win_rate > 0.7:
                    new_value = min(1.0, old_value * (1 + self.learning_rate * 0.5))
                elif win_rate < 0.3:
                    new_value = max(0.1, old_value * (1 - self.learning_rate * 0.5))
                else:
                    new_value = old_value
                    
            elif 'target' in param_name.lower():
                # Adjust targets based on reward performance
                if avg_reward > threshold.current_value * 0.8:
                    new_value = old_value * (1 + self.learning_rate * 0.2)
                elif avg_reward < threshold.current_value * 0.3:
                    new_value = old_value * (1 - self.learning_rate * 0.2)
                else:
                    new_value = old_value
                    
            else:
                # Generic adaptation
                if win_rate > 0.6:
                    new_value = old_value * (1 + self.learning_rate * 0.1)
                elif win_rate < 0.4:
                    new_value = old_value * (1 - self.learning_rate * 0.1)
                else:
                    new_value = old_value
            
            # Update threshold
            threshold.current_value = max(0.01, min(1.0, new_value))  # Clamp to valid range
            threshold.update_count += 1
            
            # Update confidence interval
            uncertainty = self._calculate_uncertainty(strategy_name, param_name)
            threshold.uncertainty = uncertainty
            threshold.confidence_interval = (
                threshold.current_value - uncertainty,
                threshold.current_value + uncertainty
            )
            
            # Record performance
            threshold.performance_history.append({
                'value': threshold.current_value,
                'win_rate': win_rate,
                'avg_reward': avg_reward,
                'timestamp': datetime.now()
            })
    
    def _calculate_uncertainty(self, strategy_name: str, param_name: str) -> float:
        """Calculate uncertainty in threshold estimate."""
        threshold = self.thresholds[strategy_name][param_name]
        
        # Base uncertainty decreases with more updates
        base_uncertainty = 0.1 / (1 + threshold.update_count * 0.1)
        
        # Add uncertainty based on performance variance
        if len(threshold.performance_history) > 5:
            recent_values = [p['value'] for p in list(threshold.performance_history)[-10:]]
            value_variance = np.var(recent_values)
            base_uncertainty += min(0.1, value_variance)
        
        return base_uncertainty
    
    def select_strategy(self, context: Dict[str, Any]) -> str:
        """
        Select optimal strategy using multi-armed bandit approach.
        
        Args:
            context: Current game context
            
        Returns:
            Selected strategy name
        """
        # Update context features
        self._update_context_features(context)
        
        # Calculate strategy scores using Thompson sampling
        strategy_scores = {}
        
        for strategy in self.strategy_names:
            # Get Bayesian estimates
            win_rate_estimator = self.bandit_estimators[strategy]
            
            # Sample from posterior distribution
            alpha = win_rate_estimator.a
            beta = win_rate_estimator.b
            
            # Thompson sampling
            sampled_win_rate = np.random.beta(alpha, beta)
            
            # Adjust based on context
            context_bonus = self._calculate_context_bonus(strategy, context)
            strategy_scores[strategy] = sampled_win_rate + context_bonus
        
        # Select strategy with highest score
        selected_strategy = max(strategy_scores, key=strategy_scores.get)
        
        return selected_strategy
    
    def _calculate_context_bonus(self, strategy_name: str, context: Dict[str, Any]) -> float:
        """Calculate context-based bonus for strategy selection."""
        bonus = 0.0
        
        # Risk level bonus
        risk_level = context.get('risk_level', 0.5)
        if strategy_name.lower().find('aggressive') != -1 and risk_level > 0.7:
            bonus += 0.1
        elif strategy_name.lower().find('conservative') != -1 and risk_level < 0.3:
            bonus += 0.1
        
        # Game progress bonus
        game_progress = context.get('game_progress', 0.5)
        if strategy_name.lower().find('analytical') != -1 and game_progress > 0.5:
            bonus += 0.05
        
        # Recent performance bonus
        if strategy_name in self.performance_history:
            recent_games = list(self.performance_history[strategy_name])[-5:]
            if recent_games:
                recent_win_rate = sum(1 for g in recent_games if g['win']) / len(recent_games)
                if recent_win_rate > 0.6:
                    bonus += 0.1
        
        return bonus
    
    def get_adaptive_thresholds(self, strategy_name: str) -> Dict[str, float]:
        """Get current adaptive thresholds for a strategy."""
        if strategy_name not in self.thresholds:
            return {}
        
        return {
            param_name: threshold.current_value
            for param_name, threshold in self.thresholds[strategy_name].items()
        }
    
    def get_threshold_uncertainty(self, strategy_name: str, param_name: str) -> Tuple[float, float, float]:
        """Get threshold value with uncertainty bounds."""
        if (strategy_name not in self.thresholds or 
            param_name not in self.thresholds[strategy_name]):
            return 0.0, 0.0, 0.0
        
        threshold = self.thresholds[strategy_name][param_name]
        return (threshold.current_value, 
                threshold.confidence_interval[0], 
                threshold.confidence_interval[1])
    
    def get_strategy_performance_summary(self) -> Dict[str, StrategyPerformance]:
        """Get performance summary for all strategies."""
        summary = {}
        
        for strategy in self.strategy_names:
            if strategy not in self.bayesian_estimators:
                continue
            
            estimators = self.bayesian_estimators[strategy]
            
            summary[strategy] = StrategyPerformance(
                win_rate=estimators['win_rate'].mean(),
                avg_reward=estimators['reward_rate'].mean(),
                risk_score=estimators['risk_score'].mean(),
                efficiency=estimators['efficiency'].mean(),
                consistency=self._calculate_consistency(strategy),
                last_updated=datetime.now()
            )
        
        return summary
    
    def _calculate_consistency(self, strategy_name: str) -> float:
        """Calculate performance consistency for a strategy."""
        if strategy_name not in self.performance_history:
            return 0.0
        
        recent_games = list(self.performance_history[strategy_name])[-20:]
        if len(recent_games) < 5:
            return 0.0
        
        rewards = [g['reward'] for g in recent_games]
        return 1.0 - np.std(rewards) / (np.mean(rewards) + 1e-6)
    
    def export_model_state(self, filepath: str):
        """Export current model state to file."""
        state = {
            'thresholds': {},
            'bayesian_estimators': {},
            'performance_history': {},
            'context_features': self.context_features,
            'timestamp': datetime.now().isoformat()
        }
        
        # Export thresholds
        for strategy, params in self.thresholds.items():
            state['thresholds'][strategy] = {}
            for param_name, threshold in params.items():
                state['thresholds'][strategy][param_name] = {
                    'current_value': threshold.current_value,
                    'confidence_interval': threshold.confidence_interval,
                    'update_count': threshold.update_count,
                    'uncertainty': threshold.uncertainty
                }
        
        # Export Bayesian estimators
        for strategy, estimators in self.bayesian_estimators.items():
            state['bayesian_estimators'][strategy] = {}
            for param_name, estimator in estimators.items():
                state['bayesian_estimators'][strategy][param_name] = {
                    'a': estimator.a,
                    'b': estimator.b
                }
        
        # Export performance history (last 100 games per strategy)
        for strategy, history in self.performance_history.items():
            state['performance_history'][strategy] = list(history)[-100:]
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_model_state(self, filepath: str):
        """Load model state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Load thresholds
        for strategy, params in state.get('thresholds', {}).items():
            if strategy in self.thresholds:
                for param_name, param_data in params.items():
                    if param_name in self.thresholds[strategy]:
                        threshold = self.thresholds[strategy][param_name]
                        threshold.current_value = param_data['current_value']
                        threshold.confidence_interval = tuple(param_data['confidence_interval'])
                        threshold.update_count = param_data['update_count']
                        threshold.uncertainty = param_data['uncertainty']
        
        # Load Bayesian estimators
        for strategy, estimators in state.get('bayesian_estimators', {}).items():
            if strategy in self.bayesian_estimators:
                for param_name, estimator_data in estimators.items():
                    if param_name in self.bayesian_estimators[strategy]:
                        estimator = self.bayesian_estimators[strategy][param_name]
                        estimator.a = estimator_data['a']
                        estimator.b = estimator_data['b']
        
        # Load context features
        self.context_features.update(state.get('context_features', {}))
        
        # Load performance history
        for strategy, history in state.get('performance_history', {}).items():
            if strategy in self.performance_history:
                self.performance_history[strategy].extend(history)


def create_adaptive_model(strategy_names: List[str],
                         initial_thresholds: Optional[Dict[str, float]] = None) -> BayesianAdaptiveModel:
    """
    Factory function for creating Bayesian adaptive model.
    
    Args:
        strategy_names: List of strategy names
        initial_thresholds: Optional initial threshold values
        
    Returns:
        Configured Bayesian adaptive model
    """
    return BayesianAdaptiveModel(
        strategy_names=strategy_names,
        initial_thresholds=initial_thresholds,
        learning_rate=0.1,
        confidence_level=0.95
    )
