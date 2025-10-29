"""
Deep Reinforcement Learning Environment for Applied Probability Framework

This module implements a comprehensive RL environment that integrates with the
existing Mines Game simulator to enable deep learning-based strategy optimization.

Features:
- OpenAI Gym-compatible interface
- State space representation optimized for neural networks
- Reward shaping for strategy learning
- Action space discretization for Mines Game
- Multi-objective reward functions
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import torch
import torch.nn as nn
from collections import deque

from optimized_game_simulator import OptimizedGameSimulator
from bayesian import BetaEstimator, VectorizedBetaEstimator


@dataclass
class RLState:
    """Reinforcement Learning state representation."""
    board_state: np.ndarray  # Flattened board representation
    revealed_mask: np.ndarray  # Boolean mask of revealed cells
    mine_probabilities: np.ndarray  # Probability of mine in each cell
    game_metrics: np.ndarray  # [clicks_made, current_payout, risk_level]
    strategy_context: np.ndarray  # Strategy-specific context features


class MinesRLEnvironment(gym.Env):
    """
    Deep Reinforcement Learning environment for Mines Game.
    
    This environment provides a rich state representation and reward structure
    optimized for training deep neural networks to play the Mines Game optimally.
    """
    
    def __init__(self, 
                 board_size: int = 5,
                 mine_count: int = 3,
                 max_steps: int = 25,
                 reward_shaping: bool = True,
                 multi_objective: bool = True):
        super().__init__()
        
        self.board_size = board_size
        self.mine_count = mine_count
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        self.multi_objective = multi_objective
        
        # Initialize simulator
        self.simulator = OptimizedGameSimulator(board_size, mine_count)
        
        # State and action spaces
        self._setup_spaces()
        
        # RL-specific components
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_history = deque(maxlen=1000)
        
        # Bayesian estimators for adaptive learning
        self.bayesian_estimators = {}
        self._initialize_bayesian_estimators()
        
        # Reward components
        self.reward_components = {
            'survival': 0.0,
            'efficiency': 0.0,
            'risk_management': 0.0,
            'exploration': 0.0,
            'exploitation': 0.0
        }
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Observation space: [board_state, revealed_mask, mine_probs, game_metrics, strategy_context]
        board_dim = self.board_size * self.board_size
        obs_dim = (board_dim * 3 +  # board_state, revealed_mask, mine_probs
                  3 +  # game_metrics [clicks_made, payout, risk]
                  10)  # strategy_context features
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: click cell (row, col) or cash out
        self.action_space = spaces.Discrete(board_dim + 1)  # +1 for cash out
    
    def _initialize_bayesian_estimators(self):
        """Initialize Bayesian estimators for adaptive learning."""
        # Estimators for different aspects of the game
        self.bayesian_estimators = {
            'win_probability': BetaEstimator(1.0, 1.0),
            'mine_density': BetaEstimator(1.0, 1.0),
            'payout_efficiency': BetaEstimator(1.0, 1.0),
            'risk_tolerance': BetaEstimator(1.0, 1.0)
        }
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        self.simulator.initialize_game()
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_history.clear()
        
        # Reset reward components
        for key in self.reward_components:
            self.reward_components[key] = 0.0
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action and return next state, reward, done, info.
        
        Args:
            action: Action index (0 to board_size^2 for clicks, board_size^2 for cash out)
            
        Returns:
            (observation, reward, done, info)
        """
        self.current_step += 1
        
        # Convert action to game action
        if action == self.board_size * self.board_size:
            # Cash out action
            success, message = self.simulator.cash_out()
            if success:
                reward = self.simulator.state.current_payout
                done = True
                info = {'action_type': 'cash_out', 'message': message}
            else:
                reward = 0.0
                done = True
                info = {'action_type': 'cash_out', 'error': message}
        else:
            # Click cell action
            row = action // self.board_size
            col = action % self.board_size
            
            success, message = self.simulator.click_cell(row, col)
            
            if not success:
                reward = -1.0  # Penalty for invalid action
                done = True
                info = {'action_type': 'click', 'error': message}
            else:
                if message == "Mine":
                    reward = -10.0  # Large penalty for hitting mine
                    done = True
                    info = {'action_type': 'click', 'result': 'mine'}
                elif message == "All safe cells revealed":
                    reward = 20.0  # Large reward for completing game
                    done = True
                    info = {'action_type': 'click', 'result': 'win'}
                else:
                    # Safe click - calculate shaped reward
                    reward = self._calculate_reward(action, message)
                    done = False
                    info = {'action_type': 'click', 'result': 'safe'}
        
        # Update Bayesian estimators
        self._update_bayesian_estimators(action, reward, done)
        
        # Update episode history
        self.episode_history.append({
            'action': action,
            'reward': reward,
            'done': done,
            'step': self.current_step
        })
        
        # Check if max steps reached
        if self.current_step >= self.max_steps:
            done = True
            if not info.get('result') == 'win':
                reward -= 5.0  # Penalty for not completing in time
        
        self.episode_reward += reward
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation as numpy array."""
        state = self.simulator.get_state()
        
        # Flatten board state
        board_flat = state['board'].flatten().astype(np.float32)
        revealed_flat = state['revealed'].flatten().astype(np.float32)
        
        # Calculate mine probabilities
        mine_probs = self._calculate_mine_probabilities()
        
        # Game metrics
        clicks_made = state['clicks_made']
        current_payout = state['current_payout']
        risk_level = self._calculate_risk_level()
        game_metrics = np.array([clicks_made, current_payout, risk_level], dtype=np.float32)
        
        # Strategy context features
        strategy_context = self._get_strategy_context()
        
        # Normalize and combine
        obs = np.concatenate([
            board_flat / 1.0,  # Normalize board state
            revealed_flat,  # Already 0/1
            mine_probs,  # Already 0-1
            game_metrics / np.array([25.0, 10.0, 1.0]),  # Normalize metrics
            strategy_context
        ])
        
        return obs.astype(np.float32)
    
    def _calculate_mine_probabilities(self) -> np.ndarray:
        """Calculate probability of mine in each cell."""
        state = self.simulator.get_state()
        board_size = state['board_size']
        mine_count = state['mine_count']
        revealed = state['revealed']
        
        # Count unrevealed cells
        unrevealed_count = np.sum(~revealed)
        if unrevealed_count == 0:
            return np.zeros(board_size * board_size, dtype=np.float32)
        
        # Base probability
        base_prob = mine_count / unrevealed_count
        
        # Create probability array
        probs = np.full(board_size * board_size, base_prob, dtype=np.float32)
        
        # Set revealed cells to 0 probability
        probs[revealed.flatten()] = 0.0
        
        return probs
    
    def _calculate_risk_level(self) -> float:
        """Calculate current risk level."""
        state = self.simulator.get_state()
        clicks_made = state['clicks_made']
        total_safe = self.board_size * self.board_size - self.mine_count
        
        if total_safe == 0:
            return 1.0
        
        # Risk increases as we get closer to the end
        progress = clicks_made / total_safe
        return min(progress, 1.0)
    
    def _get_strategy_context(self) -> np.ndarray:
        """Get strategy-specific context features."""
        # Extract features that would be useful for different strategies
        state = self.simulator.get_state()
        
        # Basic game state features
        clicks_made = state['clicks_made']
        current_payout = state['current_payout']
        revealed = state['revealed']
        
        # Calculate additional features
        revealed_count = np.sum(revealed)
        total_cells = self.board_size * self.board_size
        progress = revealed_count / total_cells
        
        # Adjacent cell analysis
        adjacent_safe_ratio = self._calculate_adjacent_safe_ratio()
        
        # Risk assessment
        risk_level = self._calculate_risk_level()
        
        # Bayesian estimates
        win_prob = self.bayesian_estimators['win_probability'].mean()
        mine_density = self.bayesian_estimators['mine_density'].mean()
        
        # Strategy-specific features
        features = np.array([
            progress,  # Game progress
            current_payout,  # Current multiplier
            risk_level,  # Risk level
            adjacent_safe_ratio,  # Adjacent cell safety
            win_prob,  # Bayesian win probability
            mine_density,  # Bayesian mine density
            clicks_made / 25.0,  # Normalized clicks
            (current_payout - 1.0) / 10.0,  # Normalized payout
            risk_level * progress,  # Risk-progress interaction
            np.random.random()  # Random factor for exploration
        ], dtype=np.float32)
        
        return features
    
    def _calculate_adjacent_safe_ratio(self) -> float:
        """Calculate ratio of safe adjacent cells."""
        state = self.simulator.get_state()
        revealed = state['revealed']
        
        safe_adjacent = 0
        total_adjacent = 0
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                if not revealed[r, c]:  # Unrevealed cell
                    # Count adjacent revealed safe cells
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < self.board_size and 
                                0 <= nc < self.board_size and 
                                revealed[nr, nc]):
                                total_adjacent += 1
                                if state['board'][nr, nc] == 0:  # Safe cell
                                    safe_adjacent += 1
        
        return safe_adjacent / max(total_adjacent, 1)
    
    def _calculate_reward(self, action: int, message: str) -> float:
        """Calculate shaped reward for the action."""
        if not self.reward_shaping:
            return 0.1  # Simple reward for safe clicks
        
        reward = 0.0
        
        # Survival reward (always positive for safe clicks)
        reward += 0.1
        
        # Efficiency reward (based on payout increase)
        state = self.simulator.get_state()
        payout_increase = state['current_payout'] - 1.0
        reward += payout_increase * 0.5
        
        # Risk management reward (penalize high-risk moves)
        risk_level = self._calculate_risk_level()
        if risk_level > 0.8:
            reward -= 0.2
        
        # Exploration reward (encourage trying new areas)
        if self._is_exploration_action(action):
            reward += 0.05
        
        # Exploitation reward (encourage safe, high-probability moves)
        if self._is_exploitation_action(action):
            reward += 0.03
        
        return reward
    
    def _is_exploration_action(self, action: int) -> bool:
        """Check if action represents exploration."""
        # Simple heuristic: actions in less explored areas
        state = self.simulator.get_state()
        revealed = state['revealed']
        
        if action == self.board_size * self.board_size:
            return False
        
        row = action // self.board_size
        col = action % self.board_size
        
        # Check if this area has been less explored
        adjacent_revealed = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if (0 <= nr < self.board_size and 
                    0 <= nc < self.board_size and 
                    revealed[nr, nc]):
                    adjacent_revealed += 1
        
        return adjacent_revealed < 3  # Less explored area
    
    def _is_exploitation_action(self, action: int) -> bool:
        """Check if action represents exploitation."""
        # Simple heuristic: actions in well-explored, safe areas
        state = self.simulator.get_state()
        revealed = state['revealed']
        
        if action == self.board_size * self.board_size:
            return True  # Cash out is exploitation
        
        row = action // self.board_size
        col = action % self.board_size
        
        # Check if this area has been well explored and is safe
        adjacent_safe = 0
        adjacent_revealed = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if (0 <= nr < self.board_size and 
                    0 <= nc < self.board_size and 
                    revealed[nr, nc]):
                    adjacent_revealed += 1
                    if state['board'][nr, nc] == 0:  # Safe cell
                        adjacent_safe += 1
        
        return adjacent_revealed >= 3 and adjacent_safe >= 2
    
    def _update_bayesian_estimators(self, action: int, reward: float, done: bool):
        """Update Bayesian estimators based on action and outcome."""
        # Update win probability
        if done:
            self.bayesian_estimators['win_probability'].update(
                wins=1 if reward > 0 else 0,
                losses=1 if reward <= 0 else 0
            )
        
        # Update mine density estimate
        state = self.simulator.get_state()
        revealed_count = np.sum(state['revealed'])
        if revealed_count > 0:
            mine_density = self.mine_count / (self.board_size * self.board_size)
            self.bayesian_estimators['mine_density'].update(
                wins=1 if mine_density > 0.1 else 0,
                losses=1 if mine_density <= 0.1 else 0
            )
        
        # Update payout efficiency
        if reward > 0:
            efficiency = reward / max(self.current_step, 1)
            self.bayesian_estimators['payout_efficiency'].update(
                wins=1 if efficiency > 0.5 else 0,
                losses=1 if efficiency <= 0.5 else 0
            )
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            state = self.simulator.get_state()
            print(f"Step: {self.current_step}")
            print(f"Payout: {state['current_payout']:.2f}x")
            print(f"Revealed: {np.sum(state['revealed'])}/{self.board_size * self.board_size}")
            print(f"Reward: {self.episode_reward:.2f}")
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get statistics for the current episode."""
        if not self.episode_history:
            return {}
        
        rewards = [step['reward'] for step in self.episode_history]
        
        return {
            'total_reward': sum(rewards),
            'avg_reward': np.mean(rewards),
            'max_reward': max(rewards),
            'min_reward': min(rewards),
            'steps': len(self.episode_history),
            'final_payout': self.simulator.state.current_payout,
            'won': self.simulator.state.won
        }
    
    def close(self):
        """Clean up environment resources."""
        pass


class MultiObjectiveRewardWrapper(gym.Wrapper):
    """
    Wrapper for multi-objective reward shaping.
    
    This wrapper provides different reward structures optimized for
    different strategy types (aggressive, conservative, analytical, etc.).
    """
    
    def __init__(self, env: MinesRLEnvironment, strategy_type: str = 'analytical'):
        super().__init__(env)
        self.strategy_type = strategy_type
        self.reward_weights = self._get_reward_weights(strategy_type)
    
    def _get_reward_weights(self, strategy_type: str) -> Dict[str, float]:
        """Get reward weights for different strategy types."""
        weights = {
            'aggressive': {
                'survival': 0.2,
                'efficiency': 0.4,
                'risk_management': 0.1,
                'exploration': 0.2,
                'exploitation': 0.1
            },
            'conservative': {
                'survival': 0.5,
                'efficiency': 0.2,
                'risk_management': 0.3,
                'exploration': 0.0,
                'exploitation': 0.0
            },
            'analytical': {
                'survival': 0.3,
                'efficiency': 0.3,
                'risk_management': 0.2,
                'exploration': 0.1,
                'exploitation': 0.1
            },
            'strategic': {
                'survival': 0.25,
                'efficiency': 0.25,
                'risk_management': 0.25,
                'exploration': 0.125,
                'exploitation': 0.125
            }
        }
        return weights.get(strategy_type, weights['analytical'])

    def step(self, action):
        """Modify reward based on strategy type."""
        obs, reward, done, info = self.env.step(action)
        
        if self.strategy_type != 'analytical':
            # Apply strategy-specific reward modification
            reward = self._modify_reward(reward, action, obs, done)
        
        return obs, reward, done, info
    
    def _modify_reward(self, base_reward: float, action: int, obs: np.ndarray, done: bool) -> float:
        """Modify reward based on strategy type."""
        # Extract features from observation
        risk_level = obs[-10]  # Risk level from strategy context
        payout = obs[-9]  # Current payout
        progress = obs[-10]  # Game progress
        
        # Apply strategy-specific modifications
        if self.strategy_type == 'aggressive':
            # Reward high-risk, high-reward moves
            if payout > 2.0:
                base_reward *= 1.5
            if risk_level > 0.7:
                base_reward *= 1.2
        
        elif self.strategy_type == 'conservative':
            # Penalize high-risk moves
            if risk_level > 0.5:
                base_reward *= 0.5
            # Reward early cash outs
            if action == self.env.board_size * self.env.board_size and progress < 0.5:
                base_reward += 1.0
        
        elif self.strategy_type == 'strategic':
            # Balance risk and reward
            if 0.3 <= risk_level <= 0.7:
                base_reward *= 1.2
            elif risk_level > 0.8:
                base_reward *= 0.8
        
        return base_reward


def create_rl_environment(board_size: int = 5, 
                         mine_count: int = 3,
                         strategy_type: str = 'analytical') -> gym.Env:
    """
    Factory function for creating RL environments.
    
    Args:
        board_size: Size of the game board
        mine_count: Number of mines
        strategy_type: Type of strategy to optimize for
        
    Returns:
        Configured RL environment
    """
    env = MinesRLEnvironment(
        board_size=board_size,
        mine_count=mine_count,
        reward_shaping=True,
        multi_objective=True
    )
    
    if strategy_type != 'analytical':
        env = MultiObjectiveRewardWrapper(env, strategy_type)
    
    return env