"""
Deep Reinforcement Learning Agent for Applied Probability Framework

This module implements a sophisticated DRL agent using PyTorch for the Mines Game,
following best practices from the TensorFlow Agents tutorial and modern RL techniques.

Features:
- Deep Q-Network (DQN) with experience replay
- Double DQN for improved stability
- Dueling DQN architecture for better value estimation
- Prioritized experience replay
- TensorBoard logging for training progress
- Integration with existing simulation loop
- Bayesian risk module integration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional, Dict, Any
import logging
from datetime import datetime
import os
import json

# TensorBoard logging
from torch.utils.tensorboard import SummaryWriter

# Import our existing modules
from drl_environment import MinesRLEnvironment, create_rl_environment
from bayesian import BetaEstimator, VectorizedBetaEstimator
from performance_profiler import profile_function


# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer for DQN.
    
    Implements proportional prioritization with importance sampling
    to improve learning efficiency.
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
    def add(self, experience: Experience):
        """Add experience to buffer with maximum priority."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with prioritized selection."""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network architecture.
    
    Separates value and advantage estimation for better learning
    of state values and action advantages.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 512,
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dueling DQN."""
        features = self.feature_layers(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class BayesianRiskModule:
    """
    Bayesian risk assessment module for DRL agent.
    
    Provides probabilistic risk estimates and uncertainty quantification
    to guide the DRL agent's decision-making process.
    """
    
    def __init__(self, board_size: int = 5, mine_count: int = 3):
        self.board_size = board_size
        self.mine_count = mine_count
        self.total_cells = board_size * board_size
        self.safe_cells = self.total_cells - mine_count
        
        # Bayesian estimators for different risk factors
        self.risk_estimators = {
            'mine_density': BetaEstimator(1.0, 1.0),
            'click_success': BetaEstimator(1.0, 1.0),
            'payout_efficiency': BetaEstimator(1.0, 1.0),
            'board_complexity': BetaEstimator(1.0, 1.0)
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
    
    def assess_risk(self, state: np.ndarray, action: int) -> Dict[str, float]:
        """
        Assess risk for a given state-action pair.
        
        Args:
            state: Current game state
            action: Proposed action
            
        Returns:
            Dictionary of risk metrics
        """
        # Extract game state information
        board_state = state[:self.total_cells]
        revealed_mask = state[self.total_cells:2*self.total_cells]
        mine_probs = state[2*self.total_cells:3*self.total_cells]
        game_metrics = state[3*self.total_cells:3*self.total_cells+3]
        
        clicks_made = int(game_metrics[0] * 25)  # Denormalize
        current_payout = game_metrics[1] * 10    # Denormalize
        risk_level = game_metrics[2]
        
        # Calculate mine density risk
        unrevealed_count = np.sum(~revealed_mask.astype(bool))
        if unrevealed_count > 0:
            mine_density = self.mine_count / unrevealed_count
        else:
            mine_density = 1.0
        
        # Calculate action-specific risk
        if action < self.total_cells:
            row = action // self.board_size
            col = action % self.board_size
            action_risk = mine_probs[action]
        else:
            action_risk = 0.0  # Cash out action
        
        # Calculate payout efficiency risk
        if clicks_made > 0:
            payout_efficiency = current_payout / clicks_made
        else:
            payout_efficiency = 1.0
        
        # Calculate board complexity risk
        revealed_count = np.sum(revealed_mask.astype(bool))
        board_complexity = revealed_count / self.total_cells
        
        # Update Bayesian estimators
        self.risk_estimators['mine_density'].update(
            wins=1 if mine_density < 0.5 else 0,
            losses=1 if mine_density >= 0.5 else 0
        )
        
        self.risk_estimators['click_success'].update(
            wins=1 if action_risk < 0.5 else 0,
            losses=1 if action_risk >= 0.5 else 0
        )
        
        self.risk_estimators['payout_efficiency'].update(
            wins=1 if payout_efficiency > 0.1 else 0,
            losses=1 if payout_efficiency <= 0.1 else 0
        )
        
        self.risk_estimators['board_complexity'].update(
            wins=1 if board_complexity > 0.5 else 0,
            losses=1 if board_complexity <= 0.5 else 0
        )
        
        # Calculate risk metrics
        risk_metrics = {
            'mine_density_risk': mine_density,
            'action_risk': action_risk,
            'payout_efficiency_risk': 1.0 - min(payout_efficiency / 2.0, 1.0),
            'board_complexity_risk': 1.0 - board_complexity,
            'overall_risk': (mine_density + action_risk + (1.0 - payout_efficiency) + (1.0 - board_complexity)) / 4.0,
            'bayesian_confidence': self._calculate_confidence(),
            'risk_category': self._categorize_risk(mine_density + action_risk)
        }
        
        return risk_metrics
    
    def _calculate_confidence(self) -> float:
        """Calculate overall confidence in risk estimates."""
        confidences = []
        for estimator in self.risk_estimators.values():
            n = estimator.a + estimator.b
            if n > 0:
                confidence = 1.0 - (estimator.variance() / 0.25)  # Normalize variance
                confidences.append(max(0.0, min(1.0, confidence)))
        
        return np.mean(confidences) if confidences else 0.5
    
    def _categorize_risk(self, combined_risk: float) -> str:
        """Categorize risk level."""
        if combined_risk < self.risk_thresholds['low']:
            return 'low'
        elif combined_risk < self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'high'


class DRLAgent:
    """
    Deep Reinforcement Learning Agent for Mines Game.
    
    Implements Double Dueling DQN with prioritized experience replay
    and Bayesian risk integration.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 64,
                 memory_size: int = 100000,
                 target_update_freq: int = 1000,
                 device: str = 'auto'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device selection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logging.info(f"DRL Agent using device: {self.device}")
        
        # Neural networks
        self.q_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay
        self.memory = PrioritizedReplayBuffer(memory_size)
        
        # Bayesian risk module
        self.risk_module = BayesianRiskModule()
        
        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
        # TensorBoard logging
        self.writer = SummaryWriter(f'runs/drl_agent_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
    
    @profile_function(track_memory=True)
    def select_action(self, state: np.ndarray, valid_actions: List[int] = None) -> int:
        """
        Select action using epsilon-greedy policy with risk consideration.

        Args:
            state: Current state
            valid_actions: List of valid actions (if None, all actions are valid)
            
        Returns:
            Selected action
        """
        if valid_actions is None:
            valid_actions = list(range(self.action_dim))
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random action
            action = random.choice(valid_actions)
        else:
            # Greedy action with risk consideration
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
            
            # Apply risk-based action filtering
            risk_adjusted_q_values = self._apply_risk_adjustment(state, q_values, valid_actions)
            
            # Select best valid action
            valid_q_values = [risk_adjusted_q_values[a] for a in valid_actions]
            action = valid_actions[np.argmax(valid_q_values)]
        
        return action
    
    def _apply_risk_adjustment(self, 
                              state: np.ndarray, 
                              q_values: np.ndarray, 
                              valid_actions: List[int]) -> np.ndarray:
        """Apply risk-based adjustment to Q-values."""
        adjusted_q_values = q_values.copy()
        
        for action in valid_actions:
            # Assess risk for this action
            risk_metrics = self.risk_module.assess_risk(state, action)
            
            # Adjust Q-value based on risk
            risk_penalty = risk_metrics['overall_risk'] * 0.5  # Scale risk penalty
            confidence_bonus = risk_metrics['bayesian_confidence'] * 0.1
            
            adjusted_q_values[action] = q_values[action] - risk_penalty + confidence_bonus
        
        return adjusted_q_values
    
    def store_experience(self, 
                        state: np.ndarray,
                        action: int,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool):
        """Store experience in replay buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.add(experience)
    
    @profile_function(track_memory=True)
    def train(self) -> Optional[float]:
        """
        Train the DQN on a batch of experiences.
        
        Returns:
            Training loss if training occurred, None otherwise
        """
        if len(self.memory.buffer) < self.batch_size:
            return None
        
        # Sample batch with prioritization
        experiences, indices, weights = self.memory.sample(self.batch_size)
        
        if not experiences:
            return None
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values using target network (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).max(1)[1]
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calculate TD errors
        td_errors = target_q_values - current_q_values
        
        # Calculate loss with importance sampling weights
        loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update priorities
        priorities = torch.abs(td_errors).cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, priorities)
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Log training metrics
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        self.writer.add_scalar('Training/Loss', loss_value, self.training_step)
        self.writer.add_scalar('Training/Epsilon', self.epsilon, self.training_step)
        
        return loss_value
    
    def train_episode(self, env: MinesRLEnvironment) -> Dict[str, float]:
        """
        Train the agent for one episode.

        Args:
            env: RL environment

        Returns:
            Episode statistics
        """
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done and steps < 1000:  # Prevent infinite episodes
            # Get valid actions
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            # Select action
            action = self.select_action(state, valid_actions)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            self.store_experience(state, action, reward, next_state, done)
            
            # Train
            loss = self.train()
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
        
        # Record episode statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        
        # Log episode metrics
        episode_num = len(self.episode_rewards)
        self.writer.add_scalar('Episode/Reward', total_reward, episode_num)
        self.writer.add_scalar('Episode/Length', steps, episode_num)
        self.writer.add_scalar('Episode/Epsilon', self.epsilon, episode_num)
        
        # Log average performance
        if episode_num % 100 == 0:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
            
            self.writer.add_scalar('Performance/AvgReward_100', avg_reward, episode_num)
            self.writer.add_scalar('Performance/AvgLength_100', avg_length, episode_num)
            
            logging.info(f"Episode {episode_num}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.2f}")
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'win': total_reward > 0,
            'epsilon': self.epsilon
        }
    
    def evaluate(self, env: MinesRLEnvironment, num_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate the agent's performance.

        Args:
            env: RL environment
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation statistics
        """
        original_epsilon = self.epsilon
        self.epsilon = 0.0  # No exploration during evaluation
        
        eval_rewards = []
        eval_lengths = []
        wins = 0
        
        for _ in range(num_episodes):
            state = env.reset()
            total_reward = 0.0
            steps = 0
            done = False
            
            while not done and steps < 1000:
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                
                action = self.select_action(state, valid_actions)
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            eval_rewards.append(total_reward)
            eval_lengths.append(steps)
            if total_reward > 0:
                wins += 1
        
        # Restore epsilon
        self.epsilon = original_epsilon
        
        # Calculate statistics
        stats = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_length': np.mean(eval_lengths),
            'win_rate': wins / num_episodes,
            'max_reward': np.max(eval_rewards),
            'min_reward': np.min(eval_rewards)
        }
        
        # Log evaluation metrics
        self.writer.add_scalar('Evaluation/AvgReward', stats['avg_reward'], len(self.episode_rewards))
        self.writer.add_scalar('Evaluation/WinRate', stats['win_rate'], len(self.episode_rewards))
        
        return stats
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'performance_history': list(self.performance_history)
        }, filepath)
        
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint['episode_rewards']
        self.performance_history = deque(checkpoint['performance_history'], maxlen=1000)
        
        logging.info(f"Model loaded from {filepath}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        
        return {
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards),
            'recent_avg_reward': np.mean(recent_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'win_rate': sum(1 for r in self.episode_rewards if r > 0) / len(self.episode_rewards),
            'recent_win_rate': sum(1 for r in recent_rewards if r > 0) / len(recent_rewards),
            'avg_episode_length': np.mean(self.episode_lengths),
            'current_epsilon': self.epsilon,
            'training_steps': self.training_step,
            'memory_size': len(self.memory.buffer)
        }
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


def train_drl_agent(num_episodes: int = 10000,
                   board_size: int = 5,
                   mine_count: int = 3,
                   strategy_type: str = 'analytical',
                   save_frequency: int = 1000,
                   eval_frequency: int = 500) -> DRLAgent:
    """
    Train a DRL agent for the Mines Game.
    
    Args:
        num_episodes: Number of training episodes
        board_size: Size of the game board
        mine_count: Number of mines
        strategy_type: Type of strategy to optimize for
        save_frequency: How often to save the model
        eval_frequency: How often to evaluate the agent
        
    Returns:
        Trained DRL agent
    """
    # Create environment
    env = create_rl_environment(
        board_size=board_size,
        mine_count=mine_count,
        strategy_type=strategy_type
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DRLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=100000,
        target_update_freq=1000
    )
    
    logging.info(f"Starting DRL training for {num_episodes} episodes")
    logging.info(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Training loop
    for episode in range(num_episodes):
        # Train one episode
        episode_stats = agent.train_episode(env)
        
        # Log progress
        if episode % 100 == 0:
            logging.info(f"Episode {episode}: Reward = {episode_stats['total_reward']:.2f}, "
                        f"Steps = {episode_stats['steps']}, Epsilon = {agent.epsilon:.3f}")
        
        # Evaluate agent
        if episode % eval_frequency == 0 and episode > 0:
            eval_stats = agent.evaluate(env, num_episodes=100)
            logging.info(f"Evaluation at episode {episode}: "
                        f"Avg Reward = {eval_stats['avg_reward']:.2f}, "
                        f"Win Rate = {eval_stats['win_rate']:.2%}")
        
        # Save model
        if episode % save_frequency == 0 and episode > 0:
            model_path = f"models/drl_agent_episode_{episode}.pth"
            os.makedirs("models", exist_ok=True)
            agent.save_model(model_path)
    
    # Final evaluation
    final_eval = agent.evaluate(env, num_episodes=1000)
    logging.info(f"Final evaluation: {final_eval}")
    
    # Save final model
    agent.save_model("models/drl_agent_final.pth")
    
    return agent


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('drl_training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Train DRL agent
    agent = train_drl_agent(
        num_episodes=50000,  # 50k episodes as requested
        board_size=5,
        mine_count=3,
        strategy_type='analytical',
        save_frequency=5000,
        eval_frequency=2500
    )
    
    # Print final performance summary
    performance = agent.get_performance_summary()
    print("\n" + "="*50)
    print("DRL AGENT TRAINING COMPLETE")
    print("="*50)
    print(f"Total Episodes: {performance['total_episodes']}")
    print(f"Average Reward: {performance['avg_reward']:.2f}")
    print(f"Recent Average Reward: {performance['recent_avg_reward']:.2f}")
    print(f"Win Rate: {performance['win_rate']:.2%}")
    print(f"Recent Win Rate: {performance['recent_win_rate']:.2%}")
    print(f"Max Reward: {performance['max_reward']:.2f}")
    print(f"Training Steps: {performance['training_steps']}")
    print("="*50)
    
    agent.close()