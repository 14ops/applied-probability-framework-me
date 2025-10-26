"""
Adaptive Learning Strategy Base Class

This module provides an enhanced strategy base class with matrix-based evolution
and learning capabilities. Strategies that inherit from this can automatically
learn and adapt over time.
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

from core.base_strategy import BaseStrategy
from core.evolution_matrix import QMatrix, ExperienceReplayBuffer, EvolutionMatrix


class AdaptiveLearningStrategy(BaseStrategy):
    """
    Enhanced strategy with matrix-based evolution and learning.
    
    Features:
    - Q-Learning for action-value estimation
    - Experience replay for sample efficiency
    - Parameter evolution via genetic algorithms
    - Automatic learning from gameplay
    - Persistence across sessions
    
    Subclasses can:
    1. Use Q-learning for pure learned behavior
    2. Blend Q-learning with heuristics
    3. Evolve strategy parameters automatically
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        use_q_learning: bool = True,
        use_experience_replay: bool = True,
        use_parameter_evolution: bool = False,
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Initialize Adaptive Learning Strategy.
        
        Args:
            name: Strategy name
            config: Configuration dictionary
            use_q_learning: Enable Q-learning
            use_experience_replay: Enable experience replay
            use_parameter_evolution: Enable parameter evolution
            parameter_ranges: Ranges for evolvable parameters
        """
        super().__init__(name, config)
        
        # Learning configuration
        self.use_q_learning = use_q_learning
        self.use_experience_replay = use_experience_replay
        self.use_parameter_evolution = use_parameter_evolution
        
        # Q-Learning Matrix
        if self.use_q_learning:
            self.q_matrix = QMatrix(
                learning_rate=config.get('learning_rate', 0.1) if config else 0.1,
                discount_factor=config.get('discount_factor', 0.95) if config else 0.95,
                epsilon=config.get('epsilon', 0.1) if config else 0.1,
                epsilon_decay=config.get('epsilon_decay', 0.995) if config else 0.995,
                use_double_q=config.get('use_double_q', True) if config else True
            )
        else:
            self.q_matrix = None
        
        # Experience Replay Buffer
        if self.use_experience_replay:
            self.replay_buffer = ExperienceReplayBuffer(
                max_size=config.get('replay_buffer_size', 10000) if config else 10000,
                prioritized=config.get('prioritized_replay', True) if config else True
            )
            self.replay_batch_size = config.get('replay_batch_size', 32) if config else 32
            self.replay_frequency = config.get('replay_frequency', 4) if config else 4
        else:
            self.replay_buffer = None
        
        # Parameter Evolution
        if self.use_parameter_evolution and parameter_ranges:
            self.evolution_matrix = EvolutionMatrix(
                parameter_ranges=parameter_ranges,
                population_size=config.get('population_size', 20) if config else 20,
                mutation_rate=config.get('mutation_rate', 0.1) if config else 0.1
            )
            self.current_individual_idx = 0
            self.episode_count = 0
            self.evolution_frequency = config.get('evolution_frequency', 10) if config else 10
        else:
            self.evolution_matrix = None
        
        # Learning state
        self.last_state = None
        self.last_action = None
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        
        # Blending parameter (0 = pure heuristic, 1 = pure Q-learning)
        self.q_learning_weight = config.get('q_learning_weight', 0.5) if config else 0.5
        
    def select_action_with_q_learning(
        self,
        state: Any,
        valid_actions: List[Any],
        exploration: bool = True
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Select action using Q-learning (with optional exploration).
        
        Args:
            state: Current state
            valid_actions: List of valid actions
            exploration: Whether to use epsilon-greedy exploration
            
        Returns:
            Tuple of (action, info_dict)
        """
        if not self.q_matrix:
            raise RuntimeError("Q-learning not enabled")
        
        if exploration:
            action, is_exploration = self.q_matrix.select_action_epsilon_greedy(
                state, valid_actions
            )
        else:
            state_hash = self.q_matrix.get_state_hash(state)
            action, q_value = self.q_matrix.get_best_action(state_hash, valid_actions)
            is_exploration = False
        
        info = {
            'is_exploration': is_exploration,
            'epsilon': self.q_matrix.epsilon if self.q_matrix else 0.0,
            'q_value': self.q_matrix.get_state_value(state, valid_actions) if self.q_matrix else 0.0
        }
        
        return action, info
    
    def select_action_heuristic(self, state: Any, valid_actions: List[Any]) -> Any:
        """
        Select action using heuristic/rule-based logic.
        
        This should be overridden by subclasses to implement their strategy.
        Default implementation returns random action.
        
        Args:
            state: Current state
            valid_actions: List of valid actions
            
        Returns:
            Selected action
        """
        return np.random.choice(valid_actions) if valid_actions else None
    
    def select_action(self, state: Any, valid_actions: Any) -> Any:
        """
        Select action by blending Q-learning and heuristic approaches.
        
        Args:
            state: Current state
            valid_actions: List of valid actions
            
        Returns:
            Selected action
        """
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Store state for update
        self.last_state = state
        
        # Pure Q-learning
        if self.use_q_learning and self.q_learning_weight >= 1.0:
            action, _ = self.select_action_with_q_learning(state, valid_actions)
            self.last_action = action
            return action
        
        # Pure heuristic
        if not self.use_q_learning or self.q_learning_weight <= 0.0:
            action = self.select_action_heuristic(state, valid_actions)
            self.last_action = action
            return action
        
        # Blended approach
        if np.random.random() < self.q_learning_weight:
            # Use Q-learning
            action, _ = self.select_action_with_q_learning(state, valid_actions)
        else:
            # Use heuristic
            action = self.select_action_heuristic(state, valid_actions)
        
        self.last_action = action
        return action
    
    def update(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool
    ) -> None:
        """
        Update learning matrices based on experience.
        
        Args:
            state: State where action was taken
            action: Action that was taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        super().update(state, action, reward, next_state, done)
        
        # Track episode reward
        self.current_episode_reward += reward
        
        # Update Q-Learning
        if self.use_q_learning and self.q_matrix:
            # Get valid next actions (if not done)
            valid_next_actions = None
            if not done and hasattr(next_state, 'get') and isinstance(next_state, dict):
                # Try to extract valid actions from state (game-specific)
                valid_next_actions = next_state.get('valid_actions', None)
            
            # Update Q-value
            td_error = self.q_matrix.update_q_value(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                valid_next_actions=valid_next_actions
            )
            
            # Add to experience replay buffer
            if self.use_experience_replay and self.replay_buffer:
                state_hash = self.q_matrix.get_state_hash(state)
                next_state_hash = self.q_matrix.get_state_hash(next_state)
                
                # Priority is TD error magnitude
                priority = abs(td_error) if td_error else 1.0
                
                self.replay_buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    state_hash=state_hash,
                    next_state_hash=next_state_hash,
                    priority=priority
                )
        
        # Replay experiences periodically
        if (self.use_experience_replay and 
            self.replay_buffer and 
            len(self.replay_buffer) >= self.replay_batch_size and
            self.q_matrix.updates % self.replay_frequency == 0):
            self._replay_experiences()
        
        # Handle episode end
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
            self.episode_count += 1
            
            # Evolve parameters periodically
            if (self.use_parameter_evolution and 
                self.evolution_matrix and
                self.episode_count % self.evolution_frequency == 0):
                self._evolve_parameters()
    
    def _replay_experiences(self) -> None:
        """Replay batch of experiences for learning."""
        if not self.replay_buffer or not self.q_matrix:
            return
        
        # Sample batch
        experiences, weights, indices = self.replay_buffer.sample(self.replay_batch_size)
        
        # Update Q-values for each experience
        td_errors = []
        for exp, weight in zip(experiences, weights):
            # Note: We can't easily get valid_next_actions from stored experiences
            # So we use None (which assumes all actions valid)
            td_error = self.q_matrix.update_q_value(
                state={'_hash': exp.state_hash},  # Minimal state representation
                action=exp.action,
                reward=exp.reward * weight,  # Weight by importance
                next_state={'_hash': exp.next_state_hash},
                done=exp.done,
                valid_next_actions=None
            )
            td_errors.append(abs(td_error))
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, np.array(td_errors))
    
    def _evolve_parameters(self) -> None:
        """Evolve strategy parameters using genetic algorithm."""
        if not self.evolution_matrix:
            return
        
        # Evaluate current individual's fitness
        fitness = np.mean(self.episode_rewards[-self.evolution_frequency:])
        self.evolution_matrix.evaluate_fitness(self.current_individual_idx, fitness)
        
        # Move to next individual in population
        self.current_individual_idx += 1
        
        # If we've evaluated all individuals, evolve to next generation
        if self.current_individual_idx >= self.evolution_matrix.population_size:
            best_params = self.evolution_matrix.evolve()
            self._apply_evolved_parameters(best_params)
            self.current_individual_idx = 0
            
            print(f"Generation {self.evolution_matrix.generation}: "
                  f"Best Fitness = {self.evolution_matrix.best_fitness_history[-1]:.4f}, "
                  f"Avg Fitness = {self.evolution_matrix.avg_fitness_history[-1]:.4f}")
    
    def _apply_evolved_parameters(self, params: Dict[str, float]) -> None:
        """
        Apply evolved parameters to strategy.
        
        This should be overridden by subclasses to handle their specific parameters.
        
        Args:
            params: Evolved parameter values
        """
        # Update config with evolved parameters
        for key, value in params.items():
            self.config[key] = value
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive learning statistics.
        
        Returns:
            Dictionary of learning metrics
        """
        stats = super().get_statistics()
        
        if self.use_q_learning and self.q_matrix:
            stats['q_learning'] = {
                'epsilon': self.q_matrix.epsilon,
                'updates': self.q_matrix.updates,
                'q_table_size': len(self.q_matrix.q_table_a),
                'total_learned_reward': self.q_matrix.total_reward,
            }
        
        if self.use_experience_replay and self.replay_buffer:
            stats['experience_replay'] = {
                'buffer_size': len(self.replay_buffer),
                'total_experiences': self.replay_buffer.total_experiences,
            }
        
        if self.use_parameter_evolution and self.evolution_matrix:
            stats['evolution'] = {
                'generation': self.evolution_matrix.generation,
                'best_fitness': (self.evolution_matrix.best_fitness_history[-1] 
                               if self.evolution_matrix.best_fitness_history else 0.0),
                'avg_fitness': (self.evolution_matrix.avg_fitness_history[-1]
                              if self.evolution_matrix.avg_fitness_history else 0.0),
            }
        
        stats['episode_rewards'] = {
            'recent_mean': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0,
            'recent_std': np.std(self.episode_rewards[-10:]) if self.episode_rewards else 0.0,
            'total_episodes': len(self.episode_rewards),
        }
        
        return stats
    
    def save_learning_state(self, directory: str) -> None:
        """
        Save all learning matrices to disk.
        
        Args:
            directory: Directory to save to
        """
        from pathlib import Path
        
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Q-matrix
        if self.use_q_learning and self.q_matrix:
            self.q_matrix.save(str(save_dir / 'q_matrix.json'))
        
        # Save replay buffer
        if self.use_experience_replay and self.replay_buffer:
            self.replay_buffer.save(str(save_dir / 'replay_buffer.json'))
        
        # Save evolution matrix
        if self.use_parameter_evolution and self.evolution_matrix:
            self.evolution_matrix.save(str(save_dir / 'evolution_matrix.json'))
        
        # Save strategy config and stats
        self.save(str(save_dir / 'strategy.json'))
        
        print(f"Saved learning state to {directory}")
    
    def load_learning_state(self, directory: str) -> None:
        """
        Load all learning matrices from disk.
        
        Args:
            directory: Directory to load from
        """
        from pathlib import Path
        
        load_dir = Path(directory)
        
        # Load Q-matrix
        if self.use_q_learning and self.q_matrix:
            q_path = load_dir / 'q_matrix.json'
            if q_path.exists():
                self.q_matrix.load(str(q_path))
        
        # Load replay buffer
        if self.use_experience_replay and self.replay_buffer:
            replay_path = load_dir / 'replay_buffer.json'
            if replay_path.exists():
                self.replay_buffer.load(str(replay_path))
        
        # Load evolution matrix
        if self.use_parameter_evolution and self.evolution_matrix:
            evo_path = load_dir / 'evolution_matrix.json'
            if evo_path.exists():
                self.evolution_matrix.load(str(evo_path))
        
        # Load strategy config and stats
        strat_path = load_dir / 'strategy.json'
        if strat_path.exists():
            self.load(str(strat_path))
        
        print(f"Loaded learning state from {directory}")
    
    def reset(self) -> None:
        """Reset episode-specific state."""
        super().reset()
        self.last_state = None
        self.last_action = None

