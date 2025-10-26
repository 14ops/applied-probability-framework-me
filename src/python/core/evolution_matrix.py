"""
Evolution Matrix System for Adaptive AI Learning

This module implements matrix-based learning systems that allow strategies
to evolve and improve over time through:
- Q-Learning with state-action value matrices
- Experience replay for better sample efficiency
- Genetic algorithm-style parameter evolution
- Adaptive learning rate scheduling
"""

import numpy as np
import json
from typing import Any, Dict, List, Tuple, Optional, Deque
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state_hash: str
    action: Any
    reward: float
    next_state_hash: str
    done: bool
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


class QMatrix:
    """
    Q-Learning Matrix for state-action value estimation.
    
    Features:
    - Dynamic state space using hashing
    - Eligibility traces for temporal credit assignment
    - Adaptive learning rate
    - Exploration-exploitation balance with epsilon-greedy
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        use_double_q: bool = True
    ):
        """
        Initialize Q-Learning Matrix.
        
        Args:
            learning_rate: How quickly to update Q-values
            discount_factor: Future reward discount (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            use_double_q: Use Double Q-Learning to reduce overestimation
        """
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.use_double_q = use_double_q
        
        # Q-matrices (two for Double Q-Learning)
        self.q_table_a: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.q_table_b: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Visit counts for adaptive learning
        self.visit_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Eligibility traces
        self.traces: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.trace_decay = 0.9
        
        # Statistics
        self.updates = 0
        self.total_reward = 0.0
        
    def get_state_hash(self, state: Any) -> str:
        """
        Generate hash for state to use as key in Q-table.
        
        Args:
            state: State object (dict, array, etc.)
            
        Returns:
            String hash of state
        """
        if isinstance(state, dict):
            # Sort keys for consistent hashing
            state_str = json.dumps(state, sort_keys=True)
        elif isinstance(state, np.ndarray):
            state_str = state.tobytes().hex()
        else:
            state_str = str(state)
            
        return hashlib.md5(state_str.encode()).hexdigest()[:16]
    
    def get_action_key(self, action: Any) -> str:
        """Convert action to string key."""
        if isinstance(action, tuple):
            return f"{action[0]}_{action[1]}"
        return str(action)
    
    def get_q_value(self, state_hash: str, action_key: str, table: str = 'a') -> float:
        """
        Get Q-value for state-action pair.
        
        Args:
            state_hash: Hashed state
            action_key: Action key
            table: Which Q-table to use ('a' or 'b')
            
        Returns:
            Q-value
        """
        q_table = self.q_table_a if table == 'a' else self.q_table_b
        return q_table[state_hash][action_key]
    
    def get_best_action(self, state_hash: str, valid_actions: List[Any]) -> Tuple[Any, float]:
        """
        Get best action according to Q-values.
        
        Args:
            state_hash: Hashed state
            valid_actions: List of valid actions
            
        Returns:
            Tuple of (best_action, max_q_value)
        """
        if not valid_actions:
            raise ValueError("No valid actions provided")
        
        best_action = None
        max_q = float('-inf')
        
        for action in valid_actions:
            action_key = self.get_action_key(action)
            
            # Average Q-values from both tables if using Double Q-Learning
            if self.use_double_q:
                q_a = self.get_q_value(state_hash, action_key, 'a')
                q_b = self.get_q_value(state_hash, action_key, 'b')
                q_value = (q_a + q_b) / 2.0
            else:
                q_value = self.get_q_value(state_hash, action_key, 'a')
            
            if q_value > max_q:
                max_q = q_value
                best_action = action
        
        return best_action, max_q
    
    def select_action_epsilon_greedy(
        self, 
        state: Any, 
        valid_actions: List[Any]
    ) -> Tuple[Any, bool]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            valid_actions: List of valid actions
            
        Returns:
            Tuple of (selected_action, is_exploration)
        """
        state_hash = self.get_state_hash(state)
        
        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions), True
        
        # Exploitation
        best_action, _ = self.get_best_action(state_hash, valid_actions)
        return best_action, False
    
    def update_q_value(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        valid_next_actions: Optional[List[Any]] = None
    ) -> float:
        """
        Update Q-value using Q-Learning or Double Q-Learning.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            valid_next_actions: Valid actions in next state
            
        Returns:
            TD error (temporal difference error)
        """
        state_hash = self.get_state_hash(state)
        next_state_hash = self.get_state_hash(next_state)
        action_key = self.get_action_key(action)
        
        # Update visit count
        self.visit_counts[state_hash][action_key] += 1
        
        # Adaptive learning rate based on visit count
        visits = self.visit_counts[state_hash][action_key]
        adaptive_alpha = self.alpha / (1.0 + np.log(visits))
        
        # Double Q-Learning update
        if self.use_double_q:
            # Randomly choose which Q-table to update
            if np.random.random() < 0.5:
                update_table = self.q_table_a
                reference_table = self.q_table_b
                table_name = 'a'
            else:
                update_table = self.q_table_b
                reference_table = self.q_table_a
                table_name = 'b'
            
            current_q = update_table[state_hash][action_key]
            
            # Calculate target
            if done or not valid_next_actions:
                target = reward
            else:
                # Find best action using update_table
                best_next_action, _ = self.get_best_action(next_state_hash, valid_next_actions)
                best_next_action_key = self.get_action_key(best_next_action)
                
                # Evaluate using reference_table (Double Q-Learning)
                next_q = reference_table[next_state_hash][best_next_action_key]
                target = reward + self.gamma * next_q
            
            # TD error
            td_error = target - current_q
            
            # Update Q-value
            update_table[state_hash][action_key] = current_q + adaptive_alpha * td_error
        else:
            # Standard Q-Learning
            current_q = self.q_table_a[state_hash][action_key]
            
            if done or not valid_next_actions:
                target = reward
            else:
                _, max_next_q = self.get_best_action(next_state_hash, valid_next_actions)
                target = reward + self.gamma * max_next_q
            
            td_error = target - current_q
            self.q_table_a[state_hash][action_key] = current_q + adaptive_alpha * td_error
        
        # Update statistics
        self.updates += 1
        self.total_reward += reward
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return td_error
    
    def get_state_value(self, state: Any, valid_actions: List[Any]) -> float:
        """
        Get state value (maximum Q-value over actions).
        
        Args:
            state: State to evaluate
            valid_actions: Valid actions in state
            
        Returns:
            State value
        """
        if not valid_actions:
            return 0.0
        
        state_hash = self.get_state_hash(state)
        _, max_q = self.get_best_action(state_hash, valid_actions)
        return max_q
    
    def save(self, filepath: str) -> None:
        """Save Q-matrix to disk."""
        data = {
            'q_table_a': {k: dict(v) for k, v in self.q_table_a.items()},
            'q_table_b': {k: dict(v) for k, v in self.q_table_b.items()},
            'visit_counts': {k: dict(v) for k, v in self.visit_counts.items()},
            'epsilon': self.epsilon,
            'updates': self.updates,
            'total_reward': self.total_reward,
            'params': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'use_double_q': self.use_double_q,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load Q-matrix from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.q_table_a = defaultdict(lambda: defaultdict(float), 
                                     {k: defaultdict(float, v) for k, v in data['q_table_a'].items()})
        self.q_table_b = defaultdict(lambda: defaultdict(float),
                                     {k: defaultdict(float, v) for k, v in data['q_table_b'].items()})
        self.visit_counts = defaultdict(lambda: defaultdict(int),
                                       {k: defaultdict(int, v) for k, v in data['visit_counts'].items()})
        self.epsilon = data['epsilon']
        self.updates = data['updates']
        self.total_reward = data['total_reward']


class ExperienceReplayBuffer:
    """
    Experience Replay Buffer for learning from past experiences.
    
    Features:
    - Prioritized experience replay
    - Maximum buffer size with automatic eviction
    - Batch sampling for training
    - Experience importance weighting
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        prioritized: bool = True,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001
    ):
        """
        Initialize Experience Replay Buffer.
        
        Args:
            max_size: Maximum number of experiences to store
            prioritized: Use prioritized experience replay
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
            beta_increment: Beta increment per sample
        """
        self.max_size = max_size
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer: Deque[Experience] = deque(maxlen=max_size)
        self.priorities: Deque[float] = deque(maxlen=max_size)
        
        # Statistics
        self.total_experiences = 0
        
    def add(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        state_hash: Optional[str] = None,
        next_state_hash: Optional[str] = None,
        priority: Optional[float] = None
    ) -> None:
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            state_hash: Pre-computed state hash (optional)
            next_state_hash: Pre-computed next state hash (optional)
            priority: Priority for prioritized replay (optional)
        """
        import time
        
        # Generate hashes if not provided
        if state_hash is None:
            state_hash = hashlib.md5(str(state).encode()).hexdigest()[:16]
        if next_state_hash is None:
            next_state_hash = hashlib.md5(str(next_state).encode()).hexdigest()[:16]
        
        experience = Experience(
            state_hash=state_hash,
            action=action,
            reward=reward,
            next_state_hash=next_state_hash,
            done=done,
            timestamp=time.time()
        )
        
        self.buffer.append(experience)
        
        # Set priority (default to max priority for new experiences)
        if priority is None:
            if self.prioritized and len(self.priorities) > 0:
                priority = max(self.priorities)
            else:
                priority = 1.0
        
        self.priorities.append(priority)
        self.total_experiences += 1
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (experiences, importance_weights, indices)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if self.prioritized:
            # Compute probabilities from priorities
            priorities = np.array(self.priorities)
            probs = priorities ** self.alpha
            probs /= probs.sum()
            
            # Sample indices
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
            
            # Compute importance sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            weights /= weights.max()  # Normalize
            
            # Increment beta
            self.beta = min(1.0, self.beta + self.beta_increment)
            
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            weights = np.ones(batch_size)
        
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priorities
        """
        if not self.prioritized:
            return
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # Small constant to avoid zero priority
    
    def __len__(self) -> int:
        """Get number of experiences in buffer."""
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()
        self.priorities.clear()


class EvolutionMatrix:
    """
    Genetic Algorithm-style evolution of strategy parameters.
    
    Features:
    - Population-based parameter evolution
    - Crossover and mutation operators
    - Fitness-based selection
    - Adaptive mutation rates
    """
    
    def __init__(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism_ratio: float = 0.2
    ):
        """
        Initialize Evolution Matrix.
        
        Args:
            parameter_ranges: Dict mapping parameter names to (min, max) ranges
            population_size: Number of parameter sets in population
            mutation_rate: Probability of parameter mutation
            crossover_rate: Probability of crossover
            elitism_ratio: Ratio of top performers to preserve
        """
        self.parameter_ranges = parameter_ranges
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        
        # Initialize population
        self.population: List[Dict[str, float]] = []
        self.fitness_scores: List[float] = []
        
        self._initialize_population()
        
        # Statistics
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
    
    def _initialize_population(self) -> None:
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            individual = {}
            for param, (min_val, max_val) in self.parameter_ranges.items():
                individual[param] = np.random.uniform(min_val, max_val)
            self.population.append(individual)
        
        self.fitness_scores = [0.0] * self.population_size
    
    def evaluate_fitness(self, individual_idx: int, fitness: float) -> None:
        """
        Record fitness score for an individual.
        
        Args:
            individual_idx: Index of individual in population
            fitness: Fitness score (higher is better)
        """
        if 0 <= individual_idx < self.population_size:
            self.fitness_scores[individual_idx] = fitness
    
    def evolve(self) -> Dict[str, float]:
        """
        Evolve population to next generation.
        
        Returns:
            Best individual from current generation
        """
        # Sort population by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]
        sorted_fitness = [self.fitness_scores[i] for i in sorted_indices]
        
        # Record statistics
        self.best_fitness_history.append(sorted_fitness[0])
        self.avg_fitness_history.append(np.mean(sorted_fitness))
        
        # Elitism: preserve top performers
        elite_count = int(self.population_size * self.elitism_ratio)
        new_population = sorted_population[:elite_count].copy()
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection (tournament selection)
            parent1 = self._tournament_select(sorted_population, sorted_fitness)
            parent2 = self._tournament_select(sorted_population, sorted_fitness)
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            child = self._mutate(child)
            
            new_population.append(child)
        
        self.population = new_population[:self.population_size]
        self.fitness_scores = [0.0] * self.population_size
        self.generation += 1
        
        return sorted_population[0]
    
    def _tournament_select(
        self, 
        population: List[Dict[str, float]], 
        fitness: List[float],
        tournament_size: int = 3
    ) -> Dict[str, float]:
        """
        Tournament selection.
        
        Args:
            population: Population to select from
            fitness: Fitness scores
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected individual
        """
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(
        self, 
        parent1: Dict[str, float], 
        parent2: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Uniform crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child individual
        """
        child = {}
        for param in self.parameter_ranges.keys():
            # Randomly choose parameter from either parent
            if np.random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child
    
    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """
        Gaussian mutation.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        for param, (min_val, max_val) in self.parameter_ranges.items():
            if np.random.random() < self.mutation_rate:
                # Gaussian mutation
                range_size = max_val - min_val
                mutation = np.random.normal(0, range_size * 0.1)
                mutated[param] = np.clip(
                    mutated[param] + mutation,
                    min_val,
                    max_val
                )
        return mutated
    
    def get_best_individual(self) -> Dict[str, float]:
        """Get best individual from current population."""
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx].copy()
    
    def save(self, filepath: str) -> None:
        """Save evolution matrix to disk."""
        data = {
            'population': self.population,
            'fitness_scores': self.fitness_scores,
            'generation': self.generation,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'params': {
                'parameter_ranges': self.parameter_ranges,
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elitism_ratio': self.elitism_ratio,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load evolution matrix from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.population = data['population']
        self.fitness_scores = data['fitness_scores']
        self.generation = data['generation']
        self.best_fitness_history = data['best_fitness_history']
        self.avg_fitness_history = data['avg_fitness_history']

