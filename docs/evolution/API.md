# Matrix Evolution API Reference

Complete API documentation for the matrix-based evolution system.

## Table of Contents

- [QMatrix](#qmatrix)
- [ExperienceReplayBuffer](#experiencereplaybuffer)
- [EvolutionMatrix](#evolutionmatrix)
- [AdaptiveLearningStrategy](#adaptivelearingstrategy)

---

## QMatrix

Q-Learning Matrix for state-action value estimation.

### Constructor

```python
QMatrix(
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon: float = 0.1,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    use_double_q: bool = True
)
```

**Parameters:**
- `learning_rate`: How quickly to update Q-values (0.0-1.0)
- `discount_factor`: Future reward discount factor γ (0.0-1.0)
- `epsilon`: Initial exploration rate (0.0-1.0)
- `epsilon_decay`: Decay rate for epsilon per update (0.0-1.0)
- `epsilon_min`: Minimum epsilon value (0.0-1.0)
- `use_double_q`: Whether to use Double Q-Learning

### Methods

#### `get_state_hash(state: Any) -> str`

Generate hash for state to use as key in Q-table.

**Parameters:**
- `state`: State object (dict, array, etc.)

**Returns:**
- String hash of state

---

#### `select_action_epsilon_greedy(state: Any, valid_actions: List[Any]) -> Tuple[Any, bool]`

Select action using epsilon-greedy policy.

**Parameters:**
- `state`: Current state
- `valid_actions`: List of valid actions

**Returns:**
- Tuple of (selected_action, is_exploration)

---

#### `update_q_value(state: Any, action: Any, reward: float, next_state: Any, done: bool, valid_next_actions: Optional[List[Any]] = None) -> float`

Update Q-value using Q-Learning or Double Q-Learning.

**Parameters:**
- `state`: Current state
- `action`: Action taken
- `reward`: Reward received
- `next_state`: Next state
- `done`: Whether episode ended
- `valid_next_actions`: Valid actions in next state (optional)

**Returns:**
- TD error (temporal difference error)

---

#### `get_state_value(state: Any, valid_actions: List[Any]) -> float`

Get state value (maximum Q-value over actions).

**Parameters:**
- `state`: State to evaluate
- `valid_actions`: Valid actions in state

**Returns:**
- State value

---

#### `save(filepath: str) -> None`

Save Q-matrix to disk.

**Parameters:**
- `filepath`: Path to save file

---

#### `load(filepath: str) -> None`

Load Q-matrix from disk.

**Parameters:**
- `filepath`: Path to load from

---

## ExperienceReplayBuffer

Experience Replay Buffer for learning from past experiences.

### Constructor

```python
ExperienceReplayBuffer(
    max_size: int = 10000,
    prioritized: bool = True,
    alpha: float = 0.6,
    beta: float = 0.4,
    beta_increment: float = 0.001
)
```

**Parameters:**
- `max_size`: Maximum number of experiences to store
- `prioritized`: Use prioritized experience replay
- `alpha`: Priority exponent (0 = uniform, 1 = full prioritization)
- `beta`: Importance sampling exponent
- `beta_increment`: Beta increment per sample

### Methods

#### `add(state: Any, action: Any, reward: float, next_state: Any, done: bool, state_hash: Optional[str] = None, next_state_hash: Optional[str] = None, priority: Optional[float] = None) -> None`

Add experience to buffer.

**Parameters:**
- `state`: Current state
- `action`: Action taken
- `reward`: Reward received
- `next_state`: Next state
- `done`: Whether episode ended
- `state_hash`: Pre-computed state hash (optional)
- `next_state_hash`: Pre-computed next state hash (optional)
- `priority`: Priority for prioritized replay (optional)

---

#### `sample(batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]`

Sample batch of experiences.

**Parameters:**
- `batch_size`: Number of experiences to sample

**Returns:**
- Tuple of (experiences, importance_weights, indices)

---

#### `update_priorities(indices: np.ndarray, priorities: np.ndarray) -> None`

Update priorities for sampled experiences.

**Parameters:**
- `indices`: Indices of experiences to update
- `priorities`: New priorities

---

#### `clear() -> None`

Clear all experiences from buffer.

---

## EvolutionMatrix

Genetic Algorithm-style evolution of strategy parameters.

### Constructor

```python
EvolutionMatrix(
    parameter_ranges: Dict[str, Tuple[float, float]],
    population_size: int = 20,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    elitism_ratio: float = 0.2
)
```

**Parameters:**
- `parameter_ranges`: Dict mapping parameter names to (min, max) ranges
- `population_size`: Number of parameter sets in population
- `mutation_rate`: Probability of parameter mutation (0.0-1.0)
- `crossover_rate`: Probability of crossover (0.0-1.0)
- `elitism_ratio`: Ratio of top performers to preserve (0.0-1.0)

### Methods

#### `evaluate_fitness(individual_idx: int, fitness: float) -> None`

Record fitness score for an individual.

**Parameters:**
- `individual_idx`: Index of individual in population
- `fitness`: Fitness score (higher is better)

---

#### `evolve() -> Dict[str, float]`

Evolve population to next generation.

**Returns:**
- Best individual from current generation

---

#### `get_best_individual() -> Dict[str, float]`

Get best individual from current population.

**Returns:**
- Dictionary of best parameter values

---

#### `save(filepath: str) -> None`

Save evolution matrix to disk.

**Parameters:**
- `filepath`: Path to save file

---

#### `load(filepath: str) -> None`

Load evolution matrix from disk.

**Parameters:**
- `filepath`: Path to load from

---

## AdaptiveLearningStrategy

Enhanced strategy with matrix-based evolution and learning.

### Constructor

```python
AdaptiveLearningStrategy(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    use_q_learning: bool = True,
    use_experience_replay: bool = True,
    use_parameter_evolution: bool = False,
    parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None
)
```

**Parameters:**
- `name`: Strategy name
- `config`: Configuration dictionary
- `use_q_learning`: Enable Q-learning
- `use_experience_replay`: Enable experience replay
- `use_parameter_evolution`: Enable parameter evolution
- `parameter_ranges`: Ranges for evolvable parameters

### Methods

#### `select_action_with_q_learning(state: Any, valid_actions: List[Any], exploration: bool = True) -> Tuple[Any, Dict[str, Any]]`

Select action using Q-learning (with optional exploration).

**Parameters:**
- `state`: Current state
- `valid_actions`: List of valid actions
- `exploration`: Whether to use epsilon-greedy exploration

**Returns:**
- Tuple of (action, info_dict)

---

#### `select_action_heuristic(state: Any, valid_actions: List[Any]) -> Any`

Select action using heuristic/rule-based logic.

**This should be overridden by subclasses.**

**Parameters:**
- `state`: Current state
- `valid_actions`: List of valid actions

**Returns:**
- Selected action

---

#### `select_action(state: Any, valid_actions: Any) -> Any`

Select action by blending Q-learning and heuristic approaches.

**Parameters:**
- `state`: Current state
- `valid_actions`: List of valid actions

**Returns:**
- Selected action

---

#### `update(state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None`

Update learning matrices based on experience.

**Parameters:**
- `state`: State where action was taken
- `action`: Action that was taken
- `reward`: Reward received
- `next_state`: Resulting state
- `done`: Whether episode ended

---

#### `get_learning_statistics() -> Dict[str, Any]`

Get comprehensive learning statistics.

**Returns:**
- Dictionary of learning metrics including:
  - `q_learning`: Q-learning stats (epsilon, updates, table size)
  - `experience_replay`: Replay buffer stats
  - `evolution`: Evolution stats (generation, fitness)
  - `episode_rewards`: Episode reward statistics

---

#### `save_learning_state(directory: str) -> None`

Save all learning matrices to disk.

**Parameters:**
- `directory`: Directory to save to

---

#### `load_learning_state(directory: str) -> None`

Load all learning matrices from disk.

**Parameters:**
- `directory`: Directory to load from

---

#### `_apply_evolved_parameters(params: Dict[str, float]) -> None`

Apply evolved parameters to strategy.

**This should be overridden by subclasses to handle their specific parameters.**

**Parameters:**
- `params`: Evolved parameter values

---

## Configuration Dictionary

Complete configuration options for `AdaptiveLearningStrategy`:

```python
config = {
    # Q-Learning Parameters
    'learning_rate': 0.1,              # Learning rate α (0.0-1.0)
    'discount_factor': 0.95,           # Discount factor γ (0.0-1.0)
    'epsilon': 0.1,                    # Initial exploration rate (0.0-1.0)
    'epsilon_decay': 0.995,            # Epsilon decay per update (0.0-1.0)
    'epsilon_min': 0.01,               # Minimum epsilon (0.0-1.0)
    'use_double_q': True,              # Use Double Q-Learning
    'q_learning_weight': 0.5,          # Blend factor (0.0-1.0)
    
    # Experience Replay Parameters
    'replay_buffer_size': 10000,       # Max experiences to store
    'replay_batch_size': 32,           # Batch size for training
    'replay_frequency': 4,             # Replay every N updates
    'prioritized_replay': True,        # Use prioritized replay
    
    # Evolution Parameters
    'population_size': 20,             # Population size
    'mutation_rate': 0.1,              # Mutation probability (0.0-1.0)
    'crossover_rate': 0.7,             # Crossover probability (0.0-1.0)
    'elitism_ratio': 0.2,              # Elitism ratio (0.0-1.0)
    'evolution_frequency': 10,         # Evolve every N episodes
}
```

---

## Experience Data Class

```python
@dataclass
class Experience:
    state_hash: str           # Hashed state
    action: Any              # Action taken
    reward: float            # Reward received
    next_state_hash: str     # Hashed next state
    done: bool               # Episode terminated
    timestamp: float         # Time of experience
```

---

## Usage Examples

### Basic Q-Learning

```python
q_matrix = QMatrix(
    learning_rate=0.1,
    epsilon=0.2
)

# During gameplay
action, is_exploration = q_matrix.select_action_epsilon_greedy(state, valid_actions)
next_state, reward, done, _ = env.step(action)
td_error = q_matrix.update_q_value(state, action, reward, next_state, done, next_valid_actions)
```

### Experience Replay

```python
replay_buffer = ExperienceReplayBuffer(
    max_size=10000,
    prioritized=True
)

# Add experience
replay_buffer.add(state, action, reward, next_state, done)

# Sample and train
if len(replay_buffer) >= 32:
    experiences, weights, indices = replay_buffer.sample(32)
    # Train on experiences
    # ...
    # Update priorities
    replay_buffer.update_priorities(indices, new_priorities)
```

### Parameter Evolution

```python
evolution = EvolutionMatrix(
    parameter_ranges={
        'risk': (0.0, 1.0),
        'aggression': (0.0, 1.0),
    },
    population_size=20
)

# Evaluate individual
evolution.evaluate_fitness(individual_idx, fitness_score)

# Evolve after evaluating all individuals
best_params = evolution.evolve()
```

### Full Adaptive Strategy

```python
class MyStrategy(AdaptiveLearningStrategy):
    def __init__(self, config=None):
        super().__init__(
            name="My Strategy",
            config=config,
            use_q_learning=True,
            use_experience_replay=True,
            use_parameter_evolution=True,
            parameter_ranges={'risk': (0.0, 1.0)}
        )
    
    def select_action_heuristic(self, state, valid_actions):
        # Your heuristic logic
        return my_action
    
    def _apply_evolved_parameters(self, params):
        super()._apply_evolved_parameters(params)
        self.risk = params['risk']

# Use it
strategy = MyStrategy(config={'q_learning_weight': 0.5})
action = strategy.select_action(state, valid_actions)
strategy.update(state, action, reward, next_state, done)
```

---

## Performance Considerations

### Memory Usage

- **Q-Table Size**: Grows with unique states visited. Monitor with `len(q_matrix.q_table_a)`.
- **Replay Buffer**: Fixed size, automatically evicts old experiences.
- **Evolution Population**: Small (20-50 individuals typical).

### Computational Complexity

- **Q-Learning Update**: O(|actions|) per update
- **Experience Replay**: O(batch_size × |actions|) per replay
- **Evolution**: O(population_size × episode_length) per generation

### Optimization Tips

1. Use smaller replay buffers for faster games
2. Reduce population size for quick convergence
3. Increase replay frequency for sample efficiency
4. Use Double Q-Learning to reduce overestimation
5. Save learned models frequently to avoid re-learning

---

For more information, see the [AI Evolution Guide](AI_EVOLUTION_GUIDE.md).

