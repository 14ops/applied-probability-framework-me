# API Reference

Complete API documentation for the Applied Probability Framework.

## Table of Contents

- [Core Modules](#core-modules)
  - [base_simulator](#base_simulator)
  - [base_strategy](#base_strategy)
  - [base_estimator](#base_estimator)
  - [config](#config)
  - [plugin_system](#plugin_system)
  - [parallel_engine](#parallel_engine)
  - [reproducibility](#reproducibility)
- [CLI](#cli)
- [RL / Bayesian / Multi-Agent Modules (scaffold)](#rl--bayesian--multi-agent-modules-scaffold)

---

## Core Modules

### base_simulator

#### `BaseSimulator`

Abstract base class for all simulation environments.

```python
class BaseSimulator(ABC):
    def __init__(self, seed: Optional[int] = None, config: Optional[Dict[str, Any]] = None)
    
    @abstractmethod
    def reset(self) -> Any
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]
    
    @abstractmethod
    def get_valid_actions(self) -> List[Any]
    
    @abstractmethod
    def get_state_representation(self) -> np.ndarray
    
    def get_state(self) -> Any
    def set_seed(self, seed: int) -> None
    def get_metadata(self) -> Dict[str, Any]
```

**Parameters:**
- `seed` (int, optional): Random seed for reproducibility
- `config` (dict, optional): Configuration dictionary

**Methods:**

##### `reset() -> Any`
Reset simulator to initial state. Must be called before starting a new episode.

**Returns:**
- Initial state

##### `step(action) -> Tuple[Any, float, bool, Dict]`
Execute one simulation step.

**Parameters:**
- `action`: Action to execute

**Returns:**
- `next_state`: Resulting state
- `reward`: Immediate reward
- `done`: Whether episode terminated
- `info`: Additional information dictionary

##### `get_valid_actions() -> List[Any]`
Get list of valid actions in current state.

**Returns:**
- List of valid actions

##### `get_state_representation() -> np.ndarray`
Get numerical representation of current state for ML models.

**Returns:**
- NumPy array representing state

---

### base_strategy

#### `BaseStrategy`

Abstract base class for decision-making strategies.

```python
class BaseStrategy(ABC):
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None)
    
    @abstractmethod
    def select_action(self, state: Any, valid_actions: Any) -> Any
    
    def update(self, state: Any, action: Any, reward: float, 
               next_state: Any, done: bool) -> None
    def reset(self) -> None
    def get_statistics(self) -> Dict[str, Any]
    def save(self, filepath: str) -> None
    def load(self, filepath: str) -> None
```

**Parameters:**
- `name` (str): Strategy identifier
- `config` (dict, optional): Configuration parameters

**Methods:**

##### `select_action(state, valid_actions) -> Any`
Select action given current state.

**Parameters:**
- `state`: Current environment state
- `valid_actions`: Available actions

**Returns:**
- Selected action

##### `update(state, action, reward, next_state, done) -> None`
Update strategy after observing transition (for learning strategies).

**Parameters:**
- `state`: State where action was taken
- `action`: Action taken
- `reward`: Reward received
- `next_state`: Resulting state
- `done`: Whether episode ended

##### `get_statistics() -> Dict[str, Any]`
Get performance statistics.

**Returns:**
Dictionary with keys:
- `games_played`: Number of complete games
- `total_reward`: Cumulative reward
- `wins`: Number of wins
- `losses`: Number of losses
- `win_rate`: Win percentage
- `avg_reward`: Average reward per game

---

### base_estimator

#### `BaseEstimator`

Abstract base class for probability estimators.

```python
class BaseEstimator(ABC):
    def __init__(self, name: str = "BaseEstimator")
    
    @abstractmethod
    def estimate(self) -> float
    
    @abstractmethod
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]
    
    @abstractmethod
    def update(self, *args, **kwargs) -> None
    
    def variance(self) -> float
    def std(self) -> float
    def sample(self, n_samples: int = 1) -> np.ndarray
    def reset(self) -> None
    def get_metadata(self) -> Dict[str, Any]
```

#### `BetaEstimator`

Bayesian estimator using Beta distribution for Bernoulli processes.

```python
class BetaEstimator(BaseEstimator):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, name: str = "BetaEstimator")
    
    def estimate(self) -> float
    def variance(self) -> float
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]
    def update(self, successes: int = 0, failures: int = 0) -> None
    def sample(self, n_samples: int = 1) -> np.ndarray
    def reset(self) -> None
```

**Parameters:**
- `alpha` (float): Prior shape parameter for successes (default: 1.0)
- `beta` (float): Prior shape parameter for failures (default: 1.0)
- `name` (str): Estimator name

**Properties:**
- Posterior mean: `α / (α + β)`
- Posterior variance: `αβ / [(α+β)²(α+β+1)]`

**Methods:**

##### `update(successes=0, failures=0) -> None`
Update with new observations.

**Parameters:**
- `successes` (int): Number of successful outcomes
- `failures` (int): Number of failed outcomes

**Aliases:**
- `wins`: Alias for `successes`
- `losses`: Alias for `failures`

---

### config

Configuration system using type-safe dataclasses.

#### `EstimatorConfig`

```python
@dataclass
class EstimatorConfig:
    estimator_type: str = "beta"
    alpha: float = 1.0
    beta: float = 1.0
    confidence_level: float = 0.95
    name: str = "default_estimator"
```

#### `StrategyConfig`

```python
@dataclass
class StrategyConfig:
    strategy_type: str = "basic"
    risk_tolerance: float = 0.5
    kelly_fraction: float = 0.25
    min_confidence_threshold: float = 0.1
    estimator_config: EstimatorConfig = field(default_factory=EstimatorConfig)
    name: str = "default_strategy"
    params: Dict[str, Any] = field(default_factory=dict)
```

#### `SimulationConfig`

```python
@dataclass
class SimulationConfig:
    num_simulations: int = 10000
    num_episodes_per_sim: int = 1
    seed: Optional[int] = None
    parallel: bool = False
    n_jobs: int = -1
    convergence_check: bool = True
    convergence_window: int = 100
    convergence_tolerance: float = 0.001
    save_results: bool = True
    output_dir: str = "results"
    log_level: str = "INFO"
```

#### `FrameworkConfig`

```python
@dataclass
class FrameworkConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    game: GameConfig = field(default_factory=GameConfig)
    strategies: List[StrategyConfig] = field(default_factory=list)
    meta_strategy: Optional[StrategyConfig] = None
    
    def to_dict(self) -> Dict[str, Any]
    def to_json(self, filepath: str) -> None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FrameworkConfig'
    
    @classmethod
    def from_json(cls, filepath: str) -> 'FrameworkConfig'
    
    def validate(self) -> bool
```

**Functions:**

##### `create_default_config() -> FrameworkConfig`
Create a default configuration with sensible values.

##### `load_config(filepath: str) -> FrameworkConfig`
Load and validate configuration from JSON file.

---

### plugin_system

#### `PluginRegistry`

Central registry for plugins (simulators, strategies, estimators).

```python
class PluginRegistry:
    def register_simulator(self, name: str, simulator_class: Type[BaseSimulator]) -> None
    def register_strategy(self, name: str, strategy_class: Type[BaseStrategy]) -> None
    def register_estimator(self, name: str, estimator_class: Type[BaseEstimator]) -> None
    
    def get_simulator(self, name: str) -> Type[BaseSimulator]
    def get_strategy(self, name: str) -> Type[BaseStrategy]
    def get_estimator(self, name: str) -> Type[BaseEstimator]
    
    def list_simulators(self) -> List[str]
    def list_strategies(self) -> List[str]
    def list_estimators(self) -> List[str]
    
    def create_simulator(self, name: str, **kwargs) -> BaseSimulator
    def create_strategy(self, name: str, **kwargs) -> BaseStrategy
    def create_estimator(self, name: str, **kwargs) -> BaseEstimator
    
    def discover_plugins(self, plugin_dir: str) -> int
```

**Functions:**

##### `get_registry() -> PluginRegistry`
Get the global plugin registry instance.

**Decorators:**

##### `@register_simulator(name: str)`
Decorator to register a simulator class.

##### `@register_strategy(name: str)`
Decorator to register a strategy class.

##### `@register_estimator(name: str)`
Decorator to register an estimator class.

---

### parallel_engine

#### `SimulationResult`

```python
@dataclass
class SimulationResult:
    run_id: int
    seed: int
    reward: float
    steps: int
    success: bool
    metadata: Dict[str, Any]
    duration: float
```

#### `AggregatedResults`

```python
@dataclass
class AggregatedResults:
    num_runs: int
    mean_reward: float
    std_reward: float
    confidence_interval: Tuple[float, float]
    success_rate: float
    total_duration: float
    convergence_achieved: bool
    all_results: List[SimulationResult]
```

#### `ParallelSimulationEngine`

```python
class ParallelSimulationEngine:
    def __init__(self, config: SimulationConfig, n_jobs: Optional[int] = None)
    
    def run_batch(self, 
                  simulator: BaseSimulator,
                  strategy: BaseStrategy,
                  num_runs: Optional[int] = None,
                  show_progress: bool = True) -> AggregatedResults
```

**Methods:**

##### `run_batch(...) -> AggregatedResults`
Run batch of simulations in parallel.

**Parameters:**
- `simulator`: Simulator instance or factory
- `strategy`: Strategy instance or factory
- `num_runs`: Number of runs (uses config if None)
- `show_progress`: Whether to display progress

**Returns:**
- `AggregatedResults` with statistics

---

### reproducibility

#### `ExperimentLogger`

Comprehensive logger for experiment tracking.

```python
class ExperimentLogger:
    def __init__(self, output_dir: str, experiment_name: str = "experiment")
    
    def start_experiment(self, config: Dict[str, Any], seed: Optional[int] = None,
                        tags: Optional[List[str]] = None, description: str = "") -> None
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None
    def log_results(self, results: Dict[str, Any], checkpoint: bool = False) -> None
    def save_artifact(self, name: str, obj: Any, format: str = 'json') -> None
    def end_experiment(self, final_results: Optional[Dict[str, Any]] = None) -> None
    def get_experiment_dir(self) -> Path
```

**Functions:**

##### `set_global_seed(seed: int) -> None`
Set global random seed for all libraries (numpy, random, torch, tf).

---

## CLI

### Commands

#### `apf run`
Run Monte Carlo simulations.

```bash
apf run [OPTIONS]

Options:
  -c, --config PATH        Configuration file (JSON)
  -n, --num-simulations N  Number of simulation runs
  -s, --seed SEED         Random seed
  --simulator NAME        Simulator to use
  --strategy NAME         Strategy to use
  -p, --parallel          Run in parallel
  -j, --jobs N            Number of parallel jobs
  -o, --output-dir DIR    Output directory
  -q, --quiet             Suppress progress
```

#### `apf config`
Manage configurations.

```bash
apf config [OPTIONS]

Options:
  --create PATH      Create default configuration
  --validate PATH    Validate configuration
  --show PATH        Display configuration
```

#### `apf plugins`
Manage plugins.

```bash
apf plugins [OPTIONS]

Options:
  --list              List registered plugins
  --discover PATH     Discover plugins in directory
```

#### `apf analyze`
Analyze results.

```bash
apf analyze RESULTS_FILE [OPTIONS]

Options:
  -d, --detailed     Show detailed information
```

---

## RL / Bayesian / Multi-Agent Modules (scaffold)

### Deep Reinforcement Learning (DRL)

#### `MinesEnv`

RL environment wrapper for mines game simulation.

```python
class MinesEnv:
    def __init__(self, n_cells: int = 25)
    def reset(self) -> np.ndarray
    def step(self, action: int) -> StepResult
    def valid_actions(self) -> List[int]
    def encode_state(self) -> np.ndarray
```

**Parameters:**
- `n_cells` (int): Number of cells in the mines grid (default: 25 for 5x5)

**Methods:**

##### `reset() -> np.ndarray`
Reset environment to initial state.

**Returns:**
- Initial state vector (n_cells + 4 meta features)

##### `step(action) -> StepResult`
Execute action and return result.

**Parameters:**
- `action` (int): Action to take (0 to n_cells-1 for clicks, n_cells for cashout)

**Returns:**
- `StepResult` with next_state, reward, done, and info

#### `DQNAgent`

Deep Q-Network agent for learning optimal policies.

```python
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, gamma=0.99, lr=1e-3, 
                 eps_start=1.0, eps_end=0.05, eps_decay=20000)
    def act(self, state, valid_mask=None)
    def learn(self, batch_size=256, target_update=500)
    def epsilon(self)
```

**Parameters:**
- `state_dim` (int): Dimension of state space
- `action_dim` (int): Number of possible actions
- `gamma` (float): Discount factor (default: 0.99)
- `lr` (float): Learning rate (default: 1e-3)
- `eps_start` (float): Initial exploration rate (default: 1.0)
- `eps_end` (float): Final exploration rate (default: 0.05)
- `eps_decay` (int): Exploration decay steps (default: 20000)

**Methods:**

##### `act(state, valid_mask=None) -> int`
Select action using epsilon-greedy policy.

**Parameters:**
- `state`: Current state vector
- `valid_mask`: Optional mask for valid actions

**Returns:**
- Selected action index

##### `learn(batch_size=256, target_update=500) -> None`
Update Q-network using experience replay.

**Parameters:**
- `batch_size` (int): Size of training batch
- `target_update` (int): Steps between target network updates

#### `train_dqn`

Training function for DQN agent.

```python
def train_dqn(episodes=2000, n_cells=25, log_every=100) -> DQNAgent
```

**Parameters:**
- `episodes` (int): Number of training episodes
- `n_cells` (int): Number of cells in environment
- `log_every` (int): Logging frequency

**Returns:**
- Trained DQNAgent instance

---

### Bayesian Inference

#### `BayesianMineModel`

Bayesian model for mine probability inference.

```python
class BayesianMineModel:
    def __init__(self, n_cells=25)
    def infer_safety(self, visible: np.ndarray) -> List[Tuple[float, float]]
```

**Parameters:**
- `n_cells` (int): Number of cells in the grid

**Methods:**

##### `infer_safety(visible) -> List[Tuple[float, float]]`
Infer safety probabilities for each cell.

**Parameters:**
- `visible` (np.ndarray): Array indicating revealed cells

**Returns:**
- List of (mean, variance) tuples for each cell's safety probability

---

### Multi-Agent Coordination

#### `BaseAgent`

Abstract base class for multi-agent system participants.

```python
class BaseAgent(ABC):
    @abstractmethod
    def propose(self, state) -> Dict[str, Any]
```

**Methods:**

##### `propose(state) -> Dict[str, Any]`
Propose action with confidence and reasoning.

**Parameters:**
- `state`: Current environment state

**Returns:**
- Dictionary with keys: 'action', 'confidence', 'reason'

#### `Blackboard`

Shared information board for multi-agent communication.

```python
class Blackboard:
    def __init__(self)
    def post_obs(self, obs: Dict[str, Any])
    def post_proposal(self, prop: Dict[str, Any])
    def clear(self)
```

**Methods:**

##### `post_obs(obs) -> None`
Post observation to blackboard.

##### `post_proposal(prop) -> None`
Post action proposal to blackboard.

##### `clear() -> None`
Clear all observations and proposals.

#### `majority_vote`

Consensus mechanism for multi-agent decisions.

```python
def majority_vote(proposals: List[Dict[str, Any]]) -> Dict[str, Any]
```

**Parameters:**
- `proposals`: List of agent proposals

**Returns:**
- Consensus decision with action, confidence, and reason

#### `TeamSimulator`

Simulator for multi-agent team coordination.

```python
class TeamSimulator:
    def __init__(self, env, agents: List)
    def play_round(self)
```

**Parameters:**
- `env`: Environment instance
- `agents`: List of agent instances

**Methods:**

##### `play_round() -> bool`
Run one round of multi-agent coordination.

**Returns:**
- True if round completed successfully

---

### CLI Integration

#### `simulate` Command

Command-line interface for running RL simulations.

```bash
python -m src.python.cli.simulate [OPTIONS]

Options:
  --episodes N    Number of training episodes (default: 2000)
  --cells N       Number of cells in grid (default: 25)
```

**Example:**
```bash
python -m src.python.cli.simulate --episodes 1000 --cells 25
```

---

## Type Annotations

All public APIs include complete type hints following PEP 484:

```python
from typing import Any, Dict, Optional, List, Tuple, Type
import numpy as np

def run_simulation(
    simulator: BaseSimulator,
    strategy: BaseStrategy,
    num_episodes: int = 100,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    ...
```

---

## Error Handling

The framework uses standard Python exceptions:

- `ValueError`: Invalid parameters or configuration
- `TypeError`: Type mismatches
- `KeyError`: Missing registry entries or config keys
- `FileNotFoundError`: Missing configuration or result files

Example:

```python
try:
    config = FrameworkConfig.from_json("config.json")
    config.validate()
except FileNotFoundError:
    print("Configuration file not found")
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

---

## Best Practices

1. **Always set seeds** for reproducible experiments
2. **Use configuration files** instead of hardcoding parameters
3. **Enable logging** for experiment tracking
4. **Validate configurations** before running simulations
5. **Use parallel execution** for large-scale experiments
6. **Check convergence** to avoid unnecessary computation

---

For more examples, see the [main README](../README.md) and [tutorials](TUTORIALS.md).

