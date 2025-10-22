# Tutorials

Step-by-step tutorials for using the Applied Probability Framework.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Simulation](#basic-simulation)
3. [Custom Strategies](#custom-strategies)
4. [Custom Simulators](#custom-simulators)
5. [Bayesian Estimation](#bayesian-estimation)
6. [Parallel Execution](#parallel-execution)
7. [Experiment Tracking](#experiment-tracking)
8. [Plugin Development](#plugin-development)

---

## Getting Started

### Installation

```bash
git clone https://github.com/yourusername/applied-probability-framework.git
cd applied-probability-framework/src/python
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

```bash
# Check CLI is available
apf --version

# List available plugins
apf plugins --list
```

---

## Basic Simulation

### Using the CLI

**Step 1: Create a configuration**

```bash
apf config --create my_first_config.json
```

**Step 2: Edit the configuration** (optional)

Open `my_first_config.json` and modify parameters:

```json
{
  "simulation": {
    "num_simulations": 1000,
    "seed": 42,
    "parallel": false
  }
}
```

**Step 3: Run the simulation**

```bash
apf run --config my_first_config.json
```

**Step 4: Analyze results**

```bash
apf analyze results/results_mines_basic.json
```

### Programmatic Usage

```python
from core.config import create_default_config
from core.parallel_engine import ParallelSimulationEngine
from core.plugin_system import get_registry

# Create configuration
config = create_default_config()
config.simulation.num_simulations = 1000
config.simulation.seed = 42

# Get plugins
registry = get_registry()
simulator = registry.create_simulator("mines")
strategy = registry.create_strategy("conservative")

# Run simulation
engine = ParallelSimulationEngine(config.simulation)
results = engine.run_batch(simulator, strategy, show_progress=True)

# Print results
print(f"Mean reward: {results.mean_reward:.4f}")
print(f"Std dev: {results.std_reward:.4f}")
print(f"Success rate: {results.success_rate:.2%}")
print(f"95% CI: [{results.confidence_interval[0]:.4f}, {results.confidence_interval[1]:.4f}]")
```

---

## Custom Strategies

### Example 1: Simple Threshold Strategy

Create a strategy that only acts when confidence is above a threshold:

```python
from core.base_strategy import BaseStrategy
from core.base_estimator import BetaEstimator
from typing import Any, List

class ThresholdStrategy(BaseStrategy):
    """Strategy that requires minimum confidence before acting."""
    
    def __init__(self, name="threshold", config=None):
        super().__init__(name, config)
        self.threshold = config.get('threshold', 0.6) if config else 0.6
        self.estimator = BetaEstimator(alpha=1.0, beta=1.0)
        
    def select_action(self, state, valid_actions: List[Any]) -> Any:
        """Select action only if confidence is high enough."""
        if not valid_actions:
            raise ValueError("No valid actions")
        
        # Check confidence
        estimate = self.estimator.estimate()
        ci_lower, ci_upper = self.estimator.confidence_interval()
        ci_width = ci_upper - ci_lower
        
        # If uncertain, choose conservatively
        if ci_width > 0.3 or estimate < self.threshold:
            # Choose safest action (implementation depends on domain)
            return valid_actions[0]
        
        # If confident, be aggressive
        return valid_actions[-1]
    
    def update(self, state, action, reward, next_state, done):
        """Update estimator with outcome."""
        super().update(state, action, reward, next_state, done)
        
        if done:
            success = 1 if reward > 0 else 0
            self.estimator.update(successes=success, failures=1-success)
```

**Using the strategy:**

```python
from core.plugin_system import get_registry

# Register strategy
registry = get_registry()
registry.register_strategy("threshold", ThresholdStrategy)

# Create instance
strategy = registry.create_strategy("threshold")

# Or directly
strategy = ThresholdStrategy(config={'threshold': 0.7})
```

### Example 2: Adaptive Kelly Strategy

Implement Kelly Criterion with adaptive bet sizing:

```python
from core.base_strategy import BaseStrategy
from core.base_estimator import BetaEstimator
import numpy as np

class AdaptiveKellyStrategy(BaseStrategy):
    """Kelly Criterion strategy with adaptive fractional betting."""
    
    def __init__(self, name="adaptive_kelly", config=None):
        super().__init__(name, config)
        self.base_fraction = config.get('base_fraction', 0.25) if config else 0.25
        self.estimator = BetaEstimator(alpha=2.0, beta=2.0)
        
    def select_action(self, state, valid_actions):
        """Select action using Kelly Criterion."""
        if not valid_actions:
            raise ValueError("No valid actions")
        
        # Get probability estimate
        p = self.estimator.estimate()
        ci_lower, ci_upper = self.estimator.confidence_interval()
        ci_width = ci_upper - ci_lower
        
        # Adjust Kelly fraction based on confidence
        # More uncertain → more conservative
        if ci_width > 0.4:
            kelly_fraction = self.base_fraction * 0.5  # Very conservative
        elif ci_width > 0.2:
            kelly_fraction = self.base_fraction * 0.75
        else:
            kelly_fraction = self.base_fraction
        
        # Calculate Kelly bet
        # Assuming even money (b=1), Kelly = p - (1-p) = 2p - 1
        kelly_bet = max(0, 2*p - 1) * kelly_fraction
        
        # Choose action based on bet size
        # (domain-specific mapping)
        if kelly_bet > 0.3:
            return 'aggressive_action' if 'aggressive_action' in valid_actions else valid_actions[-1]
        elif kelly_bet > 0.1:
            return 'moderate_action' if 'moderate_action' in valid_actions else valid_actions[len(valid_actions)//2]
        else:
            return 'conservative_action' if 'conservative_action' in valid_actions else valid_actions[0]
    
    def update(self, state, action, reward, next_state, done):
        """Update estimator."""
        super().update(state, action, reward, next_state, done)
        
        if done:
            win = 1 if reward > 0 else 0
            self.estimator.update(successes=win, failures=1-win)
```

---

## Custom Simulators

### Example: Coin Flip Game

```python
from core.base_simulator import BaseSimulator
from typing import Any, List, Tuple, Dict
import numpy as np

class CoinFlipSimulator(BaseSimulator):
    """Simple coin flip game simulator."""
    
    def __init__(self, seed=None, config=None):
        super().__init__(seed, config)
        self.bias = config.get('bias', 0.5) if config else 0.5  # Coin bias
        self.max_flips = config.get('max_flips', 10) if config else 10
        
    def reset(self) -> Dict[str, Any]:
        """Start new game."""
        self.state = {
            'flips': 0,
            'heads': 0,
            'tails': 0,
            'balance': 100.0,
            'done': False
        }
        return self.state
    
    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        """Execute one flip."""
        if self.state['done']:
            return self.state, 0.0, True, {}
        
        # Flip coin
        is_heads = self.rng.random() < self.bias
        
        # Process bet
        bet_amount = 10.0  # Fixed bet
        if action == 'bet_heads':
            if is_heads:
                reward = bet_amount
                self.state['balance'] += bet_amount
            else:
                reward = -bet_amount
                self.state['balance'] -= bet_amount
        elif action == 'bet_tails':
            if not is_heads:
                reward = bet_amount
                self.state['balance'] += bet_amount
            else:
                reward = -bet_amount
                self.state['balance'] -= bet_amount
        else:
            reward = 0.0  # Pass
        
        # Update state
        self.state['flips'] += 1
        if is_heads:
            self.state['heads'] += 1
        else:
            self.state['tails'] += 1
        
        # Check if done
        done = (self.state['flips'] >= self.max_flips or 
                self.state['balance'] <= 0)
        self.state['done'] = done
        
        info = {
            'is_heads': is_heads,
            'balance': self.state['balance']
        }
        
        return self.state, reward, done, info
    
    def get_valid_actions(self) -> List[str]:
        """Available actions."""
        if self.state and not self.state['done'] and self.state['balance'] > 0:
            return ['bet_heads', 'bet_tails', 'pass']
        return []
    
    def get_state_representation(self) -> np.ndarray:
        """Numerical state for ML."""
        if self.state:
            return np.array([
                self.state['flips'],
                self.state['heads'],
                self.state['tails'],
                self.state['balance']
            ])
        return np.zeros(4)
```

**Using the custom simulator:**

```python
from core.plugin_system import get_registry, register_simulator

# Register
@register_simulator("coinflip")
class CoinFlipSimulator(BaseSimulator):
    # ... (as above)
    pass

# Use it
registry = get_registry()
simulator = registry.create_simulator("coinflip", config={'bias': 0.55, 'max_flips': 20})

# Run simulation
state = simulator.reset()
done = False
total_reward = 0

while not done:
    actions = simulator.get_valid_actions()
    action = actions[0]  # Simple policy
    state, reward, done, info = simulator.step(action)
    total_reward += reward

print(f"Final balance: ${state['balance']:.2f}")
```

---

## Bayesian Estimation

### Understanding the Beta Distribution

The Beta distribution is ideal for modeling probabilities:

```python
from core.base_estimator import BetaEstimator
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Start with uniform prior (no knowledge)
estimator = BetaEstimator(alpha=1.0, beta=1.0)

print(f"Prior estimate: {estimator.estimate():.4f}")  # 0.5

# After 7 wins, 3 losses
estimator.update(successes=7, failures=3)
print(f"After 10 trials: {estimator.estimate():.4f}")  # ~0.7273

# Get confidence interval
ci = estimator.confidence_interval(confidence=0.95)
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

# Visualize the posterior
x = np.linspace(0, 1, 100)
y = stats.beta.pdf(x, estimator.alpha, estimator.beta)
plt.plot(x, y)
plt.title("Posterior Distribution")
plt.xlabel("Probability")
plt.ylabel("Density")
plt.show()
```

### Thompson Sampling

Use Beta estimator for multi-armed bandits:

```python
from core.base_estimator import BetaEstimator
import numpy as np

class ThompsonSamplingStrategy(BaseStrategy):
    """Thompson Sampling for multi-armed bandits."""
    
    def __init__(self, name="thompson", config=None):
        super().__init__(name, config)
        self.estimators = {}  # One per action
        
    def select_action(self, state, valid_actions):
        """Sample from each arm's posterior and choose best."""
        if not valid_actions:
            raise ValueError("No valid actions")
        
        # Initialize estimators for new actions
        for action in valid_actions:
            if action not in self.estimators:
                self.estimators[action] = BetaEstimator(alpha=1.0, beta=1.0)
        
        # Sample from each posterior
        samples = {
            action: self.estimators[action].sample(1)[0]
            for action in valid_actions
        }
        
        # Choose action with highest sample
        return max(samples, key=samples.get)
    
    def update(self, state, action, reward, next_state, done):
        """Update the selected arm's posterior."""
        super().update(state, action, reward, next_state, done)
        
        if done and action in self.estimators:
            win = 1 if reward > 0 else 0
            self.estimators[action].update(successes=win, failures=1-win)
```

---

## Parallel Execution

### Basic Parallel Simulation

```python
from core.config import SimulationConfig
from core.parallel_engine import ParallelSimulationEngine

# Configure for parallel execution
config = SimulationConfig(
    num_simulations=10000,
    seed=42,
    parallel=True,
    n_jobs=-1,  # Use all CPU cores
    save_results=True,
    output_dir="parallel_results"
)

# Run
engine = ParallelSimulationEngine(config)
results = engine.run_batch(simulator, strategy, show_progress=True)

print(f"Completed {results.num_runs} runs in {results.total_duration:.2f}s")
print(f"Throughput: {results.num_runs / results.total_duration:.1f} runs/second")
```

### Convergence-Based Termination

```python
# Enable convergence detection
config = SimulationConfig(
    num_simulations=100000,  # Maximum runs
    convergence_check=True,
    convergence_window=1000,  # Compare windows of 1000 runs
    convergence_tolerance=0.001,  # 0.1% change threshold
    parallel=True
)

engine = ParallelSimulationEngine(config)
results = engine.run_batch(simulator, strategy)

if results.convergence_achieved:
    print(f"Converged after {results.num_runs} runs (saved time!)")
else:
    print(f"Did not converge within {config.num_simulations} runs")
```

---

## Experiment Tracking

### Full Experiment Pipeline

```python
from core.reproducibility import ExperimentLogger, set_global_seed

# Create logger
logger = ExperimentLogger(
    output_dir="experiments",
    experiment_name="kelly_comparison"
)

# Configure experiment
config = {
    'simulation': {'num_runs': 5000, 'seed': 42},
    'strategies': ['conservative', 'balanced', 'aggressive'],
    'hypothesis': 'Aggressive Kelly outperforms conservative in high-variance environments'
}

# Start experiment
logger.start_experiment(
    config=config,
    seed=42,
    tags=['kelly', 'comparison', 'variance-analysis'],
    description="Comparing Kelly fractions across risk levels"
)

# Set reproducible seed
set_global_seed(42)

# Run simulations
for strategy_name in config['strategies']:
    logger.log_event("strategy_start", {'name': strategy_name})
    
    # Run simulation (pseudocode)
    # results = run_simulation(strategy_name)
    
    logger.log_event("strategy_complete", {
        'name': strategy_name,
        'mean_reward': 0.0,  # Replace with actual
        'success_rate': 0.0
    })

# Save final results
logger.end_experiment(final_results={'status': 'complete'})

# Experiment directory contains:
# - metadata.json
# - config.json
# - events.jsonl
# - results.json
# - summary.json
# - random_states/*.pkl

print(f"Experiment saved to: {logger.get_experiment_dir()}")
```

---

## Plugin Development

### Creating a Plugin Package

**File structure:**
```
my_plugins/
├── __init__.py
├── my_simulator.py
└── my_strategy.py
```

**my_simulator.py:**
```python
from core.base_simulator import BaseSimulator

class MySimulator(BaseSimulator):
    register = True  # Enable auto-discovery
    plugin_name = "my_custom_sim"
    
    def reset(self):
        # Implementation
        pass
    
    def step(self, action):
        # Implementation
        pass
    
    def get_valid_actions(self):
        # Implementation
        pass
    
    def get_state_representation(self):
        # Implementation
        pass
```

**Discovering plugins:**

```python
from core.plugin_system import get_registry

registry = get_registry()
count = registry.discover_plugins("my_plugins/")
print(f"Discovered {count} plugins")

# Use discovered plugins
simulator = registry.create_simulator("my_custom_sim")
```

**Or use CLI:**

```bash
apf plugins --discover my_plugins/
apf plugins --list
```

---

## Next Steps

- Read the [API Reference](API_REFERENCE.md) for detailed documentation
- Check out [Theoretical Background](THEORETICAL_BACKGROUND.md) for mathematical foundations
- Explore example implementations in `examples/` directory
- Join our community and contribute!


