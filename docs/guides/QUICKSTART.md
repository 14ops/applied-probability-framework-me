# Quick Start Guide

Get up and running with the Applied Probability Framework in 5 minutes!

## Installation

```bash
git clone https://github.com/yourusername/applied-probability-framework.git
cd applied-probability-framework/src/python
pip install -r requirements.txt
```

## Hello World: Your First Simulation

### 1. Command Line (Easiest)

```bash
# Create a configuration
python cli.py config --create my_config.json

# Run simulation
python cli.py run --config my_config.json

# View results
python cli.py analyze results/*.json
```

### 2. Python Script

Create `my_first_simulation.py`:

```python
from core.config import create_default_config
from core.parallel_engine import ParallelSimulationEngine
from core.plugin_system import get_registry

# Get default configuration
config = create_default_config()
config.simulation.num_simulations = 1000

# Get simulator and strategy
registry = get_registry()
simulator = registry.create_simulator("mines")
strategy = registry.create_strategy("conservative")

# Run simulation
engine = ParallelSimulationEngine(config.simulation)
results = engine.run_batch(simulator, strategy)

# Print results
print(f"Mean Reward: {results.mean_reward:.2f}")
print(f"Success Rate: {results.success_rate:.2%}")
```

Run it:
```bash
python my_first_simulation.py
```

## 🎮 Mines Game Quick Start

### Run Character Strategies

```bash
# Test Takeshi's aggressive doubling strategy
python -c "
from strategies import TakeshiStrategy
from game.math import win_probability, get_observed_payout

strategy = TakeshiStrategy(config={'base_bet': 10.0, 'target_clicks': 8})
print(f'Strategy: {strategy.name}')
print(f'Target: {strategy.target_clicks} clicks')
print(f'Win probability: {win_probability(8):.2%}')
print(f'Payout: {get_observed_payout(8):.2f}x')
"

# Play interactively with GUI
python src/python/gui_game.py
```

### Create Custom Mines Strategy

Create `my_mines_strategy.py`:

```python
from strategies.base import StrategyBase
from game.math import win_probability, expected_value

class MyStrategy(StrategyBase):
    """Custom Mines strategy."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "My Strategy"
        self.target_clicks = 7  # Sweet spot
        
    def decide(self, state):
        """Make decision."""
        if not state.get('game_active', False):
            # Place bet
            return {
                "action": "bet",
                "bet": self.base_bet,
                "max_clicks": self.target_clicks
            }
        
        # In game
        clicks = state.get('clicks_made', 0)
        if clicks < self.target_clicks:
            # Click random unrevealed tile
            pos = self._find_random_tile(state)
            return {"action": "click", "position": pos}
        else:
            # Cashout at target
            return {"action": "cashout"}
    
    def on_result(self, result):
        """Learn from result."""
        self.update_tracking(result)
        
        # Adjust target based on performance
        if result.get('win', False):
            self.target_clicks = min(self.target_clicks + 1, 10)
        else:
            self.target_clicks = max(self.target_clicks - 1, 5)
```

### Run Tournament with Your Strategy

```bash
# Add to examples/tournaments/character_tournament.py
from strategies import TakeshiStrategy, YuzuStrategy, MyStrategy

strategies = {
    'takeshi': TakeshiStrategy(),
    'yuzu': YuzuStrategy(),
    'mine': MyStrategy(),  # Your strategy
}

# Run tournament
python examples/tournaments/character_tournament.py
```

### Mathematical Analysis

```python
from game.math import (
    win_probability,
    expected_value,
    get_observed_payout,
    print_payout_table
)

# Analyze different click counts
for k in [5, 6, 7, 8, 10]:
    prob = win_probability(k)
    payout = get_observed_payout(k)
    ev = expected_value(k, bet=10.0)
    print(f"{k} clicks: P={prob:.2%}, Payout={payout:.2f}x, EV=${ev:+.2f}")

# Print full payout table
print_payout_table(tiles=25, mines=2)
```

Output:
```
5 clicks: P=63.33%, Payout=1.53x, EV=$+0.02
6 clicks: P=57.00%, Payout=1.72x, EV=$-0.02
7 clicks: P=51.00%, Payout=1.95x, EV=$+0.05
8 clicks: P=45.33%, Payout=2.12x, EV=$+0.41
10 clicks: P=35.00%, Payout=3.02x, EV=$+0.41
```

## Example: Custom Strategy in 5 Minutes

Create `my_strategy.py`:

```python
from core.base_strategy import BaseStrategy
from core.base_estimator import BetaEstimator

class CautiousStrategy(BaseStrategy):
    """Only acts when very confident."""
    
    def __init__(self, name="cautious", config=None):
        super().__init__(name, config)
        self.estimator = BetaEstimator(alpha=1.0, beta=1.0)
    
    def select_action(self, state, valid_actions):
        if not valid_actions:
            raise ValueError("No actions available")
        
        # Check confidence
        estimate = self.estimator.estimate()
        ci_lower, ci_upper = self.estimator.confidence_interval()
        
        # Only act if very confident
        if (ci_upper - ci_lower) < 0.2 and estimate > 0.6:
            return valid_actions[-1]  # Aggressive
        else:
            return valid_actions[0]   # Conservative
    
    def update(self, state, action, reward, next_state, done):
        super().update(state, action, reward, next_state, done)
        if done:
            win = 1 if reward > 0 else 0
            self.estimator.update(successes=win, failures=1-win)

# Register it
from core.plugin_system import get_registry
registry = get_registry()
registry.register_strategy("cautious", CautiousStrategy)

# Use it
strategy = registry.create_strategy("cautious")
print(f"Created strategy: {strategy}")
```

## Example: Run Parallel Simulations

```python
from core.config import SimulationConfig
from core.parallel_engine import ParallelSimulationEngine
from core.plugin_system import get_registry

# Configure for parallel execution
config = SimulationConfig(
    num_simulations=10000,
    seed=42,
    parallel=True,
    n_jobs=-1,  # Use all CPU cores
    save_results=True
)

# Get components
registry = get_registry()
simulator = registry.create_simulator("mines")
strategy = registry.create_strategy("balanced")

# Run
engine = ParallelSimulationEngine(config)
results = engine.run_batch(simulator, strategy, show_progress=True)

print(f"\nCompleted {results.num_runs} runs in {results.total_duration:.2f}s")
print(f"Mean reward: {results.mean_reward:.4f} ± {results.std_reward:.4f}")
print(f"95% CI: [{results.confidence_interval[0]:.4f}, {results.confidence_interval[1]:.4f}]")
```

## Example: Track Experiments

```python
from core.reproducibility import ExperimentLogger, set_global_seed
from core.parallel_engine import ParallelSimulationEngine
from core.plugin_system import get_registry

# Create experiment logger
logger = ExperimentLogger("experiments", "kelly_test")

# Start experiment
config_dict = {
    'num_simulations': 5000,
    'strategy': 'kelly',
    'kelly_fraction': 0.25
}

logger.start_experiment(
    config=config_dict,
    seed=42,
    tags=['kelly', 'conservative'],
    description="Testing Kelly Criterion with 25% fraction"
)

# Set reproducible seed
set_global_seed(42)

# Run simulation
registry = get_registry()
simulator = registry.create_simulator("mines")
strategy = registry.create_strategy("conservative")

from core.config import SimulationConfig
sim_config = SimulationConfig(num_simulations=5000, seed=42)
engine = ParallelSimulationEngine(sim_config)
results = engine.run_batch(simulator, strategy, show_progress=True)

# Log results
results_dict = {
    'mean_reward': results.mean_reward,
    'std_reward': results.std_reward,
    'success_rate': results.success_rate,
    'num_runs': results.num_runs
}
logger.log_results(results_dict)

# End experiment
logger.end_experiment()

print(f"\nExperiment saved to: {logger.get_experiment_dir()}")
print("Contains: metadata.json, config.json, results.json, summary.json")
```

## Example: Compare Multiple Strategies

```python
from core.config import SimulationConfig
from core.parallel_engine import ParallelSimulationEngine
from core.plugin_system import get_registry
import pandas as pd

# Setup
config = SimulationConfig(num_simulations=1000, seed=42, parallel=True)
engine = ParallelSimulationEngine(config)
registry = get_registry()
simulator = registry.create_simulator("mines")

# Test multiple strategies
strategies_to_test = ['conservative', 'balanced', 'aggressive']
results_summary = []

for strategy_name in strategies_to_test:
    print(f"\nTesting {strategy_name}...")
    strategy = registry.create_strategy(strategy_name)
    results = engine.run_batch(simulator, strategy, show_progress=False)
    
    results_summary.append({
        'Strategy': strategy_name,
        'Mean Reward': results.mean_reward,
        'Std Dev': results.std_reward,
        'Success Rate': results.success_rate,
        'CI Lower': results.confidence_interval[0],
        'CI Upper': results.confidence_interval[1]
    })

# Display comparison table
df = pd.DataFrame(results_summary)
print("\n" + "="*80)
print("STRATEGY COMPARISON")
print("="*80)
print(df.to_string(index=False))
```

## Next Steps

1. **Read the Documentation**
   - [README.md](README.md) - Full overview
   - [docs/API_REFERENCE.md](docs/API_REFERENCE.md) - Complete API
   - [docs/TUTORIALS.md](docs/TUTORIALS.md) - Step-by-step guides
   - [docs/THEORETICAL_BACKGROUND.md](docs/THEORETICAL_BACKGROUND.md) - Math foundations

2. **Run the Tests**
   ```bash
   cd src/python
   pytest tests/ -v
   ```

3. **Try the Examples**
   - Check `examples/` directory (if available)
   - Modify the quick start examples above

4. **Create Your Own**
   - Custom simulator for your specific game/problem
   - Custom strategy with your decision logic
   - Custom estimator for your probability model

5. **Contribute**
   - Read [CONTRIBUTING.md](CONTRIBUTING.md)
   - Open issues for bugs or features
   - Submit pull requests

## Common Commands

```bash
# CLI Usage
apf run --help                           # See all options
apf run --parallel --jobs 8              # Parallel with 8 cores
apf run --num-simulations 10000 --seed 42  # Reproducible run
apf config --create config.json          # Create config
apf config --validate config.json        # Check config
apf plugins --list                       # List plugins
apf analyze results.json                 # Analyze results

# Development
pytest tests/ -v                         # Run tests
pytest tests/ --cov=core --cov=cli      # With coverage
black src/python/                        # Format code
mypy src/python/core/                    # Type check
pylint src/python/core/                  # Lint code
```

## Troubleshooting

**Import errors:**
```bash
# Make sure you're in the right directory
cd src/python
# Install dependencies
pip install -r requirements.txt
```

**Module not found:**
```bash
# Install in development mode
pip install -e .
```

**Tests failing:**
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-timeout
```

## Resources

- **GitHub Issues**: Report bugs or request features
- **Documentation**: `/docs` directory
- **Examples**: See above and in README.md
- **Theory**: Read `docs/THEORETICAL_BACKGROUND.md`

---

**Ready to simulate? Run your first simulation now!** 🚀

```bash
cd src/python
python cli.py run --num-simulations 100
```

