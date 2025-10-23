# Applied Probability Framework

[![CI Status](https://github.com/14ops/applied-probability-framework-me/workflows/Continuous%20Integration/badge.svg)](https://github.com/14ops/applied-probability-framework-me/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **professional-grade Monte Carlo simulation framework** for probability analysis, decision-making under uncertainty, and stochastic optimization. Designed following industry best practices with comprehensive testing, documentation, and extensibility.

## 🌟 Key Features

### Professional Infrastructure
- ✅ **Type-Safe**: Comprehensive type hints (PEP 484) throughout codebase
- ✅ **Well-Documented**: PEP 257 compliant docstrings on all modules, classes, and functions
- ✅ **Thoroughly Tested**: >80% test coverage with pytest, including unit and integration tests
- ✅ **CI/CD Pipeline**: Automated testing, linting, and security scanning via GitHub Actions
- ✅ **Reproducible**: Seed management, state serialization, and experiment tracking

### Extensible Architecture
- 🔌 **Plugin System**: Register custom simulators, strategies, and estimators without modifying core code
- 🎯 **Abstract Base Classes**: Clean interfaces following SOLID principles
- ⚙️ **Flexible Configuration**: Type-safe, validated configuration system using dataclasses
- 🔀 **Parallel Execution**: Efficient multiprocessing support for Monte Carlo simulations

### Advanced Capabilities
- 📊 **Bayesian Estimation**: Beta distributions for win rate estimation with confidence intervals
- 🎲 **Multiple Strategies**: Kelly Criterion, Conservative, Aggressive, Meta-strategies
- 📈 **Convergence Detection**: Adaptive termination based on statistical convergence
- 💾 **Experiment Logging**: Comprehensive tracking with JSON serialization and checkpointing

### Command-Line Interface
- 🖥️ **Professional CLI**: Full-featured command-line interface with argparse
- 📝 **Config Management**: Create, validate, and manage configurations
- 📊 **Result Analysis**: Built-in analysis and visualization tools

## 📦 Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/14ops/applied-probability-framework-me.git
cd applied-probability-framework-me

# Install dependencies
cd src/python
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### With pip (coming soon)

```bash
pip install applied-probability-framework
```

## 🚀 Quick Start

### 1. Create a Configuration

```bash
apf config --create my_config.json
```

This creates a default configuration file that you can customize:

```json
{
  "simulation": {
    "num_simulations": 10000,
    "seed": 42,
    "parallel": true,
    "n_jobs": -1
  },
  "game": {
    "game_type": "mines",
    "board_size": 5,
    "mine_count": 3
  },
  "strategies": [
    {
      "name": "conservative",
      "strategy_type": "kelly",
      "kelly_fraction": 0.1,
      "risk_tolerance": 0.3
    }
  ]
}
```

### 2. Run Simulations

```bash
# Run with default settings
apf run

# Run with custom config
apf run --config my_config.json

# Run 10000 simulations in parallel
apf run --num-simulations 10000 --parallel --jobs 8

# With specific seed for reproducibility
apf run --seed 42
```

### 3. Analyze Results

```bash
apf analyze results/results_mines_conservative.json --detailed
```

## 📚 Usage Examples

### Programmatic Usage

```python
from core.config import create_default_config
from core.parallel_engine import ParallelSimulationEngine
from core.plugin_system import get_registry

# Load configuration
config = create_default_config()

# Get registered plugins
registry = get_registry()
simulator = registry.create_simulator("mines", config=config.game.params)
strategy = registry.create_strategy("conservative")

# Run simulation
engine = ParallelSimulationEngine(config.simulation)
results = engine.run_batch(simulator, strategy)

# View results
print(f"Mean Reward: {results.mean_reward:.2f}")
print(f"Success Rate: {results.success_rate:.2%}")
print(f"95% CI: [{results.confidence_interval[0]:.2f}, {results.confidence_interval[1]:.2f}]")
```

### Creating Custom Strategies

```python
from core.base_strategy import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    """My custom decision-making strategy."""
    
    def __init__(self, name="custom", config=None):
        super().__init__(name, config)
        self.risk_factor = config.get('risk_factor', 0.5) if config else 0.5
    
    def select_action(self, state, valid_actions):
        """Select action based on custom logic."""
        if not valid_actions:
            raise ValueError("No valid actions")
        
        # Your decision logic here
        return valid_actions[0]

# Register the strategy
from core.plugin_system import register_strategy

registry = get_registry()
registry.register_strategy("my_custom", MyCustomStrategy)
```

### Creating Custom Simulators

```python
from core.base_simulator import BaseSimulator
import numpy as np

class MyGameSimulator(BaseSimulator):
    """Custom game simulator."""
    
    def reset(self):
        """Initialize game state."""
        self.state = {'round': 0, 'score': 0}
        return self.state
    
    def step(self, action):
        """Execute one game step."""
        reward = np.random.randn()  # Example reward
        self.state['round'] += 1
        self.state['score'] += reward
        
        done = self.state['round'] >= 10
        info = {'round': self.state['round']}
        
        return self.state, reward, done, info
    
    def get_valid_actions(self):
        """Return available actions."""
        return ['action_a', 'action_b', 'action_c']
    
    def get_state_representation(self):
        """Return numerical state representation."""
        return np.array([self.state['round'], self.state['score']])
```

## 🏗️ Architecture

The framework follows a modular, extensible architecture:

```
src/python/
├── core/                          # Core framework modules
│   ├── base_simulator.py          # Abstract simulator interface
│   ├── base_strategy.py           # Abstract strategy interface
│   ├── base_estimator.py          # Probability estimators
│   ├── config.py                  # Configuration system
│   ├── plugin_system.py           # Plugin discovery and registration
│   ├── parallel_engine.py         # Parallel simulation engine
│   └── reproducibility.py         # Experiment tracking
├── cli.py                         # Command-line interface
└── tests/                         # Comprehensive test suite
    ├── conftest.py                # Pytest fixtures
    ├── test_base_estimator.py     # Estimator tests
    ├── test_base_strategy.py      # Strategy tests
    ├── test_config.py             # Configuration tests
    ├── test_plugin_system.py      # Plugin tests
    └── test_integration.py        # Integration tests
```

## 🧪 Testing

The framework includes comprehensive test coverage:

```bash
# Run all tests
cd src/python
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=core --cov=cli --cov-report=html

# Run specific test file
pytest tests/test_base_estimator.py -v

# Run integration tests
pytest tests/test_integration.py -v
```

## 📊 Theoretical Background

### Bayesian Probability Estimation

The framework uses **Beta distributions** as conjugate priors for Bernoulli processes (win/loss estimation):

```
Prior: Beta(α, β)
Posterior after k successes, n-k failures: Beta(α+k, β+n-k)
Point Estimate: E[p] = α/(α+β)
Variance: Var[p] = αβ/[(α+β)²(α+β+1)]
```

This provides:
- **Principled uncertainty quantification** through credible intervals
- **Natural updating** as new evidence arrives
- **Thompson Sampling** for multi-armed bandit problems

### Kelly Criterion

For optimal bet sizing under uncertainty:

```
f* = (bp - q) / b
```

where:
- `f*` = fraction of bankroll to bet
- `b` = odds received on bet
- `p` = probability of winning
- `q` = 1 - p = probability of losing

The framework implements **fractional Kelly** for risk management:

```python
bet_fraction = kelly_fraction * kelly_optimal
```

### Convergence Criteria

Monte Carlo simulations terminate when:

```
|μ_window1 - μ_window2| / |μ_window1| < tolerance
```

Ensures sufficient samples while avoiding unnecessary computation.

## 🔒 Code Quality

### Linting and Type Checking

```bash
# Format code
black src/python/

# Sort imports
isort src/python/

# Check style
flake8 src/python/

# Type check
mypy src/python/core/

# Run pylint
pylint src/python/core/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## 📈 Performance

The framework supports efficient parallel execution:

- **Multiprocessing**: Distributes simulations across CPU cores
- **Proper seed management**: Each worker gets unique seed for statistical independence
- **Progress tracking**: Real-time progress updates during execution
- **Adaptive convergence**: Stops early when results have converged

Benchmarks on Intel Core i7 (8 cores):
- **10,000 simulations**: ~5 seconds (parallel) vs ~45 seconds (sequential)
- **100,000 simulations**: ~45 seconds (parallel) vs ~450 seconds (sequential)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests and documentation
4. Ensure all tests pass (`pytest tests/`)
5. Run linters (`black`, `isort`, `flake8`, `mypy`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

### Monte Carlo Methods
- Metropolis, N., & Ulam, S. (1949). *The Monte Carlo method*. Journal of the American Statistical Association.
- Robert, C., & Casella, G. (2004). *Monte Carlo Statistical Methods*.

### Bayesian Inference
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.).
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.

### Kelly Criterion
- Kelly, J. L. (1956). *A new interpretation of information rate*. Bell System Technical Journal.
- Thorp, E. O. (2006). *The Kelly Criterion in Blackjack Sports Betting, and the Stock Market*.

### Software Engineering
- Martin, R. C. (2008). *Clean Code: A Handbook of Agile Software Craftsmanship*.
- Gamma, E., et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*.

## 🙏 Acknowledgments

This framework was built following industry best practices and recommendations from:
- PEP 8 (Style Guide for Python Code)
- PEP 257 (Docstring Conventions)
- PEP 484 (Type Hints)
- SimDesign: TQMP Monte Carlo Simulation Guidelines
- EPA Risk Assessment Guidelines

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**Made with ❤️ for rigorous probability analysis**

