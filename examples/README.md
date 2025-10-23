# Examples

This directory contains example scripts and notebooks demonstrating the framework's capabilities.

## Quick Demo

Run the quick demo to see the framework in action:

```bash
cd applied-probability-framework-me
python examples/quick_demo.py
```

### What the demo shows:

1. **Bayesian Inference** - How posterior beliefs update with evidence
2. **Sequential Learning** - Convergence to true probabilities
3. **Thompson Sampling** - Optimal exploration-exploitation for multi-armed bandits
4. **Kelly Criterion** - Optimal bet sizing strategies

### Expected Output

The demo will print statistics showing:
- Bayesian posterior estimates with confidence intervals
- Convergence behavior as data accumulates
- Thompson Sampling arm selection performance
- Kelly Criterion bankroll growth comparison

## More Examples

Check out the main documentation for more detailed examples:

- [Tutorials](../docs/TUTORIALS.md) - Step-by-step guides
- [API Reference](../docs/API_REFERENCE.md) - Complete API documentation
- [Quick Start](../QUICKSTART.md) - 5-minute getting started guide

## Creating Your Own Examples

```python
import sys
sys.path.append('src/python')

from core.config import create_default_config
from core.parallel_engine import ParallelSimulationEngine
from core.plugin_system import get_registry

# Your code here!
```

For more complex examples, see the test suite in `src/python/tests/`.

