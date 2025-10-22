# Contributing to Applied Probability Framework

Thank you for your interest in contributing! This document provides guidelines for contributing to the framework.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/applied-probability-framework.git
cd applied-probability-framework
```

### 2. Set Up Development Environment

```bash
cd src/python
pip install -r requirements.txt
pip install -e ".[dev]"  # Install development dependencies
```

### 3. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Use prefixes:
- `feature/` for new features
- `fix/` for bug fixes
- `docs/` for documentation
- `test/` for test improvements
- `refactor/` for code refactoring

### 2. Make Changes

Follow these guidelines:

**Code Style:**
- Follow PEP 8 (enforced by Black and Flake8)
- Maximum line length: 120 characters
- Use meaningful variable names
- Keep functions focused and small

**Type Hints:**
```python
from typing import List, Dict, Optional, Any

def process_data(data: List[float], config: Optional[Dict[str, Any]] = None) -> float:
    """Process data and return result."""
    pass
```

**Docstrings (PEP 257):**
```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description of function.
    
    More detailed description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input provided
    """
    pass
```

### 3. Write Tests

**Test Coverage Requirements:**
- New code must have >80% test coverage
- All public APIs must be tested
- Include edge cases and error conditions

**Example Test:**
```python
import pytest
from core.base_estimator import BetaEstimator

def test_beta_estimator_initialization():
    """Test that BetaEstimator initializes correctly."""
    estimator = BetaEstimator(alpha=2.0, beta=3.0)
    assert estimator.alpha == 2.0
    assert estimator.beta == 3.0

def test_beta_estimator_update():
    """Test posterior update."""
    estimator = BetaEstimator(alpha=1.0, beta=1.0)
    estimator.update(successes=5, failures=3)
    
    expected = (1+5) / (1+5+1+3)  # 6/10 = 0.6
    assert abs(estimator.estimate() - expected) < 1e-10
```

### 4. Run Tests Locally

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=core --cov=cli --cov-report=html

# Specific test file
pytest tests/test_base_estimator.py -v
```

### 5. Run Linters

```bash
# Format code
black src/python/

# Sort imports
isort src/python/

# Check style
flake8 src/python/

# Type check
mypy src/python/core/

# Code quality
pylint src/python/core/
```

Or use pre-commit:
```bash
pre-commit run --all-files
```

### 6. Commit Changes

```bash
git add .
git commit -m "feat: add Thompson Sampling strategy"
```

**Commit Message Format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `ci`: CI/CD changes

**Example:**
```
feat: add Thompson Sampling strategy

Implement Thompson Sampling for multi-armed bandit problems using
Beta posteriors. Includes comprehensive tests and documentation.

Closes #123
```

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Pull Request Guidelines

### PR Checklist

- [ ] Code follows style guidelines (Black, Flake8)
- [ ] All tests pass (`pytest tests/`)
- [ ] Test coverage >80% for new code
- [ ] Documentation updated (docstrings, README, API docs)
- [ ] Type hints added for all public APIs
- [ ] Commit messages follow format
- [ ] No merge conflicts with main branch
- [ ] CI/CD pipeline passes

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code formatted
- [ ] Type hints added
```

## Adding New Features

### Adding a New Simulator

1. Create class inheriting from `BaseSimulator`:
```python
from core.base_simulator import BaseSimulator

class MySimulator(BaseSimulator):
    """Docstring describing the simulator."""
    
    def reset(self):
        """Initialize state."""
        pass
    
    def step(self, action):
        """Execute action."""
        pass
    
    def get_valid_actions(self):
        """Return available actions."""
        pass
    
    def get_state_representation(self):
        """Return numerical state."""
        pass
```

2. Register the simulator:
```python
from core.plugin_system import register_simulator

@register_simulator("my_simulator")
class MySimulator(BaseSimulator):
    pass
```

3. Write tests in `tests/test_my_simulator.py`

4. Add documentation to `docs/`

### Adding a New Strategy

Similar process as simulators, inherit from `BaseStrategy`.

### Adding Documentation

- **API changes**: Update `docs/API_REFERENCE.md`
- **New tutorials**: Add to `docs/TUTORIALS.md`
- **Theory**: Add to `docs/THEORETICAL_BACKGROUND.md`
- **Examples**: Add to README.md

## Reporting Issues

### Bug Reports

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/traceback
- Minimal code example

### Feature Requests

Include:
- Use case description
- Proposed API/interface
- Why this is useful
- Alternatives considered

## Questions?

- Open an issue with the "question" label
- Check existing documentation
- Review closed issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🎉

