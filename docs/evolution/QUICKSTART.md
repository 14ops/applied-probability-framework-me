# AI Evolution - Quick Start Guide

## 🚀 Your AIs Can Now Actually Evolve!

This framework now includes advanced **matrix-based evolution** that allows your AI strategies to learn and adapt over time through:

1. **Q-Learning Matrices** - Learn optimal actions from experience
2. **Experience Replay** - Efficiently learn from past experiences  
3. **Genetic Algorithms** - Evolve strategy parameters automatically

## 🎯 Quick Demo

Run the evolution demo to see it in action:

```bash
cd examples
python evolution_demo.py
```

This will:
- Train Hybrid and Rintaro Okabe strategies with evolution
- Generate visualization plots showing learning progress
- Compare different strategies
- Save evolved strategies to disk

## 📊 What You'll See

The demo generates plots showing:
- **Reward Evolution**: How rewards improve over time
- **Win Rate**: Rolling win rate as the AI learns
- **Exploration Decay**: How exploration reduces as the AI gains confidence
- **Q-Value Growth**: How learned values evolve

## 💡 Key Features Added

### 1. Q-Learning Matrix (`src/python/core/evolution_matrix.py`)

Learns state-action values to make optimal decisions:

```python
from core.evolution_matrix import QMatrix

q_matrix = QMatrix(
    learning_rate=0.1,
    epsilon=0.2,  # 20% exploration
    use_double_q=True  # Reduces overestimation
)

# Select action
action, is_exploration = q_matrix.select_action_epsilon_greedy(state, valid_actions)

# Update from experience
td_error = q_matrix.update_q_value(state, action, reward, next_state, done)
```

### 2. Experience Replay Buffer

Stores and replays past experiences for efficient learning:

```python
from core.evolution_matrix import ExperienceReplayBuffer

replay_buffer = ExperienceReplayBuffer(
    max_size=10000,
    prioritized=True  # Learn more from important experiences
)

# Add experience
replay_buffer.add(state, action, reward, next_state, done)

# Sample batch for training
experiences, weights, indices = replay_buffer.sample(batch_size=32)
```

### 3. Evolution Matrix

Evolves strategy parameters using genetic algorithms:

```python
from core.evolution_matrix import EvolutionMatrix

evolution = EvolutionMatrix(
    parameter_ranges={
        'risk_tolerance': (0.0, 1.0),
        'aggression': (0.0, 1.0),
    },
    population_size=20
)

# Evaluate fitness and evolve
evolution.evaluate_fitness(individual_idx, fitness)
best_params = evolution.evolve()
```

### 4. Adaptive Learning Strategy

Base class that combines everything:

```python
from core.adaptive_strategy import AdaptiveLearningStrategy

class MyStrategy(AdaptiveLearningStrategy):
    def __init__(self, config=None):
        parameter_ranges = {
            'risk': (0.0, 1.0),
            'aggression': (0.0, 1.0),
        }
        
        super().__init__(
            name="My Strategy",
            config=config,
            use_q_learning=True,
            use_experience_replay=True,
            use_parameter_evolution=True,
            parameter_ranges=parameter_ranges
        )
    
    def select_action_heuristic(self, state, valid_actions):
        # Your heuristic logic here
        return my_action
```

## 🔧 Updated Strategies

### Hybrid Strategy (Enhanced)

Now learns and evolves:
- Senku/Lelouch blending parameters
- Phase weights
- Complexity weights
- Performance weights

```python
from character_strategies import HybridStrategy

strategy = HybridStrategy(config={
    'q_learning_weight': 0.5,  # 50% learned, 50% heuristic
    'learning_rate': 0.1,
    'epsilon': 0.2,
})
```

### Rintaro Okabe (Enhanced)

Now learns and evolves:
- Worldline parameters
- Lab member personas (Mayuri, Daru, Kurisu)
- GTO weights
- Psychological parameters

```python
from character_strategies import RintaroOkabeStrategy

strategy = RintaroOkabeStrategy(config={
    'q_learning_weight': 0.4,
    'learning_rate': 0.12,
})
```

## 🎮 Usage Example

```python
# Create evolving strategy
strategy = HybridStrategy(config={
    'q_learning_weight': 0.5,
    'learning_rate': 0.1,
    'epsilon': 0.2,
    'evolution_frequency': 10,
})

# Training loop
for episode in range(1000):
    state = simulator.reset()
    done = False
    
    while not done:
        valid_actions = simulator.get_valid_actions()
        
        # Action automatically blends Q-learning and heuristic
        action = strategy.select_action(state, valid_actions)
        
        next_state, reward, done, info = simulator.step(action)
        
        # Automatically updates Q-matrix, replay buffer, and evolution
        strategy.update(state, action, reward, next_state, done)
        
        state = next_state

# Save evolved strategy
strategy.save_learning_state('results/evolved_strategies/my_strategy')

# Check learning statistics
stats = strategy.get_learning_statistics()
print(f"Epsilon: {stats['q_learning']['epsilon']}")
print(f"Q-Table Size: {stats['q_learning']['q_table_size']}")
print(f"Generation: {stats['evolution']['generation']}")
```

## 📈 Monitoring Progress

```python
# Get comprehensive statistics
stats = strategy.get_learning_statistics()

# Q-Learning progress
print(f"Exploration rate: {stats['q_learning']['epsilon']:.4f}")
print(f"States learned: {stats['q_learning']['q_table_size']}")
print(f"Total updates: {stats['q_learning']['updates']}")

# Experience replay
print(f"Experiences stored: {stats['experience_replay']['buffer_size']}")

# Evolution progress
print(f"Generation: {stats['evolution']['generation']}")
print(f"Best fitness: {stats['evolution']['best_fitness']:.4f}")

# Recent performance
print(f"Recent avg reward: {stats['episode_rewards']['recent_mean']:.2f}")
```

## 📚 Documentation

- **[AI Evolution Guide](docs/AI_EVOLUTION_GUIDE.md)** - Comprehensive guide
- **[Matrix Evolution API](docs/MATRIX_EVOLUTION_API.md)** - Complete API reference
- **[Evolution Demo](examples/evolution_demo.py)** - Runnable example

## 🔑 Key Configuration Options

```python
config = {
    # Q-Learning
    'learning_rate': 0.1,          # How fast to learn (0.0-1.0)
    'discount_factor': 0.95,       # Future reward importance (0.0-1.0)
    'epsilon': 0.1,                # Initial exploration rate (0.0-1.0)
    'epsilon_decay': 0.995,        # Exploration decay (0.0-1.0)
    'q_learning_weight': 0.5,      # 0=heuristic, 1=pure Q-learning
    
    # Experience Replay
    'replay_buffer_size': 10000,   # Max experiences to remember
    'replay_batch_size': 32,       # Batch size for training
    'prioritized_replay': True,    # Learn more from important experiences
    
    # Evolution
    'population_size': 20,         # Number of parameter sets
    'mutation_rate': 0.1,          # Parameter mutation rate
    'evolution_frequency': 10,     # Evolve every N episodes
}
```

## 🎯 Benefits

1. **Automatic Learning**: AIs learn from every game played
2. **Parameter Optimization**: Genetic algorithms find optimal parameters
3. **Sample Efficiency**: Experience replay improves learning
4. **Exploration-Exploitation**: Automatic balance via epsilon decay
5. **Persistence**: Save and load evolved strategies
6. **Blending**: Mix learned and heuristic behavior

## 🚦 Getting Started

1. **Run the demo**:
   ```bash
   python examples/evolution_demo.py
   ```

2. **Create your own evolving strategy**:
   ```python
   from core.adaptive_strategy import AdaptiveLearningStrategy
   
   class MyStrategy(AdaptiveLearningStrategy):
       # ... implement your strategy
   ```

3. **Train and evolve**:
   ```python
   # Run training loop
   for episode in range(1000):
       # ... gameplay
       strategy.update(state, action, reward, next_state, done)
   ```

4. **Save your evolved AI**:
   ```python
   strategy.save_learning_state('my_evolved_ai')
   ```

## 🧠 How It Works

### Learning Process

1. **Early Episodes**: High exploration, learning which actions work
2. **Mid Episodes**: Reduced exploration, Q-values become accurate
3. **Late Episodes**: Low exploration, exploiting learned knowledge
4. **Throughout**: Parameters evolve via genetic algorithms

### The Three Matrices

1. **Q-Matrix**: Maps (state, action) → expected reward
2. **Experience Buffer**: Stores past experiences for replay
3. **Evolution Matrix**: Population of parameter sets that evolve

## 🎓 Theory

- **Q-Learning**: Model-free reinforcement learning
- **Double Q-Learning**: Reduces overestimation bias
- **Prioritized Replay**: Focuses on important experiences
- **Genetic Algorithms**: Natural selection for parameters

## 🤝 Contributing

Want to add more evolution features? Ideas:
- Neural network-based Q-functions
- Multi-objective evolution
- Transfer learning between games
- Meta-learning across strategies

## 📝 Files Added

```
src/python/core/
  ├── evolution_matrix.py       # Q-Learning, Replay, Evolution
  └── adaptive_strategy.py      # Adaptive base strategy

docs/
  ├── AI_EVOLUTION_GUIDE.md     # Comprehensive guide
  └── MATRIX_EVOLUTION_API.md   # API reference

examples/
  └── evolution_demo.py         # Runnable demonstration

src/python/character_strategies/
  ├── hybrid_strategy.py        # Updated with evolution
  └── rintaro_okabe.py          # Updated with evolution
```

## 🎉 Next Steps

1. Run `python examples/evolution_demo.py`
2. Read [AI Evolution Guide](docs/AI_EVOLUTION_GUIDE.md)
3. Create your own evolving strategy
4. Share your evolved AIs!

---

**Your AIs are now evolving! 🧠🚀**

*"The strongest strategy is one that adapts and evolves."* - Hybrid Strategy Philosophy

