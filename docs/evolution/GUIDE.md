# AI Evolution Guide

## Overview

The Applied Probability Framework now includes advanced **matrix-based evolution** capabilities that allow your AI strategies to learn and adapt over time. This guide explains how to use these powerful features.

## Key Components

### 1. Q-Learning Matrix (`QMatrix`)

The Q-Learning Matrix enables strategies to learn optimal action-value functions through experience.

**Features:**
- Dynamic state space using efficient hashing
- Double Q-Learning to reduce overestimation bias
- Adaptive learning rate based on visit counts
- Epsilon-greedy exploration-exploitation balance
- Automatic epsilon decay
- Save/load functionality for persistent learning

**How it works:**
- Maps state-action pairs to expected rewards
- Updates Q-values based on temporal difference learning
- Balances exploration (trying new actions) and exploitation (using learned actions)
- Automatically reduces exploration over time as the AI learns

### 2. Experience Replay Buffer (`ExperienceReplayBuffer`)

The Experience Replay Buffer stores past experiences for more efficient learning.

**Features:**
- Prioritized experience replay (learns more from important experiences)
- Configurable buffer size with automatic eviction
- Importance sampling for unbiased learning
- Batch sampling for training efficiency

**How it works:**
- Stores transitions (state, action, reward, next_state)
- Assigns priorities based on temporal difference errors
- Samples batches of experiences for training
- Allows the AI to learn from past experiences multiple times

### 3. Evolution Matrix (`EvolutionMatrix`)

The Evolution Matrix evolves strategy parameters using genetic algorithms.

**Features:**
- Population-based parameter optimization
- Crossover and mutation operators
- Fitness-based selection
- Elitism (preserving top performers)
- Adaptive mutation rates

**How it works:**
- Maintains a population of parameter sets
- Evaluates fitness of each parameter set through gameplay
- Selects best performers for breeding
- Creates new parameter sets through crossover and mutation
- Evolves parameters to maximize performance

### 4. Adaptive Learning Strategy (`AdaptiveLearningStrategy`)

The base class that ties everything together, allowing strategies to seamlessly blend learned and heuristic behavior.

**Features:**
- Configurable Q-learning weight (blend learned/heuristic)
- Automatic matrix updates during gameplay
- Integrated experience replay
- Parameter evolution
- Save/load learning state
- Comprehensive learning statistics

## Using Evolution in Your Strategies

### Quick Start

```python
from core.adaptive_strategy import AdaptiveLearningStrategy

class MyEvolvingStrategy(AdaptiveLearningStrategy):
    def __init__(self, name="My Strategy", config=None):
        # Define evolvable parameters
        parameter_ranges = {
            'risk_tolerance': (0.0, 1.0),
            'aggression': (0.0, 1.0),
            'exploration_bonus': (0.0, 0.5),
        }
        
        # Initialize with evolution enabled
        super().__init__(
            name=name,
            config=config,
            use_q_learning=True,
            use_experience_replay=True,
            use_parameter_evolution=True,
            parameter_ranges=parameter_ranges
        )
        
        # Initialize your strategy-specific parameters
        self.risk_tolerance = config.get('risk_tolerance', 0.5) if config else 0.5
        self.aggression = config.get('aggression', 0.5) if config else 0.5
        self.exploration_bonus = config.get('exploration_bonus', 0.2) if config else 0.2
    
    def select_action_heuristic(self, state, valid_actions):
        """Implement your heuristic logic here."""
        # Your strategy logic using self.risk_tolerance, self.aggression, etc.
        # This will be automatically blended with Q-learning
        return my_heuristic_action
    
    def _apply_evolved_parameters(self, params):
        """Apply evolved parameters to your strategy."""
        super()._apply_evolved_parameters(params)
        self.risk_tolerance = params.get('risk_tolerance', self.risk_tolerance)
        self.aggression = params.get('aggression', self.aggression)
        self.exploration_bonus = params.get('exploration_bonus', self.exploration_bonus)
```

### Configuration Options

```python
config = {
    # Q-Learning
    'learning_rate': 0.1,          # How fast to update Q-values (0.0-1.0)
    'discount_factor': 0.95,       # Future reward discount (0.0-1.0)
    'epsilon': 0.1,                # Initial exploration rate (0.0-1.0)
    'epsilon_decay': 0.995,        # Epsilon decay per update (0.0-1.0)
    'use_double_q': True,          # Use Double Q-Learning
    'q_learning_weight': 0.5,      # Blend: 0=pure heuristic, 1=pure Q-learning
    
    # Experience Replay
    'replay_buffer_size': 10000,   # Max experiences to store
    'replay_batch_size': 32,       # Batch size for training
    'replay_frequency': 4,         # Replay every N updates
    'prioritized_replay': True,    # Use prioritized experience replay
    
    # Parameter Evolution
    'population_size': 20,         # Number of parameter sets
    'mutation_rate': 0.1,          # Probability of mutation
    'crossover_rate': 0.7,         # Probability of crossover
    'elitism_ratio': 0.2,          # Ratio of top performers to preserve
    'evolution_frequency': 10,     # Evolve every N episodes
}

strategy = MyEvolvingStrategy(config=config)
```

### Running Evolution

```python
# Run training loop
for episode in range(1000):
    state = simulator.reset()
    done = False
    
    while not done:
        # Get valid actions
        valid_actions = simulator.get_valid_actions()
        
        # Select action (automatically blends Q-learning and heuristic)
        action = strategy.select_action(state, valid_actions)
        
        # Take step
        next_state, reward, done, info = simulator.step(action)
        
        # Update strategy (automatically updates Q-matrix, replay buffer, evolution)
        strategy.update(state, action, reward, next_state, done)
        
        state = next_state
    
    # Print progress
    if (episode + 1) % 100 == 0:
        stats = strategy.get_learning_statistics()
        print(f"Episode {episode + 1}: {stats}")
```

### Saving and Loading

```python
# Save evolved strategy
strategy.save_learning_state('results/evolved_strategies/my_strategy')

# Load evolved strategy
strategy.load_learning_state('results/evolved_strategies/my_strategy')
```

## Examples

### Example 1: Hybrid Strategy

The Hybrid Strategy combines Senku's analytical approach with Lelouch's strategic thinking, and evolves the blending parameters:

```python
from character_strategies import HybridStrategy

strategy = HybridStrategy(config={
    'q_learning_weight': 0.5,      # 50% learned, 50% heuristic
    'learning_rate': 0.1,
    'epsilon': 0.2,
    'evolution_frequency': 10,
})

# Evolved parameters:
# - strategic_threshold: Controls Senku/Lelouch blend
# - w_phase: Weight for game phase in decision
# - w_complexity: Weight for board complexity
# - w_performance: Weight for recent performance
```

### Example 2: Rintaro Okabe Strategy

The Mad Scientist strategy evolves lab member personas and worldline parameters:

```python
from character_strategies import RintaroOkabeStrategy

strategy = RintaroOkabeStrategy(config={
    'q_learning_weight': 0.4,      # Reading Steiner favors heuristic
    'learning_rate': 0.12,
    'evolution_frequency': 10,
})

# Evolved parameters:
# - delta: GTO weight
# - epsilon: Psychological weight
# - mayuri_risk, mayuri_aggression: Mayuri's persona
# - daru_risk, daru_aggression: Daru's persona
# - kurisu_risk, kurisu_aggression: Kurisu's persona
```

### Running the Demo

```bash
cd examples
python evolution_demo.py
```

This will:
1. Run evolution experiments with multiple strategies
2. Generate visualization plots
3. Save evolved strategies to disk
4. Compare strategy performance

## Understanding the Learning Process

### Phase 1: Exploration (Early Episodes)

- High epsilon (exploration rate)
- Q-matrix is sparse
- Strategies rely more on heuristics
- Experience buffer fills up

### Phase 2: Learning (Mid Episodes)

- Epsilon decays gradually
- Q-matrix grows and becomes more accurate
- Experience replay improves learning efficiency
- Parameters begin to evolve

### Phase 3: Exploitation (Late Episodes)

- Low epsilon (mostly exploitation)
- Q-matrix is well-populated
- Strategies use learned Q-values confidently
- Parameters converge to optimal values

## Monitoring Learning Progress

```python
# Get comprehensive statistics
stats = strategy.get_learning_statistics()

# Q-Learning stats
print(f"Epsilon: {stats['q_learning']['epsilon']}")
print(f"Q-Table Size: {stats['q_learning']['q_table_size']}")
print(f"Total Updates: {stats['q_learning']['updates']}")

# Experience Replay stats
print(f"Buffer Size: {stats['experience_replay']['buffer_size']}")
print(f"Total Experiences: {stats['experience_replay']['total_experiences']}")

# Evolution stats
print(f"Generation: {stats['evolution']['generation']}")
print(f"Best Fitness: {stats['evolution']['best_fitness']}")
print(f"Avg Fitness: {stats['evolution']['avg_fitness']}")

# Episode stats
print(f"Recent Mean Reward: {stats['episode_rewards']['recent_mean']}")
print(f"Recent Std Dev: {stats['episode_rewards']['recent_std']}")
```

## Advanced Topics

### Custom State Hashing

By default, states are hashed using MD5. For custom state representations:

```python
def custom_state_hash(self, state):
    """Create custom hash for your state representation."""
    # Extract relevant features
    features = (state['clicks'], state['mines_remaining'], state['phase'])
    return str(hash(features))

# Override in your strategy
strategy.q_matrix.get_state_hash = custom_state_hash
```

### Multi-Objective Evolution

Define fitness functions that consider multiple objectives:

```python
def _calculate_fitness(self, rewards):
    """Custom fitness considering multiple objectives."""
    avg_reward = np.mean(rewards)
    win_rate = sum(1 for r in rewards if r > 0) / len(rewards)
    risk_adjusted = avg_reward * win_rate
    return risk_adjusted
```

### Transfer Learning

Transfer learning from one environment to another:

```python
# Train in environment A
strategy_a = MyStrategy()
# ... train ...
strategy_a.save_learning_state('env_a')

# Transfer to environment B
strategy_b = MyStrategy()
strategy_b.load_learning_state('env_a')
# Q-matrix and evolved parameters are preserved
```

## Performance Tips

1. **Start with lower Q-learning weight** (0.3-0.5) to blend heuristic and learned behavior
2. **Use prioritized experience replay** for faster learning
3. **Tune epsilon decay** to balance exploration/exploitation
4. **Adjust evolution frequency** based on episode length
5. **Save evolved strategies** regularly to preserve learning
6. **Use batch replay** for sample efficiency
7. **Monitor Q-table size** - very large tables may need pruning

## Troubleshooting

### Strategy not learning?
- Check that `update()` is being called after each step
- Verify rewards are non-zero
- Increase learning rate
- Reduce epsilon decay (explore more)

### Evolution not improving?
- Increase population size
- Adjust parameter ranges
- Increase evolution frequency
- Check fitness function

### Memory usage too high?
- Reduce replay buffer size
- Reduce population size
- Implement Q-table pruning for rarely-visited states

## Theory Background

### Q-Learning

Q-Learning learns an action-value function Q(s,a) that estimates the expected cumulative reward of taking action `a` in state `s`. The update rule is:

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:
- α = learning rate
- r = reward
- γ = discount factor
- s' = next state

### Double Q-Learning

Reduces overestimation by maintaining two Q-tables and using one to select actions and the other to evaluate them.

### Prioritized Experience Replay

Samples experiences proportionally to their temporal difference error, allowing the AI to learn more from surprising experiences.

### Genetic Algorithms

Evolves parameters through:
1. Selection: Choose best performers
2. Crossover: Combine parent parameters
3. Mutation: Random parameter changes
4. Elitism: Preserve top performers

## Next Steps

1. Run `examples/evolution_demo.py` to see evolution in action
2. Create your own evolving strategy
3. Experiment with different parameter ranges
4. Compare evolved vs non-evolved performance
5. Save and share your evolved strategies

## References

- Sutton & Barto (2018): "Reinforcement Learning: An Introduction"
- van Hasselt et al. (2016): "Deep Reinforcement Learning with Double Q-Learning"
- Schaul et al. (2016): "Prioritized Experience Replay"
- Mitchell (1998): "An Introduction to Genetic Algorithms"

---

**Your AIs can now truly evolve! 🧠🚀**

