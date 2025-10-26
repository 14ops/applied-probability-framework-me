# How to Enable Q-Learning for Strategies

## 🎯 Quick Start

### Option 1: Use Pre-Made Q-Learning Strategies (Easiest)

The game now includes **Q-Learning enabled** versions of strategies:

1. **Run the game:**
   ```bash
   python src/python/gui_game.py
   ```

2. **Select a Q-Learning strategy** from the dropdown:
   - `⚡ Takeshi (Q-Learning)` - Adaptive Takeshi with Q-learning
   - `⚡ Senku (Q-Learning)` - Adaptive Senku with Q-learning

3. **Click "Show Learning Matrices"** to see the Q-values evolve in real-time!

4. **Play games** and watch the matrices learn optimal moves

5. **Save progress** using "Save Learning Progress" button

---

## 📊 What You'll See

When you use Q-Learning strategies:

### Q-Learning Matrix Tab
- **State hashes** - Unique game situations the AI has encountered
- **Q-values** - Learned action values (green = good, red = bad)
- **Visit counts** - How often each state-action pair was explored
- **Epsilon** - Exploration rate (decreases over time)

### Experience Replay Tab
- **Recent experiences** - Stored game transitions
- **Priorities** - Important experiences highlighted in gold
- **Rewards** - Outcomes from each action

### Learning Stats Tab
- **Total states visited**
- **Update count**
- **Average rewards**
- **Learning progress**

---

## 🔧 Create Your Own Q-Learning Strategy

To convert any strategy to use Q-learning, follow this template:

```python
from core.adaptive_strategy import AdaptiveLearningStrategy
from core.plugin_system import register_strategy

@register_strategy("my_learning_strategy")
class MyAdaptiveStrategy(AdaptiveLearningStrategy):
    """Your strategy with Q-Learning enabled."""
    
    def __init__(self, config: dict = None):
        super().__init__(
            name="My Learning Strategy",
            config=config or {},
            use_q_learning=True,        # ✅ Enable Q-learning
            use_experience_replay=True,  # ✅ Enable replay buffer
            use_parameter_evolution=False # Optional
        )
        
        # Your custom parameters
        self.your_param = config.get('your_param', 0.5)
    
    def select_action_heuristic(self, state: dict, valid_actions: list):
        """
        Your strategy logic when not using Q-learning.
        This is automatically blended with Q-learning based on q_learning_weight.
        """
        # Your strategy code here
        return random.choice(valid_actions)
```

Then import it in `gui_game.py`:
```python
import my_adaptive_strategy  # Add this import
```

---

## 🎮 How Q-Learning Works

The adaptive strategies use **blend mode** by default:

1. **Early games (exploration):** Uses heuristic strategy + some random exploration
2. **Learning phase:** Gradually learns which actions lead to better rewards
3. **Late games (exploitation):** Relies more on learned Q-values

You can control the blend with `q_learning_weight` in config:
- `0.0` = Pure heuristic (no learning)
- `0.5` = 50/50 blend (default)
- `1.0` = Pure Q-learning (fully learned)

---

## 💾 Save & Load Progress

### Save Learning Progress
1. Play several games with a Q-learning strategy
2. Click **"Save Learning Progress"** button
3. Choose a directory (default: `data/learning_progress/`)
4. Files saved:
   - `q_matrix.json` - All learned Q-values
   - `replay_buffer.json` - Experience replay data
   - `strategy.json` - Strategy configuration

### Load Learning Progress
1. Click **"Load Learning Progress"** button
2. Select the directory with saved files
3. The AI continues learning from where it left off!

---

## 📈 Tips for Best Results

1. **Play many games** - Q-learning needs experience (50-100 games minimum)
2. **Enable Auto-Run** - Let the AI play automatically to learn faster
3. **Watch the matrices** - Open "Show Learning Matrices" to see learning in real-time
4. **Save regularly** - Don't lose your training progress!
5. **Adjust parameters** - Use the "Strategy Tuning" sliders to experiment

---

## 🐛 Troubleshooting

### "Q-Learning not enabled for this strategy"
- Make sure you selected a **⚡ Q-Learning** strategy from the dropdown
- Regular strategies (without ⚡) don't have Q-learning enabled
- Use `adaptive_takeshi` or `adaptive_senku` instead

### Empty matrices
- The AI needs to play games first to build the Q-matrix
- Start a game in "AI Plays" mode and let it run
- Check back after 5-10 games

### Slow learning
- Q-learning takes time (50-100 games to see patterns)
- Use Auto-Run to speed up training
- Lower exploration (epsilon) if you want more exploitation

---

## 📚 Advanced: Parameter Evolution

To enable **genetic algorithm** parameter evolution:

```python
parameter_ranges = {
    'aggression_threshold': (0.3, 0.7),
    'risk_tolerance': (0.1, 0.5),
}

super().__init__(
    name="Evolving Strategy",
    use_q_learning=True,
    use_parameter_evolution=True,  # ✅ Enable evolution
    parameter_ranges=parameter_ranges
)
```

This will evolve the parameters over generations to find optimal values!

---

## 🎯 Example Files

- `src/python/adaptive_takeshi.py` - Full example with Q-learning
- `src/python/core/adaptive_strategy.py` - Base class documentation
- `docs/evolution/GUIDE.md` - Detailed evolution documentation

---

## 🚀 Next Steps

1. **Try the Q-learning strategies** - See them learn in real-time
2. **Create your own** - Use the template above
3. **Share your results** - Compare learned strategies!

Happy learning! 🤖📊

