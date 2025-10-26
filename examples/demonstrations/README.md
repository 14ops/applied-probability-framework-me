# Demonstration Scripts

This directory contains scripts that demonstrate and explain the AI evolution system.

## Scripts

### 🧠 **evolution_demo.py**
Full demonstration of the AI evolution system.

**Features:**
- Q-Learning matrix training
- Experience replay in action
- Parameter evolution via genetic algorithms
- Visualization of learning progress
- Comparison of multiple strategies

**Usage:**
```bash
python evolution_demo.py
```

**Output:**
- Learning curves
- Win rate evolution
- Q-value convergence plots
- Generation-by-generation improvement

---

### 🎲 **explain_win_rates.py**
Mathematical explanation of why win rates are low (and why that's good!).

**Shows:**
- Theoretical maximum calculation (0.043%)
- Why our AIs do 10-20x better
- Mine density progression
- Comparison to casino games
- Risk-reward analysis

**Usage:**
```bash
python explain_win_rates.py
```

---

### 🏆 **show_champion.py**
Visual championship display with charts.

**Requirements:** matplotlib

**Features:**
- Championship podium
- Performance comparisons
- Interactive charts

**Usage:**
```bash
python show_champion.py
```

---

### 📊 **show_champion_text.py**
Text-based championship display (no dependencies).

**Features:**
- ASCII art championship display
- Performance rankings
- Statistical analysis
- No external dependencies

**Usage:**
```bash
python show_champion_text.py
```

**Perfect for:**
- Quick results viewing
- Environments without matplotlib
- Terminal-only systems

---

## Quick Start

```bash
cd examples/demonstrations

# See the champion instantly
python show_champion_text.py

# Understand the math
python explain_win_rates.py

# Full evolution demo (if matplotlib installed)
python evolution_demo.py
```

## What You'll Learn

1. **Evolution Demo**: How AI strategies learn and improve
2. **Win Rates**: Why 0.87% is actually extraordinary
3. **Champion**: Who won and why (Hybrid Ultimate!)
4. **Mathematics**: The probability theory behind it all

