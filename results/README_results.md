# Results Documentation

This directory contains raw simulation outputs and tournament results for reproducibility and analysis.

## Directory Structure

```
results/
├── README_results.md              # This file
├── results_mines_conservative.json
├── results_mines_random.json
├── tournament_results/
│   ├── README.md
│   └── SUMMARY.md
└── [additional result files]
```

## Result File Format

### Standard Result JSON Schema

```json
{
  "strategy_name": "TakeshiStrategy",
  "configuration": {
    "base_bet": 10.0,
    "target_clicks": 8,
    "initial_bankroll": 1000.0
  },
  "simulation_parameters": {
    "num_simulations": 50000,
    "seed": 42,
    "board_size": 5,
    "mine_count": 2
  },
  "results": {
    "games_played": 50000,
    "wins": 22665,
    "losses": 27335,
    "win_rate": 0.4533,
    "total_profit": 2050.50,
    "net_profit_pct": 4.10,
    "max_drawdown": -250.30,
    "final_bankroll": 1205.50,
    "avg_profit_per_game": 0.041,
    "median_profit": 11.20,
    "profit_std_dev": 15.67
  },
  "payout_profile": {
    "1_click": {"count": 0, "wins": 0, "avg_payout": 0},
    "8_clicks": {"count": 50000, "wins": 22665, "avg_payout": 2.12}
  },
  "timestamp": "2025-10-26T10:30:00Z",
  "version": "1.0.0"
}
```

### Required Fields

Every result file MUST include:

1. **`strategy_name`**: Name of strategy used
2. **`win_rate_estimate`**: Empirical win rate (wins / total games)
3. **`payout_profile`**: Distribution of click counts and payouts
4. **`rounds`**: Number of simulations run
5. **`seed`**: Random seed for reproducibility
6. **`net_profit`**: Total profit/loss across all games
7. **`max_drawdown`**: Largest loss from peak bankroll

## Regenerating Results

### Individual Strategy Results

```bash
# Takeshi (Aggressive Doubling)
python -c "
from strategies import TakeshiStrategy
from game_simulator import run_simulation

strategy = TakeshiStrategy(config={
    'base_bet': 10.0,
    'target_clicks': 8,
    'initial_bankroll': 1000.0
})

results = run_simulation(
    strategy=strategy,
    num_games=50000,
    seed=42
)

import json
with open('results/takeshi_50k.json', 'w') as f:
    json.dump(results, f, indent=2)
"

# Yuzu (Controlled Chaos)
python -c "
from strategies import YuzuStrategy
from game_simulator import run_simulation

strategy = YuzuStrategy(config={
    'base_bet': 10.0,
    'target_clicks': 7
})

results = run_simulation(strategy=strategy, num_games=50000, seed=42)

import json
with open('results/yuzu_50k.json', 'w') as f:
    json.dump(results, f, indent=2)
"
```

### Full Tournament Results

```bash
# Run complete character tournament
python examples/tournaments/character_tournament.py \
  --num-games 1000000 \
  --seed 42 \
  --output results/character_tournament_1M.json

# Fast tournament (50K games)
python examples/tournaments/instant_tournament.py
```

### Reproducibility Commands

To exactly reproduce the committed results:

#### Conservative Strategy (10K rounds)
```bash
cd src/python
python -c "
import sys
import json
import random
from game_simulator import GameSimulator
from character_strategies.kazuya_kinoshita import KazuyaKinoshitaStrategy

random.seed(12345)
simulator = GameSimulator(board_size=5, mine_count=2)
strategy = KazuyaKinoshitaStrategy({'base_bet': 10.0})

results = []
for i in range(10000):
    result = simulator.run_game(strategy)
    results.append(result)

wins = sum(1 for r in results if r['win'])
total_profit = sum(r['profit'] for r in results)

output = {
    'strategy_name': 'conservative',
    'rounds': 10000,
    'seed': 12345,
    'win_rate_estimate': wins / 10000,
    'net_profit': total_profit,
    'payout_profile': {
        '5_clicks': {
            'count': 10000,
            'wins': wins,
            'avg_payout': 1.53
        }
    }
}

with open('../../results/results_mines_conservative.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f'Conservative: {wins} wins / 10000 games = {wins/100:.2f}% win rate')
print(f'Net profit: \${total_profit:.2f}')
"
```

#### Random Strategy (10K rounds)
```bash
cd src/python
python -c "
import sys
import json
import random
from game_simulator import GameSimulator
from strategies.base import SimpleRandomStrategy

random.seed(67890)
simulator = GameSimulator(board_size=5, mine_count=2)
strategy = SimpleRandomStrategy({'base_bet': 10.0})

results = []
for i in range(10000):
    result = simulator.run_game(strategy)
    results.append(result)

wins = sum(1 for r in results if r['win'])
total_profit = sum(r['profit'] for r in results)

output = {
    'strategy_name': 'random',
    'rounds': 10000,
    'seed': 67890,
    'win_rate_estimate': wins / 10000,
    'net_profit': total_profit,
    'payout_profile': 'random'
}

with open('../../results/results_mines_random.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f'Random: {wins} wins / 10000 games')
"
```

## Analysis Scripts

### Calculate Summary Statistics

```python
import json
import glob

results_files = glob.glob('results/*.json')

for filepath in results_files:
    with open(filepath) as f:
        data = json.load(f)
    
    print(f"\n{data['strategy_name']}:")
    print(f"  Games: {data['rounds']}")
    print(f"  Win Rate: {data['win_rate_estimate']:.2%}")
    print(f"  Net Profit: ${data['net_profit']:.2f}")
    print(f"  ROI: {data['net_profit']/data['rounds']*100:.2f}%")
```

### Compare Strategies

```python
import json
import pandas as pd

strategies = ['takeshi', 'yuzu', 'aoi', 'kazuya', 'lelouch']
data = []

for strategy in strategies:
    with open(f'results/{strategy}_50k.json') as f:
        result = json.load(f)
        data.append({
            'Strategy': result['strategy_name'],
            'Win Rate': result['win_rate_estimate'],
            'Net Profit': result['net_profit'],
            'ROI': result['net_profit'] / result['rounds']
        })

df = pd.DataFrame(data)
df = df.sort_values('ROI', ascending=False)
print(df.to_string(index=False))
```

## Verification

### Validate Result File

```python
def validate_result_file(filepath):
    """Validate that a result file contains all required fields."""
    with open(filepath) as f:
        data = json.load(f)
    
    required = [
        'strategy_name',
        'rounds',
        'seed',
        'win_rate_estimate',
        'net_profit',
        'payout_profile'
    ]
    
    missing = [field for field in required if field not in data]
    
    if missing:
        print(f"❌ Missing fields: {missing}")
        return False
    else:
        print(f"✅ Valid result file")
        return True

# Usage
validate_result_file('results/takeshi_50k.json')
```

### Check Consistency

```python
def check_consistency(filepath):
    """Check internal consistency of results."""
    with open(filepath) as f:
        data = json.load(f)
    
    # Check win rate calculation
    results = data.get('results', {})
    stated_win_rate = data.get('win_rate_estimate', 0)
    calculated_win_rate = results.get('wins', 0) / results.get('games_played', 1)
    
    if abs(stated_win_rate - calculated_win_rate) < 0.0001:
        print(f"✅ Win rate consistent: {stated_win_rate:.4f}")
    else:
        print(f"❌ Win rate mismatch: {stated_win_rate} vs {calculated_win_rate}")
    
    # Check profit calculation
    # ... additional checks

# Usage
check_consistency('results/takeshi_50k.json')
```

## CI Integration

Results are automatically validated in CI pipeline:

```yaml
# .github/workflows/validate_results.yml
name: Validate Results

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Validate result files
        run: |
          python scripts/validate_results.py
```

## Adding New Results

When adding new simulation results:

1. **Run simulation** with documented seed
2. **Generate JSON** using standard schema
3. **Validate** using `validate_result_file()`
4. **Commit** with descriptive message
5. **Document** regeneration command in this file

### Commit Message Template

```
Add simulation results: [Strategy Name]

- Strategy: [name]
- Games: [count]
- Seed: [seed]
- Win Rate: [rate]
- Net Profit: [profit]

Regenerate with:
python [command]
```

## Notes on Randomness

- All results use **Python's random module** with explicit seeds
- Mines game uses **uniform random distribution** for mine placement
- Seeds ensure **deterministic reproducibility**
- Different Python versions may produce different results even with same seed

## Contact

For questions about results or simulation methodology:
- Open an issue on GitHub
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines

---

*Last Updated: October 2025*

