# Dual-Mode Training Pipeline

Complete training system for Mines bot automation and Crypto/Stock market prediction.

## 🎮 Features

### Mines Bot Training
- 10,000 round simulations with 5 character strategies
- Evolving risk parameters (learns from wins/losses)
- Real-time leaderboard
- Character evolution tracking
- CSV logging and JSON strategy export

### Trading Bot Training
- ML-based price direction prediction
- Feature engineering (MA, RSI, MACD, Volatility)
- Random Forest classifier
- Trading simulation with stop-loss/take-profit
- Transaction fee modeling
- Portfolio tracking

## 🚀 Quick Start

### Option 1: Quick Test (Recommended for first run)

```bash
# From project root
python run_training.py --quick
```

This runs:
- 100 Mines rounds (fast)
- 30 days of trading data (fast)
- Both bots

### Option 2: Full Training

```bash
# Run both bots for full training
python run_training.py

# Run only Mines bot
python run_training.py --mode mines

# Run only Trading bot  
python run_training.py --mode trading
```

### Option 3: Direct Scripts

```bash
# Mines bot only
python src/python/train_bot_simulator.py

# Trading bot only
python src/python/train_trading_bot.py

# Both (orchestration)
python src/python/orchestrate_training.py
```

## 📊 Results

Results are saved in `results/` directory:

### Mines Bot
- `training_log_TIMESTAMP.csv` - Profit per round per agent
- `decisions_TIMESTAMP.csv` - All decisions made
- `strategy_evolution_TIMESTAMP.json` - Final agent configurations

### Trading Bot
- `trades_SYMBOL_TIMESTAMP.csv` - All trades executed
- `portfolio_SYMBOL_TIMESTAMP.csv` - Portfolio value over time
- `summary_SYMBOL_TIMESTAMP.json` - Performance summary

## ⚙️ Configuration

Edit `config.json` to customize:

### Mines Bot Settings
```json
{
  "mines_bot": {
    "total_rounds": 10000,          // Total rounds to simulate
    "initial_bankroll": 1000.0,     // Starting money per agent
    "characters": {
      "takeshi": {
        "risk_factor": 0.8,         // Risk tolerance (0-1)
        "cashout_threshold": 2.5,   // When to cash out (multiplier)
        "learning_rate": 0.01       // How fast to adapt
      }
    }
  }
}
```

### Trading Bot Settings
```json
{
  "trading_bot": {
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "lookback_days": 365,
    "model": {
      "type": "random_forest",
      "n_estimators": 100
    },
    "stop_loss_pct": 0.02,          // 2% stop loss
    "take_profit_pct": 0.04         // 4% take profit
  }
}
```

## 📈 Character Strategies

### Takeshi Kovacs (Aggressive)
- High risk tolerance
- Chases big payouts
- Adapts quickly after wins
- **Best for**: High variance scenarios

### Kazuya Kinoshita (Conservative)
- Low risk tolerance
- Safety-first approach
- Early cash-outs
- **Best for**: Capital preservation

### Lelouch vi Britannia (Strategic)
- Balanced risk/reward
- Long-term planning
- Strategic depth
- **Best for**: Complex decisions

### Senku Ishigami (Analytical)
- Data-driven
- EV maximization
- Precise calculations
- **Best for**: Optimization

### Yuzu (Balanced)
- Moderate risk
- Risk management focus
- Calculated approach
- **Best for**: Steady growth

## 🔄 Learning Loop

Each agent adapts based on results:

```
Win Round → Increase risk_factor (+0.01)
           → Increase aggression_factor (+0.005)

Lose Round → Decrease risk_factor (-0.02)
            → Decrease aggression_factor (-0.01)
```

This creates dynamic strategy evolution over 10,000 rounds.

## 📦 Dependencies

Install required packages:

```bash
pip install numpy scipy pandas matplotlib yfinance scikit-learn
```

Or from requirements:
```bash
pip install -r src/python/requirements.txt
pip install yfinance scikit-learn
```

## 🎯 Example Output

### Mines Training
```
======================================================================
MINES BOT TRAINING ENGINE
======================================================================
Configuration: 5x5 board
Total rounds: 10000
Characters: takeshi, kazuya, lelouch, senku, yuzu
======================================================================

Progress: 1000/10000 (10.0%)

--- LEADERBOARD ---
Rank   Agent           Profit       Win%     Rounds   Bankroll
----------------------------------------------------------------------
1      lelouch         $510.00      81.6%    49       $1510.00
2      yuzu            $510.00      86.4%    44       $1510.00
3      takeshi         $500.00      68.6%    70       $1500.00
4      kazuya          $500.00      76.4%    55       $1500.00
5      senku           $500.00      80.0%    50       $1500.00
```

### Trading Results
```
--- Trading Results ---
Initial Capital: $10,000.00
Final Capital: $10,420.00
Total Return: 4.20%
Total Trades: 45
Wins: 28, Losses: 17
Win Rate: 62.22%
```

## 🔧 Advanced Usage

### Custom Configuration

Create your own config:
```bash
cp config.json my_config.json
# Edit my_config.json
python orchestrate_training.py --config my_config.json
```

### Individual Components

Run just the Mines simulator:
```python
from src.python.train_bot_simulator import MinesTrainingEngine
engine = MinesTrainingEngine('my_config.json')
engine.train()
```

Run just the Trading bot:
```python
from src.python.train_trading_bot import TradingTrainingEngine
engine = TradingTrainingEngine('my_config.json')
engine.train_and_simulate('BTCUSD')
```

## 📝 Notes

- All results saved to `results/` directory
- Models saved to `models/` directory (if enabled)
- Training runs completely locally (no external API calls during simulation)
- Trading bot fetches data from yfinance on first run
- Synthetic data generated if API fails

## 🐛 Troubleshooting

**Error: config.json not found**
- Run from project root directory

**Error: Module not found**
- Install dependencies: `pip install numpy pandas matplotlib yfinance scikit-learn`

**Trading bot can't fetch data**
- Check internet connection
- Bot will generate synthetic data automatically

**Training too slow**
- Use `--quick` flag for testing
- Reduce `total_rounds` in config

## 🚀 Next Steps

1. Review results in `results/` directory
2. Analyze which character performed best
3. Adjust risk parameters in config.json
4. Re-run training with new parameters
5. Compare performance across runs

## 📞 Support

For issues or questions:
- Check logs in `logs/` directory
- Review `results/` for outputs
- See `config.json` for configuration

Enjoy training! 🎉

