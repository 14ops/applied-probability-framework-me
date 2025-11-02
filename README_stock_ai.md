# Stock AI Framework - Complete Guide

A comprehensive stock trading framework for backtesting, paper trading, and live execution with Alpaca.

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install pandas numpy yfinance loguru pyyaml matplotlib streamlit alpaca-trade-api

# Or install from requirements
pip install -r stock_ai/requirements.txt
```

### 2. Configuration

First, set up your configuration files:

```bash
# Copy secrets template
cp config/secrets.yaml.template config/secrets.yaml

# Edit config/secrets.yaml with your Alpaca credentials
# NEVER commit this file to git!
```

Edit `config/secrets.yaml`:
```yaml
alpaca:
  key_id: "YOUR_ALPACA_API_KEY_ID"
  secret_key: "YOUR_ALPACA_SECRET_KEY"
  base_url: "https://paper-api.alpaca.markets"  # Paper trading URL
```

### 3. Run Backtests

```bash
# Single ticker backtest
python stock_ai/backtests/backtest_runner.py

# Multi-ticker batch backtest
python stock_ai/backtests/batch_runner.py
```

Results will be saved to `results/` directory.

### 4. Generate Reports

```bash
# Create visualization reports
python stock_ai/backtests/plot_reports.py

# Open Streamlit dashboard
streamlit run stock_ai/dashboard.py
```

### 5. Paper Trading

```bash
# Start live paper trading (requires Alpaca credentials)
python stock_ai/live_trading/live_runner.py
```

### 6. Optimize Parameters

```bash
# Optimize MA crossover strategy
python stock_ai/optimizer.py --symbol AAPL --strategy ma_crossover

# Optimize RSI filter
python stock_ai/optimizer.py --symbol AAPL --strategy rsi_filter

# Optimize Bollinger Bands
python stock_ai/optimizer.py --symbol AAPL --strategy bollinger_reversion
```

---

## 📁 Project Structure

```
stock_ai/
├── backtests/
│   ├── backtest_runner.py     # Core backtest engine
│   ├── batch_runner.py        # Multi-ticker backtests
│   ├── data_loader.py         # Data loading utilities
│   └── plot_reports.py        # Visualization reports
├── strategies/
│   ├── ma_crossover.py        # Moving average crossover
│   ├── rsi_filter.py          # RSI momentum filter
│   ├── bollinger_reversion.py # Mean reversion strategy
│   └── ensemble_vote.py       # Ensemble voting strategy
├── live_trading/
│   └── live_runner.py         # Live/paper trading loop
├── api/
│   └── alpaca_interface.py    # Alpaca API wrapper
├── optimizer.py               # Parameter optimization
├── dashboard.py               # Streamlit dashboard
└── requirements.txt           # Dependencies

config/
├── settings.yaml              # Main configuration
├── secrets.yaml.template      # Secrets template
└── best_params.yaml          # Optimized parameters

results/                       # Backtest results
├── summary.csv               # Summary metrics
├── backtest_AAPL.csv        # Individual backtests
└── optimization_*.csv       # Optimization results

reports/                       # Visualizations
├── equity_curves.png
├── drawdown_comparison.png
└── performance_comparison.png

logs/                          # Live trading logs
├── equity_history_*.csv
└── trade_log_*.csv
```

---

## 📊 Strategies

### 1. Moving Average Crossover

Simple trend-following strategy:
- **Buy**: Fast MA crosses above slow MA
- **Sell**: Fast MA crosses below slow MA

```python
from strategies import ma_crossover
df = ma_crossover.generate_signals(df, fast=10, slow=30)
```

### 2. RSI Filter

Momentum filter on MA crossover:
- **Buy**: MA crossover long AND RSI > threshold
- **Sell**: MA crossover short OR RSI < threshold

```python
from strategies import rsi_filter
df = rsi_filter.generate_signals(df, fast=10, slow=30, rsi_threshold=55)
```

### 3. Bollinger Bands Reversion

Mean-reversion strategy:
- **Buy**: Price touches lower band (oversold)
- **Sell**: Price touches upper band (overbought)

```python
from strategies import bollinger_reversion
df = bollinger_reversion.generate_signals(df, bollinger_window=20, bollinger_std=2.0)
```

### 4. Ensemble Vote

Combines multiple strategies:
- Runs MA, RSI, and Bollinger strategies
- Votes on signals
- Takes position when min_votes strategies agree

```python
from strategies import ensemble_vote
df = ensemble_vote.generate_signals(df, fast=10, slow=30, rsi_threshold=55, min_votes=2)
```

---

## 🔧 Configuration

Edit `config/settings.yaml`:

```yaml
# Backtest parameters
backtest:
  starting_cash: 10000.0
  commission: 0.001
  max_position_pct: 1.0
  slippage: 0.0005
  start_date: "2022-01-01"
  end_date: "2023-12-31"

# Symbols to trade
tickers:
  - AAPL
  - MSFT
  - NVDA
  - SPY
  - TSLA

# Strategy configuration
strategy:
  name: "ma_crossover"  # Options: ma_crossover, rsi_filter, bollinger_reversion, ensemble_vote
  params:
    fast: 10
    slow: 30
    rsi_threshold: 55
    bollinger_window: 20
    bollinger_std: 2

# Live trading settings
live:
  enabled: false
  broker: "alpaca"
  paper: true
  check_interval: 300  # seconds
```

---

## 📈 Performance Metrics

The framework calculates comprehensive metrics:

- **Total Return**: Overall percentage return
- **CAGR**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Maximum equity decline
- **Win Rate**: Percentage of profitable trades
- **Number of Trades**: Trade frequency

Example output:
```
============================================================
BACKTEST RESULTS
============================================================
Starting Capital:  $10,000.00
Final Equity:      $12,450.32
Total Return:      24.50%
CAGR:              12.25%
Sharpe Ratio:      1.423
Max Drawdown:     -8.92%

Trades:
  Total Trades:    45
  Win Rate:        58.33%

Period:            2.00 years
============================================================
```

---

## 🤖 Live Trading

### Prerequisites

1. Alpaca account: https://app.alpaca.markets/
2. API keys added to `config/secrets.yaml`
3. Verified paper trading mode

### Running Live Trading

```bash
python stock_ai/live_trading/live_runner.py
```

The live runner will:
- Monitor market status
- Generate signals on schedule
- Execute trades automatically
- Track equity and positions
- Log all activity

### Safety Features

- ✅ Paper trading mode by default
- ✅ Risk management (position sizing)
- ✅ Commission and slippage modeling
- ✅ Automatic logging
- ✅ Market hours detection

---

## 🎯 Parameter Optimization

Optimize strategies to find best parameters:

### Grid Search

```bash
# MA Crossover optimization
python stock_ai/optimizer.py --symbol AAPL --strategy ma_crossover

# RSI Filter optimization
python stock_ai/optimizer.py --symbol AAPL --strategy rsi_filter

# Bollinger Bands optimization
python stock_ai/optimizer.py --symbol AAPL --strategy bollinger_reversion
```

Results saved to:
- `results/optimization_AAPL_ma_crossover.csv`
- `config/best_params_ma_crossover.yaml`

### Custom Ranges

Modify `optimizer.py` to adjust parameter ranges:

```python
results = optimize_ma_crossover(
    symbol="AAPL",
    fast_range=[5, 10, 15, 20, 25],
    slow_range=[30, 50, 75, 100, 150, 200]
)
```

---

## 📊 Dashboard

Interactive Streamlit dashboard:

```bash
streamlit run stock_ai/dashboard.py
```

Features:
- Backtest analysis
- Live trading monitor
- Strategy performance comparison
- Configuration viewer
- Equity curve plots
- Drawdown visualization

---

## 📝 Example Workflow

### Phase 1: Backtesting

```bash
# 1. Run initial backtest
python stock_ai/backtests/backtest_runner.py

# 2. Run multi-ticker batch
python stock_ai/backtests/batch_runner.py

# 3. Generate reports
python stock_ai/backtests/plot_reports.py
```

### Phase 2: Optimization

```bash
# 1. Optimize parameters
python stock_ai/optimizer.py --symbol AAPL --strategy ma_crossover

# 2. Review results
cat config/best_params_ma_crossover.yaml

# 3. Update settings.yaml with best params
```

### Phase 3: Paper Trading

```bash
# 1. Configure secrets.yaml
# 2. Start live runner
python stock_ai/live_trading/live_runner.py

# 3. Monitor in dashboard
streamlit run stock_ai/dashboard.py
```

---

## 🔒 Security

⚠️ **Important Security Notes:**

1. **Never commit** `config/secrets.yaml` to git
2. Always use **paper trading** for testing
3. Start with **small position sizes**
4. Test thoroughly in backtests first
5. Monitor live trades closely

Add to `.gitignore`:
```
config/secrets.yaml
*.csv
*.png
*.log
```

---

## 🐛 Troubleshooting

### Common Issues

**"No data returned for ticker"**
- Check internet connection
- Verify ticker symbol is valid
- Try different date range

**"Alpaca API connection failed"**
- Verify API keys in `config/secrets.yaml`
- Check base_url (paper vs live)
- Ensure API keys are active

**"Strategy module not found"**
- Ensure Python path includes `stock_ai`
- Verify `__init__.py` files exist
- Check imports in strategy files

---

## 📚 Next Steps

After completing the setup:

1. ✅ Run baseline backtests
2. ✅ Optimize parameters
3. ✅ Start paper trading
4. ✅ Analyze results
5. ⬜ Implement new strategies
6. ⬜ Add risk management rules
7. ⬜ Deploy to live trading

---

## 🤝 Contributing

Feel free to extend the framework:

- Add new strategies
- Implement advanced risk management
- Integrate additional brokers
- Build ML-based signals
- Add reinforcement learning

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Alpaca Markets** for brokerage API
- **yfinance** for market data
- **Streamlit** for dashboard framework

---

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review example backtests

---

**Happy Trading! 📈**

