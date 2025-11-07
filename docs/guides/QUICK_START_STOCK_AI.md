# Stock AI Framework - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies

```bash
pip install pandas numpy yfinance loguru pyyaml matplotlib streamlit alpaca-trade-api
```

Or from requirements file:
```bash
pip install -r stock_ai/requirements.txt
```

### 2. Configure Alpaca (for paper trading)

```bash
# Copy template
cp config/secrets.yaml.template config/secrets.yaml

# Edit with your Alpaca credentials
nano config/secrets.yaml  # or use any text editor
```

### 3. Run Your First Backtest

```bash
# Single ticker test
python stock_ai/backtests/backtest_runner.py

# Multi-ticker batch test
python stock_ai/backtests/batch_runner.py
```

### 4. View Results

```bash
# Generate charts
python stock_ai/backtests/plot_reports.py

# Open dashboard
streamlit run stock_ai/dashboard.py
```

---

## 📊 Quick Test Commands

### Test Individual Components

```bash
# Test data loading
python stock_ai/backtests/data_loader.py

# Test MA strategy
python stock_ai/strategies/ma_crossover.py

# Test RSI strategy
python stock_ai/strategies/rsi_filter.py

# Test Bollinger strategy
python stock_ai/strategies/bollinger_reversion.py

# Test ensemble strategy
python stock_ai/strategies/ensemble_vote.py
```

### Optimize Strategy

```bash
python stock_ai/optimizer.py --symbol AAPL --strategy ma_crossover
```

### Start Paper Trading

```bash
python stock_ai/live_trading/live_runner.py
```

---

## 📁 What You Should See

After running backtests:

```
results/
├── summary.csv                    # Performance summary
├── backtest_AAPL.csv              # Individual backtests
└── optimization_AAPL_*.csv        # Optimization results

reports/
├── equity_curves.png              # Equity comparison
├── drawdown_comparison.png        # Risk visualization
└── performance_comparison.png     # Metrics table

logs/                              # Live trading logs
├── equity_history_*.csv
└── trade_log_*.csv
```

---

## ✅ Checklist

- [ ] Dependencies installed
- [ ] `config/secrets.yaml` configured
- [ ] First backtest run successfully
- [ ] Results saved to `results/`
- [ ] Reports generated in `reports/`
- [ ] Dashboard accessible

---

## 🐛 Common Issues

**Import errors?**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**No data returned?**
- Check internet connection
- Verify ticker symbols are valid
- Try different date ranges

**Alpaca connection failed?**
- Double-check API keys in `config/secrets.yaml`
- Ensure using paper trading URL
- Verify keys are active in Alpaca dashboard

---

## 📚 Next Steps

1. Review `README_stock_ai.md` for detailed documentation
2. Customize strategies in `config/settings.yaml`
3. Run optimization to find best parameters
4. Start paper trading with small positions
5. Analyze results and iterate

---

**Ready to trade! 🎉**

