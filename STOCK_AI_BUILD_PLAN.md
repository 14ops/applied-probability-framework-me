# Stock AI Framework Build Plan

## Current Status

✅ **Infrastructure**: Directory structure exists
❌ **Implementation**: Empty files, needs to be built from scratch

## Priority Implementation Order

### Phase 1: Basic Data Loading (30 min)
- Create `stock_ai/backtests/data_loader.py`
- Implement `load_ohlcv()` using yfinance
- Test with AAPL data

### Phase 2: Simple MA Crossover Strategy (1 hour)
- Create `stock_ai/strategies/ma_crossover.py`
- Implement `generate_signals()` with fast/slow MA
- Add basic buy/sell logic

### Phase 3: Backtest Runner (2 hours)
- Create `stock_ai/backtests/backtest_runner.py`
- Implement position tracking and equity calculation
- Return metrics: CAGR, Sharpe, Max DD

### Phase 4: Visualization (1 hour)
- Create `stock_ai/backtests/visualizer.py`
- Plot equity curve and drawdowns
- Save plots to results/

### Phase 5: Requirements File (15 min)
- Update `stock_ai/requirements.txt` with needed packages
- yfinance, pandas, numpy, matplotlib, loguru

Let's start building!

