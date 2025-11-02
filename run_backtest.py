#!/usr/bin/env python3
"""
Quick wrapper to run backtests from project root.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "stock_ai"))

# Now import and run
from stock_ai.backtests.backtest_runner import *
from stock_ai.backtests.data_loader import load_ohlcv
from stock_ai.strategies.ma_crossover import generate_signals

if __name__ == "__main__":
    print("Running backtest...")
    df = load_ohlcv("AAPL", "2022-01-01", "2023-01-01")
    
    if not df.empty:
        df = generate_signals(df, fast=10, slow=30)
        bt_df, metrics = run_backtest(df, starting_cash=10000)
        
        print_metrics(metrics)
        print(f"\nFirst 10 trades and equity:")
        print(bt_df[['close', 'signal', 'equity', 'drawdown']].head(10))
    else:
        print("Failed to load data")

