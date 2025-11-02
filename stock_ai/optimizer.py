"""
Parameter Optimizer

Automatically searches for optimal strategy parameters using grid search or optimization.
"""

import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
from typing import List, Dict, Tuple, Callable
from loguru import logger
import yaml
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backtests.backtest_runner import run_backtest
from backtests.data_loader import load_ohlcv
from strategies import ma_crossover, rsi_filter, bollinger_reversion


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return {}


def optimize_ma_crossover(
    symbol: str,
    fast_range: List[int] = [5, 10, 15, 20],
    slow_range: List[int] = [30, 50, 75, 100, 200],
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    starting_cash: float = 10000.0
) -> pd.DataFrame:
    """
    Optimize MA crossover parameters using grid search.
    
    Args:
        symbol: Stock symbol
        fast_range: List of fast MA periods to test
        slow_range: List of slow MA periods to test
        start_date: Backtest start date
        end_date: Backtest end date
        starting_cash: Initial capital
    
    Returns:
        DataFrame with optimization results
    """
    logger.info(f"Optimizing MA crossover for {symbol}...")
    
    # Load data
    df = load_ohlcv(symbol, start_date, end_date)
    
    if df.empty:
        logger.error(f"No data for {symbol}")
        return pd.DataFrame()
    
    # Grid search
    results = []
    
    for fast, slow in product(fast_range, slow_range):
        if fast >= slow:
            continue  # Skip invalid combinations
        
        try:
            # Generate signals
            df_signals = ma_crossover.generate_signals(df.copy(), fast=fast, slow=slow)
            
            # Run backtest
            bt_df, metrics = run_backtest(df_signals, starting_cash=starting_cash)
            
            # Store results
            results.append({
                'symbol': symbol,
                'strategy': 'ma_crossover',
                'fast': fast,
                'slow': slow,
                'total_return_pct': metrics.get('total_return_pct', 0),
                'cagr_pct': metrics.get('cagr_pct', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
                'num_trades': metrics.get('num_trades', 0),
                'win_rate_pct': metrics.get('win_rate_pct', 0)
            })
            
            logger.info(f"Fast={fast}, Slow={slow}: Sharpe={metrics.get('sharpe_ratio', 0):.3f}, Return={metrics.get('total_return_pct', 0):.2f}%")
            
        except Exception as e:
            logger.error(f"Error with fast={fast}, slow={slow}: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        logger.warning("No valid results")
        return pd.DataFrame()
    
    # Sort by Sharpe ratio (descending)
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    return results_df


def optimize_rsi_filter(
    symbol: str,
    fast_range: List[int] = [5, 10, 15, 20],
    slow_range: List[int] = [30, 50, 75, 100],
    rsi_threshold_range: List[float] = [50, 55, 60, 65, 70],
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    starting_cash: float = 10000.0
) -> pd.DataFrame:
    """
    Optimize RSI filter parameters using grid search.
    
    Args:
        symbol: Stock symbol
        fast_range: List of fast MA periods
        slow_range: List of slow MA periods
        rsi_threshold_range: List of RSI thresholds
        start_date: Backtest start date
        end_date: Backtest end date
        starting_cash: Initial capital
    
    Returns:
        DataFrame with optimization results
    """
    logger.info(f"Optimizing RSI filter for {symbol}...")
    
    # Load data
    df = load_ohlcv(symbol, start_date, end_date)
    
    if df.empty:
        logger.error(f"No data for {symbol}")
        return pd.DataFrame()
    
    # Import RSI filter
    from strategies.rsi_filter import generate_signals
    
    # Grid search
    results = []
    
    for fast, slow, rsi_threshold in product(fast_range, slow_range, rsi_threshold_range):
        if fast >= slow:
            continue
        
        try:
            # Generate signals
            df_signals = generate_signals(df.copy(), fast=fast, slow=slow, rsi_threshold=rsi_threshold)
            
            # Run backtest
            bt_df, metrics = run_backtest(df_signals, starting_cash=starting_cash)
            
            # Store results
            results.append({
                'symbol': symbol,
                'strategy': 'rsi_filter',
                'fast': fast,
                'slow': slow,
                'rsi_threshold': rsi_threshold,
                'total_return_pct': metrics.get('total_return_pct', 0),
                'cagr_pct': metrics.get('cagr_pct', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
                'num_trades': metrics.get('num_trades', 0),
                'win_rate_pct': metrics.get('win_rate_pct', 0)
            })
            
        except Exception as e:
            logger.error(f"Error with fast={fast}, slow={slow}, rsi={rsi_threshold}: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        logger.warning("No valid results")
        return pd.DataFrame()
    
    # Sort by Sharpe ratio
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    return results_df


def optimize_bollinger_bands(
    symbol: str,
    window_range: List[int] = [10, 15, 20, 25, 30],
    std_range: List[float] = [1.5, 2.0, 2.5, 3.0],
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    starting_cash: float = 10000.0
) -> pd.DataFrame:
    """
    Optimize Bollinger Bands parameters using grid search.
    
    Args:
        symbol: Stock symbol
        window_range: List of BB windows
        std_range: List of standard deviations
        start_date: Backtest start date
        end_date: Backtest end date
        starting_cash: Initial capital
    
    Returns:
        DataFrame with optimization results
    """
    logger.info(f"Optimizing Bollinger Bands for {symbol}...")
    
    # Load data
    df = load_ohlcv(symbol, start_date, end_date)
    
    if df.empty:
        logger.error(f"No data for {symbol}")
        return pd.DataFrame()
    
    # Import Bollinger strategy
    from strategies.bollinger_reversion import generate_signals
    
    # Grid search
    results = []
    
    for window, std in product(window_range, std_range):
        try:
            # Generate signals
            df_signals = generate_signals(df.copy(), bollinger_window=window, bollinger_std=std)
            
            # Run backtest
            bt_df, metrics = run_backtest(df_signals, starting_cash=starting_cash)
            
            # Store results
            results.append({
                'symbol': symbol,
                'strategy': 'bollinger_reversion',
                'bollinger_window': window,
                'bollinger_std': std,
                'total_return_pct': metrics.get('total_return_pct', 0),
                'cagr_pct': metrics.get('cagr_pct', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
                'num_trades': metrics.get('num_trades', 0),
                'win_rate_pct': metrics.get('win_rate_pct', 0)
            })
            
        except Exception as e:
            logger.error(f"Error with window={window}, std={std}: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        logger.warning("No valid results")
        return pd.DataFrame()
    
    # Sort by Sharpe ratio
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    return results_df


def save_best_params(results_df: pd.DataFrame, output_path: str = "config/best_params.yaml", top_n: int = 5):
    """Save top N parameter sets to YAML file."""
    if results_df.empty:
        logger.warning("No results to save")
        return
    
    top_params = results_df.head(top_n)
    
    # Convert to configuration format
    config = {}
    
    strategy = top_params.iloc[0]['strategy']
    
    if strategy == 'ma_crossover':
        best = top_params.iloc[0]
        config['strategy'] = {
            'name': 'ma_crossover',
            'params': {
                'fast': int(best['fast']),
                'slow': int(best['slow'])
            }
        }
    
    elif strategy == 'rsi_filter':
        best = top_params.iloc[0]
        config['strategy'] = {
            'name': 'rsi_filter',
            'params': {
                'fast': int(best['fast']),
                'slow': int(best['slow']),
                'rsi_threshold': float(best['rsi_threshold'])
            }
        }
    
    elif strategy == 'bollinger_reversion':
        best = top_params.iloc[0]
        config['strategy'] = {
            'name': 'bollinger_reversion',
            'params': {
                'bollinger_window': int(best['bollinger_window']),
                'bollinger_std': float(best['bollinger_std'])
            }
        }
    
    # Save to file
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved best parameters to {output_path}")
    
    # Print summary
    print(f"\nTop {top_n} Parameter Sets:")
    print(top_params[['fast', 'slow', 'sharpe_ratio', 'total_return_pct', 'max_drawdown_pct']].to_string(index=False))


def run_optimization(
    symbol: str,
    strategy: str,
    param_ranges: Dict = None,
    save_results: bool = True,
    output_dir: str = "results"
) -> pd.DataFrame:
    """
    Run optimization for a specific strategy.
    
    Args:
        symbol: Stock symbol
        strategy: Strategy name ('ma_crossover', 'rsi_filter', 'bollinger_reversion')
        param_ranges: Custom parameter ranges
        save_results: Whether to save results
        output_dir: Directory to save results
    
    Returns:
        DataFrame with optimization results
    """
    logger.info(f"Running optimization for {symbol} using {strategy}")
    
    # Run optimization based on strategy
    if strategy == 'ma_crossover':
        fast_range = param_ranges.get('fast', [5, 10, 15, 20]) if param_ranges else [5, 10, 15, 20]
        slow_range = param_ranges.get('slow', [30, 50, 75, 100, 200]) if param_ranges else [30, 50, 75, 100, 200]
        results_df = optimize_ma_crossover(symbol, fast_range=fast_range, slow_range=slow_range)
    
    elif strategy == 'rsi_filter':
        fast_range = param_ranges.get('fast', [5, 10, 15, 20]) if param_ranges else [5, 10, 15, 20]
        slow_range = param_ranges.get('slow', [30, 50, 75, 100]) if param_ranges else [30, 50, 75, 100]
        rsi_range = param_ranges.get('rsi_threshold', [50, 55, 60, 65, 70]) if param_ranges else [50, 55, 60, 65, 70]
        results_df = optimize_rsi_filter(symbol, fast_range=fast_range, slow_range=slow_range, rsi_threshold_range=rsi_range)
    
    elif strategy == 'bollinger_reversion':
        window_range = param_ranges.get('bollinger_window', [10, 15, 20, 25, 30]) if param_ranges else [10, 15, 20, 25, 30]
        std_range = param_ranges.get('bollinger_std', [1.5, 2.0, 2.5, 3.0]) if param_ranges else [1.5, 2.0, 2.5, 3.0]
        results_df = optimize_bollinger_bands(symbol, window_range=window_range, std_range=std_range)
    
    else:
        logger.error(f"Unknown strategy: {strategy}")
        return pd.DataFrame()
    
    # Save results
    if save_results and not results_df.empty:
        output_path = Path(output_dir) / f"optimization_{symbol}_{strategy}.csv"
        output_path.parent.mkdir(exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
        
        # Save best parameters
        save_best_params(results_df, f"config/best_params_{strategy}.yaml")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("--strategy", default="ma_crossover", choices=['ma_crossover', 'rsi_filter', 'bollinger_reversion'],
                       help="Strategy to optimize")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()
    
    print(f"Optimizing {args.strategy} for {args.symbol}...")
    results_df = run_optimization(args.symbol, args.strategy, save_results=True, output_dir=args.output_dir)
    
    if not results_df.empty:
        print("\nOptimization complete!")
        print(f"\nTop 5 results:\n{results_df.head(5)}")
    else:
        print("Optimization failed or no results")

