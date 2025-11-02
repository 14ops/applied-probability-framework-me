"""
Batch Runner for Multi-Ticker Backtests

Runs backtests across multiple symbols and generates summary reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from loguru import logger
import yaml
import sys

from backtests.backtest_runner import run_backtest, calculate_metrics
from backtests.data_loader import load_ohlcv
from strategies import ma_crossover, rsi_filter, bollinger_reversion, ensemble_vote


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {
            'tickers': ['AAPL', 'MSFT', 'NVDA', 'SPY', 'TSLA'],
            'backtest': {
                'starting_cash': 10000.0,
                'start_date': '2022-01-01',
                'end_date': '2023-12-31'
            },
            'strategy': {
                'name': 'ma_crossover',
                'params': {'fast': 10, 'slow': 30}
            }
        }


def get_strategy_function(strategy_name: str):
    """Get the strategy signal generation function."""
    strategies = {
        'ma_crossover': ma_crossover.generate_signals,
        'rsi_filter': rsi_filter.generate_signals,
        'bollinger_reversion': bollinger_reversion.generate_signals,
        'ensemble_vote': ensemble_vote.generate_signals
    }
    
    if strategy_name not in strategies:
        logger.error(f"Unknown strategy: {strategy_name}. Using ma_crossover.")
        return ma_crossover.generate_signals
    
    return strategies[strategy_name]


def run_batch_backtest(
    tickers: List[str],
    strategy_name: str,
    strategy_params: dict,
    config: dict
) -> pd.DataFrame:
    """
    Run backtest on multiple tickers and return summary.
    
    Args:
        tickers: List of ticker symbols
        strategy_name: Name of strategy to use
        strategy_params: Strategy-specific parameters
        config: Full configuration dictionary
    
    Returns:
        DataFrame with summary metrics per ticker
    """
    logger.info(f"Running batch backtest for {len(tickers)} tickers using {strategy_name}")
    
    backtest_config = config.get('backtest', {})
    start_date = backtest_config.get('start_date', '2022-01-01')
    end_date = backtest_config.get('end_date', '2023-12-31')
    
    strategy_func = get_strategy_function(strategy_name)
    results = []
    
    for ticker in tickers:
        logger.info(f"Processing {ticker}...")
        
        try:
            # Load data
            df = load_ohlcv(ticker, start_date, end_date)
            
            if df.empty:
                logger.warning(f"No data for {ticker}, skipping...")
                continue
            
            # Generate signals
            df = strategy_func(df, **strategy_params)
            
            # Run backtest
            bt_df, metrics = run_backtest(
                df,
                starting_cash=backtest_config.get('starting_cash', 10000),
                commission=backtest_config.get('commission', 0.001),
                max_position_pct=backtest_config.get('max_position_pct', 1.0),
                slippage=backtest_config.get('slippage', 0.0005)
            )
            
            # Save individual backtest
            if config.get('results', {}).get('save_equity', True):
                results_dir = Path(config.get('results', {}).get('directory', 'results'))
                results_dir.mkdir(exist_ok=True)
                output_path = results_dir / f"backtest_{ticker}.csv"
                bt_df.to_csv(output_path)
                logger.info(f"Saved: {output_path}")
            
            # Add ticker and strategy to metrics
            metrics['ticker'] = ticker
            metrics['strategy'] = strategy_name
            
            results.append(metrics)
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            continue
    
    # Create summary DataFrame
    if not results:
        logger.error("No successful backtests!")
        return pd.DataFrame()
    
    summary_df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    cols = ['ticker', 'strategy', 'num_trades', 'win_rate_pct', 'total_return_pct',
            'cagr_pct', 'sharpe_ratio', 'max_drawdown_pct']
    summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
    
    return summary_df


def save_summary(summary_df: pd.DataFrame, output_path: str = "results/summary.csv"):
    """Save summary DataFrame to CSV."""
    Path(output_path).parent.mkdir(exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    logger.info(f"Summary saved to {output_path}")
    print(f"\nSummary saved to {output_path}")
    print(summary_df.to_string(index=False))


def print_summary_stats(summary_df: pd.DataFrame):
    """Print summary statistics across all tickers."""
    if summary_df.empty:
        return
    
    print("\n" + "="*80)
    print("BATCH BACKTEST SUMMARY")
    print("="*80)
    
    # Average metrics
    avg_metrics = summary_df.select_dtypes(include=[np.number]).mean()
    
    print("\nAverage Performance Across All Tickers:")
    print(f"  Win Rate:        {avg_metrics.get('win_rate_pct', 0):.2f}%")
    print(f"  CAGR:            {avg_metrics.get('cagr_pct', 0):.2f}%")
    print(f"  Sharpe Ratio:    {avg_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown:    {avg_metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Total Return:    {avg_metrics.get('total_return_pct', 0):.2f}%")
    
    # Best and worst performers
    if 'total_return_pct' in summary_df.columns:
        best = summary_df.loc[summary_df['total_return_pct'].idxmax()]
        worst = summary_df.loc[summary_df['total_return_pct'].idxmin()]
        
        print(f"\nBest Performer:  {best['ticker']} ({best['total_return_pct']:.2f}%)")
        print(f"Worst Performer: {worst['ticker']} ({worst['total_return_pct']:.2f}%)")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    logger.info("Starting batch backtest runner...")
    
    # Load configuration
    config = load_config()
    
    # Get parameters
    tickers = config.get('tickers', ['AAPL'])
    strategy_config = config.get('strategy', {})
    strategy_name = strategy_config.get('name', 'ma_crossover')
    strategy_params = strategy_config.get('params', {})
    
    # Run batch backtest
    summary_df = run_batch_backtest(tickers, strategy_name, strategy_params, config)
    
    # Save and display results
    if not summary_df.empty:
        save_summary(summary_df)
        print_summary_stats(summary_df)
    else:
        logger.error("No results to display!")
        sys.exit(1)

