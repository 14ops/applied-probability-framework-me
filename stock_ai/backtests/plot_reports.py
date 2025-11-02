"""
Plot Reports Generator

Creates visualization reports from backtest results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from loguru import logger
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    logger.warning("matplotlib not installed. Install with: pip install matplotlib")
    plt = None


def load_backtest_results(results_dir: str = "results") -> pd.DataFrame:
    """Load all backtest result files."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return pd.DataFrame()
    
    # Load all backtest CSV files
    backtest_files = list(results_path.glob("backtest_*.csv"))
    
    if not backtest_files:
        logger.warning("No backtest files found")
        return pd.DataFrame()
    
    results = {}
    for file in backtest_files:
        ticker = file.stem.replace("backtest_", "")
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            results[ticker] = df
            logger.info(f"Loaded {file.name}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading {file.name}: {e}")
    
    return results


def plot_equity_curves(
    backtest_results: Dict[str, pd.DataFrame],
    output_path: str = "reports/equity_curves.png"
):
    """
    Plot equity curves for multiple tickers.
    
    Args:
        backtest_results: Dictionary mapping ticker to DataFrame
        output_path: Path to save the plot
    """
    if plt is None:
        logger.error("matplotlib not installed")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for ticker, df in backtest_results.items():
        if 'equity' in df.columns:
            ax.plot(df.index, df['equity'], label=ticker, linewidth=2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity ($)')
    ax.set_title('Equity Curves Comparison')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved equity curves to {output_path}")
    
    plt.close()


def plot_drawdown(
    backtest_results: Dict[str, pd.DataFrame],
    output_path: str = "reports/drawdown_comparison.png"
):
    """
    Plot drawdown comparison for multiple tickers.
    
    Args:
        backtest_results: Dictionary mapping ticker to DataFrame
        output_path: Path to save the plot
    """
    if plt is None:
        logger.error("matplotlib not installed")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for ticker, df in backtest_results.items():
        if 'drawdown' in df.columns:
            # Convert to percentage
            drawdown_pct = df['drawdown'] * 100
            ax.plot(df.index, drawdown_pct, label=ticker, linewidth=2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown Comparison')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved drawdown comparison to {output_path}")
    
    plt.close()


def plot_strategy_components(
    df: pd.DataFrame,
    ticker: str,
    output_path: str = None
):
    """
    Plot price action with strategy indicators.
    
    Args:
        df: DataFrame with OHLCV and strategy signals
        ticker: Ticker symbol
        output_path: Path to save the plot (optional)
    """
    if plt is None:
        logger.error("matplotlib not installed")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Price with MA and signals
    ax1.plot(df.index, df['close'], label='Close', color='black', linewidth=1.5)
    
    if 'ma_fast' in df.columns:
        ax1.plot(df.index, df['ma_fast'], label='Fast MA', alpha=0.7)
    
    if 'ma_slow' in df.columns:
        ax1.plot(df.index, df['ma_slow'], label='Slow MA', alpha=0.7)
    
    if 'bb_upper' in df.columns:
        ax1.plot(df.index, df['bb_upper'], label='BB Upper', alpha=0.3, color='orange')
        ax1.plot(df.index, df['bb_lower'], label='BB Lower', alpha=0.3, color='orange')
        if 'bb_middle' in df.columns:
            ax1.plot(df.index, df['bb_middle'], label='BB Middle', alpha=0.3, color='orange')
    
    # Mark buy/sell signals
    buy_signals = df[df['signal'] > 0]
    sell_signals = df[df['signal'] < 0]
    
    if len(buy_signals) > 0:
        ax1.scatter(buy_signals.index, buy_signals['close'], 
                   marker='^', color='green', s=100, label='Buy Signal', zorder=5)
    
    if len(sell_signals) > 0:
        ax1.scatter(sell_signals.index, sell_signals['close'],
                   marker='v', color='red', s=100, label='Sell Signal', zorder=5)
    
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{ticker} - Price with Strategy Signals')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # Plot 2: Equity curve
    if 'equity' in df.columns:
        ax2.plot(df.index, df['equity'], label='Equity', color='blue', linewidth=2)
        ax2.set_ylabel('Equity ($)')
        ax2.set_xlabel('Date')
        ax2.set_title('Equity Curve')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#f8f9fa')
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    if output_path:
        Path(output_path).parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved strategy plot to {output_path}")
    
    plt.close()


def create_performance_comparison_table(
    summary_df: pd.DataFrame,
    output_path: str = "reports/performance_comparison.png"
):
    """
    Create a table visualization of performance metrics.
    
    Args:
        summary_df: DataFrame with summary metrics
        output_path: Path to save the plot
    """
    if plt is None:
        logger.error("matplotlib not installed")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f1f1f2')
    
    plt.title('Backtest Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved performance table to {output_path}")
    
    plt.close()


def generate_all_reports(results_dir: str = "results", output_dir: str = "reports"):
    """
    Generate all reports from backtest results.
    
    Args:
        results_dir: Directory containing backtest CSV files
        output_dir: Directory to save reports
    """
    logger.info("Generating visualization reports...")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load backtest results
    backtest_results = load_backtest_results(results_dir)
    
    if not backtest_results:
        logger.error("No backtest results to plot")
        return
    
    # Load summary if available
    summary_path = Path(results_dir) / "summary.csv"
    summary_df = None
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        logger.info(f"Loaded summary from {summary_path}")
    
    # Generate plots
    if backtest_results:
        # Equity curves comparison
        plot_equity_curves(backtest_results, f"{output_dir}/equity_curves.png")
        
        # Drawdown comparison
        plot_drawdown(backtest_results, f"{output_dir}/drawdown_comparison.png")
        
        # Individual strategy plots
        for ticker, df in backtest_results.items():
            plot_path = f"{output_dir}/strategy_{ticker}.png"
            plot_strategy_components(df, ticker, plot_path)
        
        # Performance comparison table
        if summary_df is not None:
            create_performance_comparison_table(summary_df, f"{output_dir}/performance_comparison.png")
    
    logger.info("All reports generated successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate backtest visualization reports")
    parser.add_argument("--results-dir", default="results", help="Directory with backtest CSV files")
    parser.add_argument("--output-dir", default="reports", help="Directory to save reports")
    args = parser.parse_args()
    
    generate_all_reports(args.results_dir, args.output_dir)

