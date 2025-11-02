"""
Visualization tools for backtesting results.

Creates equity curves, drawdown plots, and performance charts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional


def plot_equity_curve(
    df: pd.DataFrame,
    title: str = "Equity Curve",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot the equity curve from backtest results.
    
    Args:
        df: DataFrame with 'equity' column
        title: Plot title
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    if df.empty or 'equity' not in df.columns:
        print("No equity data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df.index, df['equity'], linewidth=2, label='Equity')
    ax.fill_between(df.index, df['equity'], alpha=0.3)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Equity ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved equity curve to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_drawdown(
    df: pd.DataFrame,
    title: str = "Drawdown Curve",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot the drawdown curve from backtest results.
    
    Args:
        df: DataFrame with 'drawdown' column
        title: Plot title
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    if df.empty or 'drawdown' not in df.columns:
        print("No drawdown data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.fill_between(df.index, df['drawdown'], 0, alpha=0.3, color='red')
    ax.plot(df.index, df['drawdown'], linewidth=2, color='red', label='Drawdown')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved drawdown curve to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_performance_comparison(
    df: pd.DataFrame,
    title: str = "Performance Comparison",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot equity curve with drawdown on the same figure.
    
    Args:
        df: DataFrame with 'equity' and 'drawdown' columns
        title: Plot title
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    if df.empty:
        print("No data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Equity curve
    if 'equity' in df.columns:
        ax1.plot(df.index, df['equity'], linewidth=2, label='Equity', color='blue')
        ax1.fill_between(df.index, df['equity'], alpha=0.3, color='blue')
        ax1.set_ylabel('Equity ($)', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
    
    # Drawdown
    if 'drawdown' in df.columns:
        ax2.fill_between(df.index, df['drawdown'], 0, alpha=0.3, color='red')
        ax2.plot(df.index, df['drawdown'], linewidth=2, color='red', label='Drawdown')
        ax2.set_ylabel('Drawdown', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backtests.data_loader import load_ohlcv
    from strategies.ma_crossover import generate_signals
    from backtests.backtest_runner import run_backtest
    
    print("Testing visualizations...")
    df = load_ohlcv("TSLA", "2022-01-01", "2023-01-01")
    
    if not df.empty:
        df = generate_signals(df, fast=10, slow=30)
        bt_df, metrics = run_backtest(df, starting_cash=10000)
        
        # Create results directory if it doesn't exist
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Plot and save
        plot_performance_comparison(
            bt_df,
            title=f"TSLA MA Crossover Strategy",
            save_path=str(results_dir / "backtest_results.png"),
            show=False
        )
        print("\nVisualization complete!")

