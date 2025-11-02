"""
RSI Filter Strategy

Adds RSI momentum filter to trading signals.
Only enters long positions when RSI is above threshold, indicating bullish momentum.
"""

import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger


def calculate_rsi(df: pd.DataFrame, period: int = 14, price_col: str = "close") -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        df: DataFrame with price data
        period: RSI calculation period
        price_col: Column name for price data
    
    Returns:
        Series with RSI values
    """
    if df.empty:
        return pd.Series()
    
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def generate_signals(
    df: pd.DataFrame,
    fast: int = 10,
    slow: int = 30,
    rsi_threshold: float = 55,
    rsi_period: int = 14,
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Generate signals using MA crossover with RSI filter.
    
    Strategy:
    - Uses MA crossover as base signal
    - Only enters long when RSI > threshold (bullish momentum)
    - Exits long when RSI crosses below threshold
    
    Args:
        df: DataFrame with OHLCV data
        fast: Fast moving average period
        slow: Slow moving average period
        rsi_threshold: RSI level required for long entry
        rsi_period: RSI calculation period
        price_col: Column name for price data
    
    Returns:
        DataFrame with added columns: ma_fast, ma_slow, rsi, signal
    """
    if df.empty:
        return df
    
    data = df.copy()
    
    # Calculate MA crossover base signal
    data['ma_fast'] = data[price_col].rolling(window=fast).mean()
    data['ma_slow'] = data[price_col].rolling(window=slow).mean()
    
    # Calculate RSI
    data['rsi'] = calculate_rsi(data, period=rsi_period, price_col=price_col)
    
    # Generate base MA crossover signal
    data['ma_signal'] = 0
    data.loc[data['ma_fast'] > data['ma_slow'], 'ma_signal'] = 1  # Long
    data.loc[data['ma_fast'] <= data['ma_slow'], 'ma_signal'] = -1  # Short
    
    # Apply RSI filter
    # Only buy when RSI > threshold (strong momentum)
    # Sell when RSI < threshold or MA crossover
    data['signal'] = 0
    
    # Buy signal: MA crossover long AND RSI > threshold
    data.loc[(data['ma_signal'] > 0) & (data['rsi'] > rsi_threshold), 'signal'] = 1
    
    # Sell signal: MA crossover short OR RSI drops below threshold
    data.loc[(data['ma_signal'] < 0), 'signal'] = -1
    data.loc[(data['ma_signal'] > 0) & (data['rsi'] < rsi_threshold), 'signal'] = -1
    
    # Only generate signals at actual changes
    data['signal_change'] = data['signal'].diff()
    data.loc[data['signal_change'] == 0, 'signal'] = 0  # Hold during no change
    data['signal'] = data['signal_change']
    data = data.drop(columns=['signal_change', 'ma_signal'], errors='ignore')
    
    # Fill NaN values
    data['signal'] = data['signal'].fillna(0)
    data['rsi'] = data['rsi'].fillna(50.0)
    
    return data


def get_strategy_metrics(df: pd.DataFrame) -> dict:
    """Calculate strategy-specific metrics."""
    if df.empty or 'signal' not in df.columns:
        return {}
    
    signals = df[df['signal'] != 0]
    
    metrics = {
        'total_signals': len(signals),
        'buy_signals': len(signals[signals['signal'] > 0]),
        'sell_signals': len(signals[signals['signal'] < 0]),
        'signal_rate': len(signals) / len(df) if len(df) > 0 else 0
    }
    
    if 'rsi' in df.columns:
        rsi_values = df['rsi'].dropna()
        metrics['avg_rsi'] = rsi_values.mean()
        metrics['rsi_at_entries'] = signals[signals['signal'] > 0]['rsi'].mean() if len(signals[signals['signal'] > 0]) > 0 else 0
    
    return metrics


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backtests.data_loader import load_ohlcv
    
    print("Testing RSI Filter Strategy...")
    df = load_ohlcv("AAPL", "2022-01-01", "2023-01-01")
    
    if not df.empty:
        df = generate_signals(df, fast=10, slow=30, rsi_threshold=55)
        print(f"\nData with signals:\n{df[['close', 'ma_fast', 'ma_slow', 'rsi', 'signal']].tail(20)}")
        
        metrics = get_strategy_metrics(df)
        print(f"\nStrategy Metrics: {metrics}")

