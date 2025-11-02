"""
Moving Average Crossover Strategy

Simple momentum strategy that generates buy/sell signals based on
the crossover of fast and slow moving averages.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def generate_signals(
    df: pd.DataFrame,
    fast: int = 10,
    slow: int = 30,
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Generate buy/sell signals using MA crossover.
    
    When fast MA crosses above slow MA → Buy signal (1)
    When fast MA crosses below slow MA → Sell signal (-1)
    Otherwise → Hold (0)
    
    Args:
        df: DataFrame with OHLCV data
        fast: Fast moving average period
        slow: Slow moving average period
        price_col: Column name for price data
    
    Returns:
        DataFrame with added columns: ma_fast, ma_slow, signal
    
    Example:
        >>> df = load_ohlcv("AAPL", "2022-01-01", "2023-01-01")
        >>> df = generate_signals(df, fast=10, slow=30)
        >>> print(df[['close', 'ma_fast', 'ma_slow', 'signal']].tail())
    """
    if df.empty:
        return df
    
    data = df.copy()
    
    # Calculate moving averages
    data['ma_fast'] = data[price_col].rolling(window=fast).mean()
    data['ma_slow'] = data[price_col].rolling(window=slow).mean()
    
    # Generate signals based on crossover
    # 1 = Buy, -1 = Sell, 0 = Hold
    data['signal'] = 0
    data.loc[data['ma_fast'] > data['ma_slow'], 'signal'] = 1  # Buy
    data.loc[data['ma_fast'] <= data['ma_slow'], 'signal'] = -1  # Sell
    
    # Only generate signals at actual crossovers
    # Current signal different from previous = crossover occurred
    data['signal_change'] = data['signal'].diff()
    data.loc[data['signal_change'] == 0, 'signal'] = 0  # Hold during no change
    data['signal_change'] = data['signal']  # Set final signal
    data = data.drop(columns=['signal_change'], errors='ignore')
    
    # First 'slow' rows will have NaN for ma_slow, set signal to 0
    data['signal'] = data['signal'].fillna(0)
    
    return data


def generate_signals_rsi(
    df: pd.DataFrame,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Generate buy/sell signals using RSI indicators.
    
    RSI < oversold → Buy signal (1)
    RSI > overbought → Sell signal (-1)
    Otherwise → Hold (0)
    
    Args:
        df: DataFrame with OHLCV data
        rsi_period: RSI calculation period
        rsi_oversold: Oversold threshold
        rsi_overbought: Overbought threshold
        price_col: Column name for price data
    
    Returns:
        DataFrame with added columns: rsi, signal
    """
    if df.empty:
        return df
    
    data = df.copy()
    
    # Calculate RSI manually
    delta = data[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Generate signals
    data['signal'] = 0
    data.loc[data['rsi'] < rsi_oversold, 'signal'] = 1  # Buy (oversold)
    data.loc[data['rsi'] > rsi_overbought, 'signal'] = -1  # Sell (overbought)
    
    return data


def get_strategy_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate basic strategy metrics from signals.
    
    Args:
        df: DataFrame with signals generated
    
    Returns:
        Dictionary with metrics: num_signals, buy_signals, sell_signals, etc.
    """
    if df.empty or 'signal' not in df.columns:
        return {}
    
    signals = df[df['signal'] != 0]
    
    metrics = {
        'total_signals': len(signals),
        'buy_signals': len(signals[signals['signal'] > 0]),
        'sell_signals': len(signals[signals['signal'] < 0]),
        'signal_rate': len(signals) / len(df) if len(df) > 0 else 0
    }
    
    return metrics


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backtests.data_loader import load_ohlcv
    
    print("Testing MA Crossover Strategy...")
    df = load_ohlcv("AAPL", "2022-01-01", "2023-01-01")
    
    if not df.empty:
        df = generate_signals(df, fast=10, slow=30)
        print(f"\nData with signals:\n{df[['close', 'ma_fast', 'ma_slow', 'signal']].tail(20)}")
        
        metrics = get_strategy_metrics(df)
        print(f"\nStrategy Metrics: {metrics}")

