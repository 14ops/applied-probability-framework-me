"""
Bollinger Bands Mean-Reversion Strategy

Trades based on price reversion to moving average when price reaches
Bollinger Band extremes (overbought/oversold conditions).
"""

import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger


def calculate_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    std_dev: float = 2.0,
    price_col: str = "close"
) -> tuple:
    """
    Calculate Bollinger Bands.
    
    Args:
        df: DataFrame with price data
        window: Moving average window
        std_dev: Number of standard deviations for bands
        price_col: Column name for price data
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if df.empty:
        return pd.Series(), pd.Series(), pd.Series()
    
    middle_band = df[price_col].rolling(window=window).mean()
    std = df[price_col].rolling(window=window).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band


def generate_signals(
    df: pd.DataFrame,
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Generate signals using Bollinger Band mean reversion.
    
    Strategy:
    - Buy when price touches lower band (oversold)
    - Sell when price touches upper band (overbought)
    - Can also use middle band as exit or trailing stop
    
    Args:
        df: DataFrame with OHLCV data
        bollinger_window: Moving average window for Bollinger Bands
        bollinger_std: Number of standard deviations for bands
        price_col: Column name for price data
    
    Returns:
        DataFrame with added columns: bb_upper, bb_middle, bb_lower, bb_width, signal
    """
    if df.empty:
        return df
    
    data = df.copy()
    
    # Calculate Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(
        data, 
        window=bollinger_window, 
        std_dev=bollinger_std,
        price_col=price_col
    )
    
    data['bb_upper'] = upper
    data['bb_middle'] = middle
    data['bb_lower'] = lower
    data['bb_width'] = (upper - lower) / middle  # Band width as % of price
    
    # Generate signals based on price touching bands
    data['signal'] = 0
    
    # Buy when price touches or crosses below lower band (oversold)
    data.loc[data[price_col] <= data['bb_lower'], 'signal'] = 1
    
    # Sell when price touches or crosses above upper band (overbought)
    data.loc[data[price_col] >= data['bb_upper'], 'signal'] = -1
    
    # Advanced: Exit position when price returns to middle band
    # This can be added as an optional exit signal
    
    # Only generate signals at actual changes
    data['signal_change'] = data['signal'].diff()
    data.loc[data['signal_change'] == 0, 'signal'] = 0  # Hold during no change
    data['signal'] = data['signal_change']
    data = data.drop(columns=['signal_change'], errors='ignore')
    
    # Fill NaN values
    data['signal'] = data['signal'].fillna(0)
    data['bb_width'] = data['bb_width'].fillna(0)
    
    return data


def generate_signals_with_rsi_filter(
    df: pd.DataFrame,
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Generate Bollinger Band signals with RSI confirmation.
    
    Reduces false signals by requiring RSI confirmation.
    """
    if df.empty:
        return df
    
    data = df.copy()
    
    # Calculate Bollinger Bands and RSI
    upper, middle, lower = calculate_bollinger_bands(
        data, 
        window=bollinger_window, 
        std_dev=bollinger_std,
        price_col=price_col
    )
    
    data['bb_upper'] = upper
    data['bb_middle'] = middle
    data['bb_lower'] = lower
    
    # Calculate RSI
    delta = data[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Generate signals with RSI confirmation
    data['signal'] = 0
    
    # Buy: Price at lower band AND RSI oversold
    data.loc[(data[price_col] <= data['bb_lower']) & (data['rsi'] < rsi_oversold), 'signal'] = 1
    
    # Sell: Price at upper band AND RSI overbought
    data.loc[(data[price_col] >= data['bb_upper']) & (data['rsi'] > rsi_overbought), 'signal'] = -1
    
    # Only generate signals at actual changes
    data['signal_change'] = data['signal'].diff()
    data.loc[data['signal_change'] == 0, 'signal'] = 0
    data['signal'] = data['signal_change']
    data = data.drop(columns=['signal_change'], errors='ignore')
    
    # Fill NaN values
    data['signal'] = data['signal'].fillna(0)
    
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
    
    if 'bb_width' in df.columns:
        metrics['avg_bb_width'] = df['bb_width'].mean()
    
    return metrics


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backtests.data_loader import load_ohlcv
    
    print("Testing Bollinger Bands Mean-Reversion Strategy...")
    df = load_ohlcv("AAPL", "2022-01-01", "2023-01-01")
    
    if not df.empty:
        df = generate_signals(df, bollinger_window=20, bollinger_std=2.0)
        print(f"\nData with signals:\n{df[['close', 'bb_upper', 'bb_middle', 'bb_lower', 'signal']].tail(20)}")
        
        metrics = get_strategy_metrics(df)
        print(f"\nStrategy Metrics: {metrics}")

