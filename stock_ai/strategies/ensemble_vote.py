"""
Ensemble Voting Strategy

Combines signals from multiple strategies (MA, RSI, Bollinger) using voting mechanism.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from loguru import logger
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from strategies import ma_crossover, rsi_filter, bollinger_reversion


def generate_signals(
    df: pd.DataFrame,
    fast: int = 10,
    slow: int = 30,
    rsi_threshold: float = 55,
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
    min_votes: int = 2,  # Minimum strategies that agree
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Generate signals using ensemble voting from multiple strategies.
    
    Strategy:
    - Runs MA crossover, RSI filter, and Bollinger reversion
    - Combines signals using voting (simple majority)
    - Requires min_votes strategies to agree before taking position
    
    Args:
        df: DataFrame with OHLCV data
        fast: Fast moving average period
        slow: Slow moving average period
        rsi_threshold: RSI threshold for RSI filter strategy
        bollinger_window: Bollinger Band window
        bollinger_std: Bollinger Band standard deviations
        min_votes: Minimum number of strategies that must agree
        price_col: Column name for price data
    
    Returns:
        DataFrame with added columns from all strategies and final 'signal'
    """
    if df.empty:
        return df
    
    data = df.copy()
    
    # Generate signals from each strategy
    logger.debug("Generating MA crossover signals...")
    data = ma_crossover.generate_signals(data, fast=fast, slow=slow, price_col=price_col)
    data['signal_ma'] = data['signal']
    
    logger.debug("Generating RSI filter signals...")
    data = rsi_filter.generate_signals(
        data, 
        fast=fast, 
        slow=slow, 
        rsi_threshold=rsi_threshold,
        price_col=price_col
    )
    data['signal_rsi'] = data['signal']
    
    logger.debug("Generating Bollinger reversion signals...")
    data = bollinger_reversion.generate_signals(
        data, 
        bollinger_window=bollinger_window,
        bollinger_std=bollinger_std,
        price_col=price_col
    )
    data['signal_bb'] = data['signal']
    
    # Combine signals using voting
    # Count buy votes (signal > 0) and sell votes (signal < 0)
    data['buy_votes'] = (
        (data['signal_ma'] > 0).astype(int) + 
        (data['signal_rsi'] > 0).astype(int) + 
        (data['signal_bb'] > 0).astype(int)
    )
    
    data['sell_votes'] = (
        (data['signal_ma'] < 0).astype(int) + 
        (data['signal_rsi'] < 0).astype(int) + 
        (data['signal_bb'] < 0).astype(int)
    )
    
    # Generate final signal based on voting
    data['signal'] = 0
    
    # Buy if enough buy votes
    data.loc[data['buy_votes'] >= min_votes, 'signal'] = 1
    
    # Sell if enough sell votes
    data.loc[data['sell_votes'] >= min_votes, 'signal'] = -1
    
    # Only generate signals at actual changes
    data['signal_change'] = data['signal'].diff()
    data.loc[data['signal_change'] == 0, 'signal'] = 0  # Hold during no change
    data['signal'] = data['signal_change']
    data = data.drop(columns=['signal_change'], errors='ignore')
    
    # Fill NaN values
    data['signal'] = data['signal'].fillna(0)
    data['buy_votes'] = data['buy_votes'].fillna(0)
    data['sell_votes'] = data['sell_votes'].fillna(0)
    
    return data


def generate_signals_weighted(
    df: pd.DataFrame,
    fast: int = 10,
    slow: int = 30,
    rsi_threshold: float = 55,
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
    weights: Dict[str, float] = None,
    min_score: float = 0.5,  # Minimum weighted score to take position
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Generate signals using weighted combination of strategies.
    
    Args:
        df: DataFrame with OHLCV data
        fast: Fast moving average period
        slow: Slow moving average period
        rsi_threshold: RSI threshold for RSI filter strategy
        bollinger_window: Bollinger Band window
        bollinger_std: Bollinger Band standard deviations
        weights: Dictionary mapping strategy name to weight
        min_score: Minimum weighted score to generate signal
        price_col: Column name for price data
    
    Returns:
        DataFrame with added columns and final 'signal'
    """
    if weights is None:
        weights = {'ma': 0.4, 'rsi': 0.35, 'bb': 0.25}
    
    if df.empty:
        return df
    
    data = df.copy()
    
    # Generate signals from each strategy
    data = ma_crossover.generate_signals(data, fast=fast, slow=slow, price_col=price_col)
    data['signal_ma'] = data['signal']
    
    data = rsi_filter.generate_signals(
        data, 
        fast=fast, 
        slow=slow, 
        rsi_threshold=rsi_threshold,
        price_col=price_col
    )
    data['signal_rsi'] = data['signal']
    
    data = bollinger_reversion.generate_signals(
        data, 
        bollinger_window=bollinger_window,
        bollinger_std=bollinger_std,
        price_col=price_col
    )
    data['signal_bb'] = data['signal']
    
    # Calculate weighted score
    # Normalize individual signals to [-1, 1] range, then weight and sum
    data['score_ma'] = data['signal_ma'] * weights.get('ma', 0.33)
    data['score_rsi'] = data['signal_rsi'] * weights.get('rsi', 0.33)
    data['score_bb'] = data['signal_bb'] * weights.get('bb', 0.33)
    
    data['weighted_score'] = data['score_ma'] + data['score_rsi'] + data['score_bb']
    
    # Generate final signal based on weighted score
    data['signal'] = 0
    data.loc[data['weighted_score'] >= min_score, 'signal'] = 1
    data.loc[data['weighted_score'] <= -min_score, 'signal'] = -1
    
    # Only generate signals at actual changes
    data['signal_change'] = data['signal'].diff()
    data.loc[data['signal_change'] == 0, 'signal'] = 0
    data['signal'] = data['signal_change']
    data = data.drop(columns=['signal_change'], errors='ignore')
    
    # Fill NaN values
    data['signal'] = data['signal'].fillna(0)
    data['weighted_score'] = data['weighted_score'].fillna(0)
    
    return data


def get_strategy_metrics(df: pd.DataFrame) -> dict:
    """Calculate ensemble-specific metrics."""
    if df.empty or 'signal' not in df.columns:
        return {}
    
    signals = df[df['signal'] != 0]
    
    metrics = {
        'total_signals': len(signals),
        'buy_signals': len(signals[signals['signal'] > 0]),
        'sell_signals': len(signals[signals['signal'] < 0]),
        'signal_rate': len(signals) / len(df) if len(df) > 0 else 0
    }
    
    # Calculate agreement statistics
    if 'buy_votes' in df.columns:
        buy_signals_with_votes = signals[signals['signal'] > 0]
        sell_signals_with_votes = signals[signals['signal'] < 0]
        
        metrics['avg_buy_votes'] = buy_signals_with_votes['buy_votes'].mean() if len(buy_signals_with_votes) > 0 else 0
        metrics['avg_sell_votes'] = sell_signals_with_votes['sell_votes'].mean() if len(sell_signals_with_votes) > 0 else 0
    
    if 'weighted_score' in df.columns:
        buy_signals_with_score = signals[signals['signal'] > 0]
        sell_signals_with_score = signals[signals['signal'] < 0]
        
        metrics['avg_buy_score'] = buy_signals_with_score['weighted_score'].mean() if len(buy_signals_with_score) > 0 else 0
        metrics['avg_sell_score'] = sell_signals_with_score['weighted_score'].mean() if len(sell_signals_with_score) > 0 else 0
    
    return metrics


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backtests.data_loader import load_ohlcv
    
    print("Testing Ensemble Voting Strategy...")
    df = load_ohlcv("AAPL", "2022-01-01", "2023-01-01")
    
    if not df.empty:
        df = generate_signals(df, fast=10, slow=30, rsi_threshold=55, min_votes=2)
        print(f"\nData with ensemble signals:\n{df[['close', 'signal_ma', 'signal_rsi', 'signal_bb', 'buy_votes', 'signal']].tail(20)}")
        
        metrics = get_strategy_metrics(df)
        print(f"\nEnsemble Strategy Metrics: {metrics}")

