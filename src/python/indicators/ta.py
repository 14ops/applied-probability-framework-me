import pandas as pd
import numpy as np

def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple Moving Average (SMA)"""
    return series.rolling(n).mean()

def ema(series: pd.Series, n: int) -> pd.Series:
    """Exponential Moving Average (EMA)"""
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)"""
    # Calculate price changes
    delta = series.diff()
    
    # Separate gains (up) and losses (down)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate subsequent average gain and loss (Wilder's smoothing/EMA)
    # The formula is: new_avg = (prev_avg * (n - 1) + current_value) / n
    # This is equivalent to pandas' ewm(alpha=1/n, adjust=False)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    
    # Calculate Relative Strength (RS)
    # Handle division by zero by replacing 0 with a small number or setting RS to 0/100
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # The first 'n' values will be NaN due to the initial rolling mean calculation
    # We can use the ewm result directly, but for strict RSI definition, the first n-1 values are often NaN
    # The ewm approach handles the initial period smoothing more gracefully than the strict SMA-based initial calculation
    return rsi

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range (ATR)"""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    # Calculate True Range (TR)
    tr = pd.DataFrame({
        'h-l': high - low,
        'h-pc': abs(high - close),
        'l-pc': abs(low - close)
    }).max(axis=1)
    
    # Calculate ATR (EMA of TR)
    atr = tr.ewm(span=n, adjust=False).mean()
    return atr

def bollinger(series: pd.Series, n: int = 20, k: int = 2) -> pd.DataFrame:
    """Bollinger Bands (BB)"""
    # Middle Band (SMA)
    mid = series.rolling(n).mean()
    
    # Standard Deviation
    std = series.rolling(n).std()
    
    # Upper and Lower Bands
    upper = mid + (std * k)
    lower = mid - (std * k)
    
    return pd.DataFrame({'mid': mid, 'upper': upper, 'lower': lower})
