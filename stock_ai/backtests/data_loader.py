"""
Data loader for stock backtesting.

Loads OHLCV data from various sources (yfinance, Alpaca, etc.)
"""

import yfinance as yf
import pandas as pd
from typing import Optional, Dict
from loguru import logger


def load_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    source: str = "yfinance"
) -> pd.DataFrame:
    """
    Load OHLCV (Open, High, Low, Close, Volume) data for a stock.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '1h', '5m', etc.)
        source: Data source ('yfinance', 'alpaca', etc.)
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    
    Example:
        >>> df = load_ohlcv("AAPL", "2022-01-01", "2023-01-01")
        >>> print(df.head())
    """
    logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
    
    if source == "yfinance":
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                logger.error(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Rename columns to match standard naming
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Stock Splits': 'stock_splits',
                'Dividends': 'dividends'
            })
            
            # Convert index to datetime if it's a Period
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            logger.info(f"Loaded {len(df)} rows of data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    else:
        logger.error(f"Unknown data source: {source}")
        return pd.DataFrame()


def load_multiple_symbols(
    symbols: list,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple symbols.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval
    
    Returns:
        Dictionary mapping symbol to DataFrame
    
    Example:
        >>> data = load_multiple_symbols(['AAPL', 'TSLA'], '2022-01-01', '2023-01-01')
    """
    results = {}
    for symbol in symbols:
        df = load_ohlcv(symbol, start_date, end_date, interval)
        if not df.empty:
            results[symbol] = df
    return results


if __name__ == "__main__":
    # Quick smoke test
    print("Running smoke test...")
    df = load_ohlcv("AAPL", "2022-01-01", "2023-01-01")
    print(f"\nData shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nLast few rows:\n{df.tail()}")

