"""
Backtest Runner

Executes trading strategies on historical data and calculates performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from loguru import logger


def run_backtest(
    df: pd.DataFrame,
    starting_cash: float = 10000.0,
    commission: float = 0.001,  # 0.1% per trade
    max_position_pct: float = 1.0,  # Max 100% of cash
    slippage: float = 0.0005  # 0.05% slippage
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run a backtest on a DataFrame with trading signals.
    
    Args:
        df: DataFrame with OHLCV data and 'signal' column
        starting_cash: Initial capital
        commission: Commission rate per trade (0.001 = 0.1%)
        max_position_pct: Maximum position size as % of cash
        slippage: Slippage factor
    
    Returns:
        Tuple of (backtest_df, metrics_dict)
    
    Example:
        >>> from backtests.data_loader import load_ohlcv
        >>> from strategies.ma_crossover import generate_signals
        >>> 
        >>> df = load_ohlcv("AAPL", "2022-01-01", "2023-01-01")
        >>> df = generate_signals(df)
        >>> bt_df, metrics = run_backtest(df, starting_cash=10000)
        >>> print(metrics)
    """
    logger.info(f"Running backtest with starting cash: ${starting_cash:,.2f}")
    
    if df.empty or 'signal' not in df.columns:
        logger.error("DataFrame is empty or missing 'signal' column")
        return df, {}
    
    data = df.copy()
    
    # Initialize tracking variables
    cash = starting_cash
    position = 0  # Number of shares
    equity = []  # Track equity over time
    trades = []  # Track all trades
    
    # Initial equity
    equity.append(cash)
    
    for i in range(len(data)):
        current_price = data.iloc[i]['close']
        signal = data.iloc[i]['signal']
        
        # Execute trades based on signal
        if signal > 0 and position == 0:  # Buy signal
            # Calculate position size
            trade_value = cash * max_position_pct
            shares_to_buy = int(trade_value / current_price)
            cost = shares_to_buy * current_price * (1 + commission + slippage)
            
            if cost <= cash and shares_to_buy > 0:
                position = shares_to_buy
                cash -= cost
                
                trades.append({
                    'date': data.index[i],
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'cost': cost,
                    'cash_after': cash
                })
                logger.debug(f"BUY {shares_to_buy} shares @ ${current_price:.2f}")
        
        elif signal < 0 and position > 0:  # Sell signal
            # Sell all position
            proceeds = position * current_price * (1 - commission - slippage)
            cash += proceeds
            
            trades.append({
                'date': data.index[i],
                'action': 'SELL',
                'price': current_price,
                'shares': position,
                'proceeds': proceeds,
                'cash_after': cash
            })
            logger.debug(f"SELL {position} shares @ ${current_price:.2f}")
            position = 0
        
        # Calculate current equity (cash + position value)
        current_equity = cash + (position * current_price)
        equity.append(current_equity)
    
    # Add equity column to data
    data['equity'] = equity[1:]  # Skip first duplicate
    
    # Calculate returns
    data['returns'] = data['equity'].pct_change()
    data['cumulative_returns'] = (1 + data['returns']).cumprod() - 1
    
    # Calculate drawdown
    data['running_max'] = data['equity'].expanding().max()
    data['drawdown'] = (data['equity'] - data['running_max']) / data['running_max']
    
    # Calculate metrics
    metrics = calculate_metrics(data, starting_cash, trades)
    
    logger.info(f"Backtest complete. Final equity: ${equity[-1]:,.2f}")
    
    return data, metrics


def calculate_metrics(
    df: pd.DataFrame,
    starting_cash: float,
    trades: list
) -> Dict:
    """
    Calculate performance metrics from backtest results.
    
    Args:
        df: Backtest DataFrame with equity, returns, drawdown
        starting_cash: Initial capital
        trades: List of trade dictionaries
    
    Returns:
        Dictionary of performance metrics
    """
    if df.empty or 'equity' not in df.columns:
        return {}
    
    final_equity = df['equity'].iloc[-1]
    total_return = (final_equity - starting_cash) / starting_cash
    
    # Calculate annualized metrics (assuming daily data)
    days = len(df)
    years = days / 252  # Trading days per year
    
    # Annualized return (CAGR)
    cagr = (final_equity / starting_cash) ** (1 / years) - 1 if years > 0 else 0
    
    # Sharpe ratio (assuming risk-free rate of 0)
    returns = df['returns'].dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
    
    # Maximum drawdown
    max_dd = df['drawdown'].min()
    
    # Win rate
    winning_trades = 0
    losing_trades = 0
    if len(trades) >= 2:
        for i in range(1, len(trades), 2):  # Each BUY followed by SELL
            if i < len(trades):
                buy_price = trades[i-1]['price']
                sell_price = trades[i]['price']
                if sell_price > buy_price:
                    winning_trades += 1
                else:
                    losing_trades += 1
    
    win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
    
    metrics = {
        'starting_cash': starting_cash,
        'final_equity': final_equity,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'cagr': cagr,
        'cagr_pct': cagr * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd * 100,
        'num_trades': len([t for t in trades if t['action'] == 'BUY']),
        'win_rate': win_rate,
        'win_rate_pct': win_rate * 100,
        'total_days': days,
        'years': years
    }
    
    return metrics


def print_metrics(metrics: Dict):
    """
    Pretty print backtest metrics.
    
    Args:
        metrics: Dictionary of performance metrics
    """
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Starting Capital:  ${metrics.get('starting_cash', 0):,.2f}")
    print(f"Final Equity:      ${metrics.get('final_equity', 0):,.2f}")
    print(f"Total Return:      {metrics.get('total_return_pct', 0):.2f}%")
    print(f"CAGR:              {metrics.get('cagr_pct', 0):.2f}%")
    print(f"Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown:      {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"\nTrades:")
    print(f"  Total Trades:    {metrics.get('num_trades', 0)}")
    print(f"  Win Rate:        {metrics.get('win_rate_pct', 0):.2f}%")
    print(f"\nPeriod:            {metrics.get('years', 0):.2f} years")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backtests.data_loader import load_ohlcv
    from strategies.ma_crossover import generate_signals
    
    print("Running backtest...")
    df = load_ohlcv("AAPL", "2022-01-01", "2023-01-01")
    
    if not df.empty:
        df = generate_signals(df, fast=10, slow=30)
        bt_df, metrics = run_backtest(df, starting_cash=10000)
        
        print_metrics(metrics)
        print(f"\nFirst 10 trades and equity:")
        print(bt_df[['close', 'signal', 'equity', 'drawdown']].head(10))

