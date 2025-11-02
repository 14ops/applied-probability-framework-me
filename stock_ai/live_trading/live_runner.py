"""
Live Trading Runner

Main loop for live paper trading on Alpaca.
Implements risk management, order execution, and portfolio monitoring.
"""

import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger
import yaml
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.alpaca_interface import AlpacaInterface
from backtests.data_loader import load_ohlcv
from backtests.backtest_runner import print_metrics
from strategies import ma_crossover, rsi_filter, bollinger_reversion, ensemble_vote


def load_configs() -> tuple:
    """Load configuration files."""
    try:
        with open("config/settings.yaml", 'r') as f:
            settings = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("config/settings.yaml not found!")
        raise
    
    return settings


def get_strategy_function(strategy_name: str):
    """Get strategy signal generation function."""
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


class LiveTradingRunner:
    """
    Live trading runner for paper/live trading on Alpaca.
    
    Features:
    - Continuous monitoring of market conditions
    - Automatic order execution based on signals
    - Risk management and position sizing
    - Portfolio tracking and logging
    """
    
    def __init__(self, config: Dict):
        """Initialize live trading runner."""
        self.config = config
        self.alpaca = AlpacaInterface()
        
        # Load strategy
        strategy_config = config.get('strategy', {})
        self.strategy_name = strategy_config.get('name', 'ma_crossover')
        self.strategy_params = strategy_config.get('params', {})
        self.strategy_func = get_strategy_function(self.strategy_name)
        
        # Trading parameters
        self.check_interval = config.get('live', {}).get('check_interval', 300)
        self.max_position_size = config.get('backtest', {}).get('max_position_pct', 1.0)
        self.commission = config.get('backtest', {}).get('commission', 0.001)
        
        # Portfolio tracking
        self.equity_history = []
        self.trade_log = []
        self.last_check = None
        
        logger.info(f"Initialized Live Trading Runner with strategy: {self.strategy_name}")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        clock = self.alpaca.get_clock()
        return clock.get('is_open', False)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            # Try to get from position first
            position = self.alpaca.get_position(symbol)
            if position:
                return position['current_price']
            
            # Otherwise get latest bar
            bars = self.alpaca.get_bars([symbol], '1Min', limit=1)
            if bars and symbol in bars and len(bars[symbol]) > 0:
                return bars[symbol][-1]['close']
            
            return None
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, signal: int) -> int:
        """
        Calculate position size based on signal and portfolio.
        
        Args:
            symbol: Stock symbol
            signal: Signal strength (1 = buy, -1 = sell, 0 = hold)
        
        Returns:
            Number of shares to trade
        """
        if signal == 0:
            return 0
        
        try:
            account = self.alpaca.get_account()
            buying_power = account.get('buying_power', 0)
            
            if buying_power <= 0:
                logger.warning("No buying power available")
                return 0
            
            # Get current price
            current_price = self.get_current_price(symbol)
            if not current_price:
                logger.warning(f"Could not get price for {symbol}")
                return 0
            
            # Calculate position size
            max_trade_value = buying_power * self.max_position_size
            shares = int(max_trade_value / current_price)
            
            logger.info(f"Position size for {symbol}: {shares} shares @ ${current_price:.2f}")
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def generate_signals(self, symbol: str) -> pd.DataFrame:
        """Generate trading signals for a symbol."""
        try:
            # Load recent data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            
            df = load_ohlcv(symbol, start_date, end_date, source='yfinance')
            
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return pd.DataFrame()
            
            # Generate signals using strategy
            df = self.strategy_func(df, **self.strategy_params)
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return pd.DataFrame()
    
    def should_trade(self, symbol: str) -> tuple:
        """
        Determine if we should trade and what action to take.
        
        Returns:
            (should_buy, should_sell)
        """
        try:
            # Generate signals
            df = self.generate_signals(symbol)
            
            if df.empty or len(df) == 0:
                return False, False
            
            # Get latest signal
            latest_signal = df['signal'].iloc[-1]
            
            # Get current position
            position = self.alpaca.get_position(symbol)
            has_position = position is not None
            
            # Decide on action
            should_buy = latest_signal > 0 and not has_position
            should_sell = latest_signal < 0 and has_position
            
            logger.info(f"{symbol}: signal={latest_signal:.0f}, has_position={has_position}, should_buy={should_buy}, should_sell={should_sell}")
            
            return should_buy, should_sell
            
        except Exception as e:
            logger.error(f"Error determining trade for {symbol}: {e}")
            return False, False
    
    def execute_trade(self, symbol: str, action: str, qty: int) -> Optional[str]:
        """Execute a trade order."""
        try:
            if action == 'buy':
                order_id = self.alpaca.submit_order(symbol, qty, side='buy', order_type='market')
            elif action == 'sell':
                # For sell, if we have position, sell all
                position = self.alpaca.get_position(symbol)
                if position:
                    qty = abs(position['qty'])  # Ensure positive qty for sell
                order_id = self.alpaca.submit_order(symbol, qty, side='sell', order_type='market')
            else:
                logger.warning(f"Unknown action: {action}")
                return None
            
            if order_id:
                logger.info(f"Executed {action} order for {qty} shares of {symbol}")
                
                # Log trade
                self.trade_log.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': action,
                    'qty': qty,
                    'order_id': order_id
                })
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def update_equity(self):
        """Update equity tracking."""
        try:
            account = self.alpaca.get_account()
            equity = account.get('portfolio_value', 0)
            
            self.equity_history.append({
                'timestamp': datetime.now(),
                'equity': equity,
                'cash': account.get('cash', 0),
                'positions_value': account.get('portfolio_value', 0) - account.get('cash', 0)
            })
            
            logger.debug(f"Equity updated: ${equity:,.2f}")
            
        except Exception as e:
            logger.error(f"Error updating equity: {e}")
    
    def save_logs(self):
        """Save trading logs to disk."""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # Save equity history
            if self.equity_history:
                equity_df = pd.DataFrame(self.equity_history)
                equity_path = log_dir / f"equity_history_{datetime.now().strftime('%Y%m%d')}.csv"
                equity_df.to_csv(equity_path, index=False)
                logger.info(f"Saved equity history to {equity_path}")
            
            # Save trade log
            if self.trade_log:
                trade_df = pd.DataFrame(self.trade_log)
                trade_path = log_dir / f"trade_log_{datetime.now().strftime('%Y%m%d')}.csv"
                trade_df.to_csv(trade_path, index=False)
                logger.info(f"Saved trade log to {trade_path}")
            
        except Exception as e:
            logger.error(f"Error saving logs: {e}")
    
    def run_one_cycle(self, symbol: str):
        """Run one trading cycle for a symbol."""
        logger.info(f"Running trading cycle for {symbol}...")
        
        # Check if market is open
        if not self.is_market_open():
            logger.info("Market is closed, skipping cycle")
            return
        
        # Determine if we should trade
        should_buy, should_sell = self.should_trade(symbol)
        
        # Execute trades
        if should_buy:
            qty = self.calculate_position_size(symbol, 1)
            if qty > 0:
                self.execute_trade(symbol, 'buy', qty)
        
        if should_sell:
            qty = self.calculate_position_size(symbol, -1)
            if qty > 0:
                self.execute_trade(symbol, 'sell', qty)
        
        # Update equity
        self.update_equity()
    
    def run_continuous(self, symbols: List[str]):
        """Run continuous trading loop."""
        logger.info(f"Starting continuous trading for symbols: {symbols}")
        
        try:
            while True:
                logger.info("=" * 60)
                logger.info("Starting trading cycle...")
                logger.info(f"Current time: {datetime.now()}")
                
                # Run cycle for each symbol
                for symbol in symbols:
                    try:
                        self.run_one_cycle(symbol)
                    except Exception as e:
                        logger.error(f"Error in trading cycle for {symbol}: {e}")
                
                # Save logs periodically
                if self.last_check is None or (datetime.now() - self.last_check).seconds > 3600:
                    self.save_logs()
                    self.last_check = datetime.now()
                
                # Wait before next cycle
                logger.info(f"Waiting {self.check_interval} seconds until next cycle...")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            # Final save
            self.save_logs()
            logger.info("Trading session ended")


def print_portfolio_summary(alpaca: AlpacaInterface):
    """Print current portfolio summary."""
    account = alpaca.get_account()
    positions = alpaca.get_positions()
    
    print("\n" + "="*80)
    print("PORTFOLIO SUMMARY")
    print("="*80)
    print(f"Cash:             ${account.get('cash', 0):,.2f}")
    print(f"Portfolio Value:  ${account.get('portfolio_value', 0):,.2f}")
    print(f"Equity:           ${account.get('equity', 0):,.2f}")
    print(f"Buying Power:     ${account.get('buying_power', 0):,.2f}")
    print(f"\nOpen Positions: {len(positions)}")
    
    for pos in positions:
        print(f"  {pos['symbol']}: {pos['qty']} shares @ ${pos['current_price']:.2f} "
              f"(PL: ${pos['unrealized_pl']:.2f}, {pos['unrealized_plpc']*100:.2f}%)")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    logger.info("Starting Live Trading Runner...")
    
    # Load configuration
    config = load_configs()
    
    # Get trading symbols
    tickers = config.get('tickers', ['AAPL'])
    
    # Create runner
    runner = LiveTradingRunner(config)
    
    # Print initial portfolio
    print_portfolio_summary(runner.alpaca)
    
    # Ask for confirmation
    print("\n⚠️  WARNING: This will execute live/paper trades!")
    print("Make sure you have:")
    print("  1. Configured config/secrets.yaml with your Alpaca credentials")
    print("  2. Verified you're using paper trading URL")
    print("  3. Tested your strategy thoroughly in backtests")
    
    response = input("\nProceed with live trading? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        # Run continuous trading
        runner.run_continuous(tickers)
    else:
        print("Trading cancelled.")

