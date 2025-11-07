"""
Live Trading Runner

Main loop for live or paper trading across supported brokers.
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
import os

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging with rotation
try:
    from utils.logging_setup import setup_live_trading_logging
    setup_live_trading_logging()
except ImportError:
    # Fallback if logging setup not available
    pass

from api import BrokerFactory, BrokerInterface, BrokerNotAvailableError
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
    Live trading runner for paper/live trading through any registered broker.
    
    Features:
    - Continuous monitoring of market conditions
    - Automatic order execution based on signals
    - Risk management and position sizing
    - Portfolio tracking and logging
    - Kill switch and safety guards
    """
    
    def __init__(self, config: Dict):
        """Initialize live trading runner."""
        self.config = config

        broker_cfg = config.get('live', {}).get('broker') or config.get('broker', 'alpaca')
        if isinstance(broker_cfg, dict):
            broker_name = broker_cfg.get('name', 'alpaca')
            broker_kwargs = {k: v for k, v in broker_cfg.items() if k != 'name'}
        else:
            broker_name = str(broker_cfg)
            broker_kwargs = {}

        secrets_path = config.get('paths', {}).get('secrets', 'config/secrets.yaml')
        broker_kwargs.setdefault('config_path', secrets_path)

        try:
            self.broker: BrokerInterface = BrokerFactory.create(broker_name, **broker_kwargs)
        except BrokerNotAvailableError as exc:
            logger.error(f"Failed to initialize broker '{broker_name}': {exc}")
            raise

        self.broker_name = broker_name
        
        # Load strategy
        strategy_config = config.get('strategy', {})
        self.strategy_name = strategy_config.get('name', 'ma_crossover')
        self.strategy_params = strategy_config.get('params', {})
        self.strategy_func = get_strategy_function(self.strategy_name)
        
        # Trading parameters
        self.check_interval = config.get('live', {}).get('check_interval', 300)
        self.max_position_size = config.get('backtest', {}).get('max_position_pct', 1.0)
        self.commission = config.get('backtest', {}).get('commission', 0.001)
        
        # Risk management parameters
        risk_config = config.get('risk', {})
        self.max_drawdown_pct = risk_config.get('max_drawdown_pct', 5.0)
        self.max_consecutive_losses = risk_config.get('max_consecutive_losses', 5)
        self.kill_switch_file = risk_config.get('kill_switch_file', 'stop.txt')
        self.initial_equity = None
        
        # Portfolio tracking
        self.equity_history = []
        self.trade_log = []
        self.last_check = None
        self.consecutive_losses = 0
        self.last_trade_was_loss = False
        self.max_equity_seen = None
        self.trading_enabled = True
        
        logger.info(f"Initialized Live Trading Runner with strategy: {self.strategy_name}")
        logger.info(f"Risk limits: max_drawdown={self.max_drawdown_pct}%, max_losses={self.max_consecutive_losses}")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        clock = self.broker.get_clock() or {}
        return clock.get('is_open', False)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol via the configured broker."""
        return self.broker.get_current_price(symbol)
    
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
            account = self.broker.get_account()
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
            
            # Generate signals using strategy - filter params to only include valid ones
            import inspect
            sig = inspect.signature(self.strategy_func)
            valid_params = {k: v for k, v in self.strategy_params.items() if k in sig.parameters}
            
            if len(valid_params) < len(self.strategy_params):
                invalid = set(self.strategy_params.keys()) - set(valid_params.keys())
                logger.debug(f"{symbol}: Filtered out invalid params: {invalid}")
            
            df = self.strategy_func(df, **valid_params)
            
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
            position = self.broker.get_position(symbol)
            has_position = position is not None
            
            # Decide on action
            should_buy = latest_signal > 0 and not has_position
            should_sell = latest_signal < 0 and has_position
            
            logger.info(f"{symbol}: signal={latest_signal:.0f}, has_position={has_position}, should_buy={should_buy}, should_sell={should_sell}")
            
            # Additional logging for debugging
            if not should_buy and not should_sell:
                if has_position and latest_signal > 0:
                    logger.debug(f"{symbol}: Signal is BUY but already have position (holding)")
                elif not has_position and latest_signal < 0:
                    logger.debug(f"{symbol}: Signal is SELL but no position (waiting)")
                elif latest_signal == 0:
                    logger.debug(f"{symbol}: Signal is HOLD (no action)")
            
            return should_buy, should_sell
            
        except Exception as e:
            logger.error(f"Error determining trade for {symbol}: {e}")
            return False, False
    
    def execute_trade(self, symbol: str, action: str, qty: int) -> Optional[str]:
        """Execute a trade order."""
        # Check risk limits before trading
        should_stop, reason = self.check_risk_limits()
        if should_stop:
            logger.warning(f"Trade execution blocked: {reason}")
            return None
        
        try:
            price = self.get_current_price(symbol)
            if not price:
                logger.error(f"Could not get price for {symbol}, aborting trade")
                return None
            
            result = None
            if action == 'buy':
                result = self.broker.submit_order(symbol, qty, side='buy', order_type='market')
            elif action == 'sell':
                # For sell, if we have position, sell all
                position = self.broker.get_position(symbol)
                if position:
                    qty = abs(position['qty'])  # Ensure positive qty for sell
                result = self.broker.submit_order(symbol, qty, side='sell', order_type='market')
            else:
                logger.warning(f"Unknown action: {action}")
                return None

            order_id = getattr(result, 'order_id', None)
            if order_id:
                logger.info(f"Executed {action} order for {qty} shares of {symbol} @ ${price:.2f} via {self.broker_name}")

                # Log trade with full details
                trade_entry = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': action,
                    'qty': qty,
                    'price': price,
                    'order_id': order_id,
                    'broker': self.broker_name,
                }
                self.trade_log.append(trade_entry)

                # Update trade statistics
                self.update_trade_stats(symbol, action, price, qty)

            return order_id
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def update_equity(self):
        """Update equity tracking."""
        try:
            account = self.broker.get_account()
            equity = account.get('portfolio_value', 0)
            
            # Initialize or update max equity
            if self.initial_equity is None:
                self.initial_equity = equity
                self.max_equity_seen = equity
            else:
                if equity > self.max_equity_seen:
                    self.max_equity_seen = equity
            
            # Calculate drawdown
            drawdown_pct = 0
            if self.max_equity_seen and equity < self.max_equity_seen:
                drawdown_pct = ((self.max_equity_seen - equity) / self.max_equity_seen) * 100
            
            self.equity_history.append({
                'timestamp': datetime.now(),
                'equity': equity,
                'cash': account.get('cash', 0),
                'positions_value': account.get('portfolio_value', 0) - account.get('cash', 0),
                'drawdown_pct': drawdown_pct,
                'max_equity': self.max_equity_seen
            })
            
            logger.debug(f"Equity: ${equity:,.2f} | Max: ${self.max_equity_seen:,.2f} | Drawdown: {drawdown_pct:.2f}%")
            
        except Exception as e:
            logger.error(f"Error updating equity: {e}")
    
    def check_kill_switch(self) -> bool:
        """Check if kill switch file exists."""
        kill_switch_path = Path(self.kill_switch_file)
        if kill_switch_path.exists():
            logger.warning(f"Kill switch file detected: {self.kill_switch_file}. Stopping trading.")
            self.trading_enabled = False
            return True
        return False
    
    def check_risk_limits(self) -> tuple:
        """
        Check if risk limits are exceeded.
        
        Returns:
            (should_stop, reason)
        """
        if not self.trading_enabled:
            return True, "Trading disabled (kill switch)"
        
        # Check kill switch file
        if self.check_kill_switch():
            return True, "Kill switch file detected"
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.error(f"Max consecutive losses ({self.max_consecutive_losses}) exceeded!")
            self.trading_enabled = False
            return True, f"Max consecutive losses ({self.max_consecutive_losses}) exceeded"
        
        # Check drawdown
        if self.max_equity_seen is not None:
            account = self.broker.get_account()
            current_equity = account.get('portfolio_value', 0)
            
            if current_equity < self.max_equity_seen:
                drawdown_pct = ((self.max_equity_seen - current_equity) / self.max_equity_seen) * 100
                
                if drawdown_pct >= self.max_drawdown_pct:
                    logger.error(f"Max drawdown ({self.max_drawdown_pct}%) exceeded! Current: {drawdown_pct:.2f}%")
                    self.trading_enabled = False
                    return True, f"Max drawdown ({self.max_drawdown_pct}%) exceeded: {drawdown_pct:.2f}%"
        
        return False, ""
    
    def update_trade_stats(self, symbol: str, action: str, price: float, qty: int):
        """Update trade statistics and check for losses."""
        # Track if last trade was a loss
        if action == 'sell' and len(self.trade_log) > 0:
            # Find entry price for this position
            entry_price = None
            for trade in reversed(self.trade_log):
                if trade['symbol'] == symbol and trade['action'] == 'buy':
                    entry_price = trade.get('price', 0)
                    break
            
            if entry_price:
                pnl_pct = ((price - entry_price) / entry_price) * 100
                if pnl_pct < 0:
                    self.consecutive_losses += 1
                    self.last_trade_was_loss = True
                    logger.warning(f"Trade loss: {pnl_pct:.2f}%. Consecutive losses: {self.consecutive_losses}")
                else:
                    self.consecutive_losses = 0
                    self.last_trade_was_loss = False
                    logger.info(f"Trade profit: {pnl_pct:.2f}%. Reset consecutive losses.")
    
    def save_logs(self):
        """Save trading logs to disk."""
        try:
            log_dir = Path("logs")
            results_dir = Path("results")
            log_dir.mkdir(exist_ok=True)
            results_dir.mkdir(exist_ok=True)
            
            # Save equity history
            if self.equity_history:
                equity_df = pd.DataFrame(self.equity_history)
                equity_path = log_dir / f"equity_history_{datetime.now().strftime('%Y%m%d')}.csv"
                equity_df.to_csv(equity_path, index=False)
                logger.info(f"Saved equity history to {equity_path}")
            
            # Save trade log to both logs and results
            if self.trade_log:
                trade_df = pd.DataFrame(self.trade_log)
                
                # Save to logs with date
                trade_path_log = log_dir / f"trade_log_{datetime.now().strftime('%Y%m%d')}.csv"
                trade_df.to_csv(trade_path_log, index=False)
                logger.info(f"Saved trade log to {trade_path_log}")
                
                # Save to results/trades_log.csv (append mode)
                trades_log_path = results_dir / "trades_log.csv"
                if trades_log_path.exists():
                    existing_df = pd.read_csv(trades_log_path)
                    combined_df = pd.concat([existing_df, trade_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['timestamp', 'symbol', 'action'], keep='last')
                    combined_df.to_csv(trades_log_path, index=False)
                else:
                    trade_df.to_csv(trades_log_path, index=False)
                logger.info(f"Updated trades log: {trades_log_path}")
            
        except Exception as e:
            logger.error(f"Error saving logs: {e}")
    
    def run_one_cycle(self, symbol: str):
        """Run one trading cycle for a symbol."""
        logger.info(f"Running trading cycle for {symbol}...")
        
        # Check risk limits first
        should_stop, reason = self.check_risk_limits()
        if should_stop:
            logger.warning(f"Trading cycle skipped: {reason}")
            return
        
        # Check if market is open
        market_open = self.is_market_open()
        
        # Allow trading outside market hours if enabled in config (for testing)
        allow_after_hours = self.config.get('live', {}).get('allow_after_hours', False)
        
        if not market_open:
            if allow_after_hours:
                logger.warning("Market is closed, but allow_after_hours is enabled - proceeding with caution")
            else:
                logger.info("Market is closed, skipping cycle")
                return
        
        # Determine if we should trade
        should_buy, should_sell = self.should_trade(symbol)
        
        # Execute trades (only if trading is enabled)
        if self.trading_enabled:
            if should_buy:
                qty = self.calculate_position_size(symbol, 1)
                if qty > 0:
                    logger.info(f"{symbol}: Executing BUY order for {qty} shares")
                    self.execute_trade(symbol, 'buy', qty)
                else:
                    logger.warning(f"{symbol}: Cannot execute BUY - calculated qty is 0")
            
            if should_sell:
                qty = self.calculate_position_size(symbol, -1)
                if qty > 0:
                    logger.info(f"{symbol}: Executing SELL order for {qty} shares")
                    self.execute_trade(symbol, 'sell', qty)
                else:
                    logger.warning(f"{symbol}: Cannot execute SELL - calculated qty is 0")
        else:
            logger.warning(f"{symbol}: Trading is disabled - skipping execution")
        
        # Update equity
        self.update_equity()
    
    def run_continuous(self, symbols: List[str]):
        """Run continuous trading loop with risk management."""
        logger.info(f"Starting continuous trading for symbols: {symbols}")
        
        # Initialize equity tracking
        self.update_equity()
        
        try:
            while self.trading_enabled:
                logger.info("=" * 60)
                logger.info("Starting trading cycle...")
                logger.info(f"Current time: {datetime.now()}")
                
                # Check risk limits before each cycle
                should_stop, reason = self.check_risk_limits()
                if should_stop:
                    logger.error(f"Trading stopped: {reason}")
                    break
                
                # Run cycle for each symbol
                for symbol in symbols:
                    try:
                        if not self.trading_enabled:
                            break
                        self.run_one_cycle(symbol)
                    except Exception as e:
                        logger.error(f"Error in trading cycle for {symbol}: {e}")
                
                # Save logs periodically (every hour or when stopped)
                if self.last_check is None or (datetime.now() - self.last_check).seconds > 3600:
                    self.save_logs()
                    self.last_check = datetime.now()
                
                # Print status summary
                account = self.broker.get_account()
                equity = account.get('portfolio_value', 0)
                drawdown_info = ""
                if self.max_equity_seen:
                    drawdown_pct = ((self.max_equity_seen - equity) / self.max_equity_seen) * 100 if equity < self.max_equity_seen else 0
                    drawdown_info = f" | Drawdown: {drawdown_pct:.2f}%"
                
                logger.info(f"Equity: ${equity:,.2f}{drawdown_info} | Consecutive Losses: {self.consecutive_losses}/{self.max_consecutive_losses}")
                
                # Wait before next cycle
                logger.info(f"Waiting {self.check_interval} seconds until next cycle...")
                
                # Check for stop file during wait (check every 10 seconds)
                for _ in range(self.check_interval // 10):
                    if self.check_kill_switch():
                        break
                    time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Trading stopped by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            # Final save
            self.save_logs()
            logger.info("Trading session ended")
            
            # Print final summary
            account = self.broker.get_account()
            print_portfolio_summary(self.broker)


def print_portfolio_summary(broker: BrokerInterface):
    """Print current portfolio summary."""
    account = broker.get_account()
    positions = broker.get_positions()
    
    print("\n" + "="*80)
    print("PORTFOLIO SUMMARY")
    print("="*80)
    print(f"Cash:             ${account.get('cash', 0):,.2f}")
    print(f"Portfolio Value:  ${account.get('portfolio_value', 0):,.2f}")
    print(f"Equity:           ${account.get('equity', 0):,.2f}")
    print(f"Buying Power:     ${account.get('buying_power', 0):,.2f}")
    print(f"\nOpen Positions: {len(positions)}")
    
    for pos in positions:
        current_price = pos.get('current_price') or broker.get_current_price(pos['symbol']) or 0
        unrealized_pl = pos.get('unrealized_pl', 0)
        unrealized_pct = pos.get('unrealized_plpc', 0)
        pct_display = f"{unrealized_pct*100:.2f}%" if isinstance(unrealized_pct, (int, float)) else unrealized_pct
        print(
            f"  {pos['symbol']}: {pos['qty']} shares @ ${current_price:.2f} "
            f"(PL: ${unrealized_pl:.2f}, {pct_display})"
        )
    
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
    print_portfolio_summary(runner.broker)
    
    # Ask for confirmation (or check for auto-confirm)
    auto_confirm = os.getenv('AUTO_CONFIRM', 'false').lower() == 'true' or '--auto-confirm' in sys.argv
    
    if not auto_confirm:
        print("\n⚠️  WARNING: This will execute live/paper trades!")
        print("Make sure you have:")
        print(f"  1. Configured config/secrets.yaml with your {runner.broker_name.title()} credentials")
        print("  2. Verified you're targeting the correct paper/live environment")
        print("  3. Tested your strategy thoroughly in backtests")
        
        response = input("\nProceed with live trading? (yes/no): ")
        
        if response.lower() not in ['yes', 'y']:
            print("Trading cancelled.")
            sys.exit(0)
    else:
        logger.info("Auto-confirmation enabled. Starting trading...")
    
    # Run continuous trading
    runner.run_continuous(tickers)

