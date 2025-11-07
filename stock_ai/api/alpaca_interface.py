"""
Alpaca API Interface

Handles all interactions with Alpaca trading API for paper and live trading.
"""

import time
from typing import Dict, List, Optional
from loguru import logger
import yaml

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    # Silently set to None - warning will be shown only when actually trying to use it
    tradeapi = None

from .base_broker import BrokerInterface, OrderResult


class AlpacaInterface(BrokerInterface):
    """
    Interface to Alpaca trading API.
    
    Handles:
    - Market data retrieval
    - Order placement and management
    - Position and portfolio queries
    - Account information
    """
    
    name = "alpaca"

    def __init__(self, config_path: str = "config/secrets.yaml"):
        """Initialize Alpaca API client."""
        if tradeapi is None:
            raise ImportError("alpaca_trade_api not installed. Install with: pip install alpaca-trade-api")
        
        # Load credentials from secrets file
        try:
            with open(config_path, 'r') as f:
                secrets = yaml.safe_load(f)
            
            alpaca_config = secrets.get('alpaca', {})
            self.key_id = alpaca_config.get('key_id', '').strip()
            self.secret_key = alpaca_config.get('secret_key', '').strip()
            self.base_url = alpaca_config.get('base_url', 'https://paper-api.alpaca.markets').strip()
            
            if not self.key_id or not self.secret_key:
                raise ValueError("Alpaca credentials not found in config file")
            
            # Validate key formats
            if len(self.key_id) < 20 or len(self.secret_key) < 40:
                logger.warning(f"API key lengths seem unusual - Key ID: {len(self.key_id)}, Secret: {len(self.secret_key)}")
            
        except FileNotFoundError:
            logger.error(f"Secrets file not found: {config_path}")
            raise
        
        # Initialize API client
        self.api = tradeapi.REST(
            self.key_id,
            self.secret_key,
            self.base_url,
            api_version='v2'
        )
        
        # Test connection
        try:
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca ({'Paper' if 'paper' in self.base_url else 'Live'} Trading)")
            logger.info(f"Account status: Portfolio Value=${float(account.portfolio_value):,.2f}")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to connect to Alpaca: {error_msg}")
            if "unauthorized" in error_msg.lower():
                logger.error("Authentication failed. Please verify:")
                logger.error("  1. API keys are correct in config/secrets.yaml")
                logger.error("  2. Keys are for paper trading (not live trading)")
                logger.error("  3. Keys haven't expired (check Alpaca dashboard)")
                logger.error(f"  4. Base URL is correct: {self.base_url}")
            raise
    
    def get_account(self) -> Dict:
        """Get account information."""
        try:
            account = self.api.get_account()
            return {
                'account_number': account.account_number,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'equity': float(account.equity),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        try:
            positions = self.api.list_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'qty': int(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'side': pos.side
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a specific symbol."""
        try:
            position = self.api.get_position(symbol)
            return {
                'symbol': position.symbol,
                'qty': int(position.qty),
                'avg_entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'market_value': float(position.market_value),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'side': position.side
            }
        except Exception as e:
            # Position doesn't exist
            return None
    
    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str = 'buy',
        order_type: str = 'market',
        time_in_force: str = 'day',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        extended_hours: bool = False,
    ) -> OrderResult:
        """
        Submit an order.
        
        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            time_in_force: 'day', 'gtc', 'opg', 'cls', 'ioc', 'fok'
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
        
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            if order_type == 'market':
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force,
                    extended_hours=extended_hours
                )
            elif order_type == 'limit':
                if limit_price is None:
                    raise ValueError("limit_price required for limit orders")
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force,
                    limit_price=limit_price,
                    extended_hours=extended_hours
                )
            elif order_type == 'stop':
                if stop_price is None:
                    raise ValueError("stop_price required for stop orders")
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force,
                    stop_price=stop_price,
                    extended_hours=extended_hours
                )
            else:
                raise ValueError(f"Unsupported order_type: {order_type}")
            
            logger.info(f"Submitted {side} order for {qty} shares of {symbol}")
            filled_avg_price = getattr(order, "filled_avg_price", None)
            filled_qty = getattr(order, "filled_qty", None)
            return OrderResult(
                broker=self.name,
                order_id=getattr(order, "id", None),
                status=getattr(order, "status", "submitted"),
                submitted_at=str(getattr(order, "submitted_at", "")) if order else None,
                filled_qty=float(filled_qty) if filled_qty is not None else None,
                avg_fill_price=float(filled_avg_price) if filled_avg_price is not None else None,
                raw=getattr(order, "_raw", None),
            )

        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return OrderResult(
                broker=self.name,
                order_id=None,
                status="error",
                raw={"error": str(e)},
            )

    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get a single order by ID."""
        try:
            order = self.api.get_order(order_id)
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': int(order.qty),
                'filled_qty': int(order.filled_qty),
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
            }
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            return None

    def wait_for_fill(self, order_id: str, timeout_seconds: int = 90, poll_interval: float = 2.0) -> Optional[Dict]:
        """Poll order status until it is filled or timeout."""
        import time as _time
        deadline = _time.time() + timeout_seconds
        while _time.time() < deadline:
            info = self.get_order(order_id)
            if not info:
                return None
            if info.get('status') in ('filled', 'partially_filled'):
                return info
            if info.get('status') in ('canceled', 'rejected', 'expired'): 
                return info
            _time.sleep(poll_interval)
        return self.get_order(order_id)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_orders(self, status: str = 'all', limit: int = 100) -> List[Dict]:
        """Get orders by status."""
        try:
            orders = self.api.list_orders(status=status, limit=limit)
            return [
                {
                    'id': order.id,
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'qty': int(order.qty),
                    'filled_qty': int(order.filled_qty),
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                    'side': order.side,
                    'type': order.type,
                    'status': order.status,
                    'time_in_force': order.time_in_force,
                    'submitted_at': order.submitted_at,
                    'created_at': order.submitted_at,  # Alias for compatibility
                    'filled_at': order.filled_at
                }
                for order in orders
            ]
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []

    def list_orders(self, status: str = 'all', limit: int = 100):
        """Alias that conforms to :class:`BrokerInterface`."""
        return self.get_orders(status=status, limit=limit)
    
    def get_bars(
        self,
        symbols: List[str],
        timeframe: str = '1Day',
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100
    ) -> Dict:
        """Get historical bars for symbols."""
        try:
            bars = self.api.get_bars(
                symbols,
                timeframe,
                start=start,
                end=end,
                limit=limit
            )
            
            return {
                symbol: [
                    {
                        'timestamp': bar.t,
                        'open': float(bar.o),
                        'high': float(bar.h),
                        'low': float(bar.l),
                        'close': float(bar.c),
                        'volume': int(bar.v)
                    }
                    for bar in symbol_bars
                ]
                for symbol, symbol_bars in bars.items()
            }
        except Exception as e:
            logger.error(f"Error getting bars: {e}")
            return {}
    
    def close_all_positions(self) -> bool:
        """Close all open positions."""
        try:
            self.api.close_all_positions()
            logger.info("Closed all positions")
            return True
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return False
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get current quote (bid/ask) for a symbol.
        
        Returns:
            Dict with bid_price, ask_price, bid_size, ask_size, timestamp
        """
        try:
            quote = self.api.get_latest_quote(symbol)
            if quote:
                # Some SDK versions expose sizes with different attribute names; keep optional and robust
                bp = getattr(quote, 'bp', None)
                ap = getattr(quote, 'ap', None)
                bs = getattr(quote, 'bs', None)
                # 'as' is reserved, some versions use 'as_', others may omit
                ask_size_attr = 'as_'
                asz = getattr(quote, ask_size_attr, None)
                return {
                    'symbol': symbol,
                    'bid_price': float(bp) if bp is not None else None,
                    'ask_price': float(ap) if ap is not None else None,
                    'bid_size': int(bs) if bs is not None else None,
                    'ask_size': int(asz) if asz is not None else None,
                    'timestamp': getattr(quote, 't', None)
                }
            return None
        except Exception as e:
            logger.debug(f"Could not get quote for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Tries multiple methods:
        1. From position if we have one
        2. Latest trade from API
        3. Latest bar (1-minute)
        """
        try:
            # Try to get from position first (most accurate)
            position = self.get_position(symbol)
            if position:
                return position['current_price']
            
            # Try to get latest trade
            try:
                latest_trade = self.api.get_latest_trade(symbol)
                if latest_trade and latest_trade.price:
                    return float(latest_trade.price)
            except:
                pass
            
            # Otherwise get latest bar
            bars = self.get_bars([symbol], '1Min', limit=1)
            if bars and symbol in bars and len(bars[symbol]) > 0:
                return bars[symbol][-1]['close']
            
            return None
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_clock(self) -> Dict:
        """Get market clock information."""
        try:
            clock = self.api.get_clock()
            return {
                'timestamp': clock.timestamp,
                'is_open': clock.is_open,
                'next_open': clock.next_open,
                'next_close': clock.next_close
            }
        except Exception as e:
            logger.error(f"Error getting clock: {e}")
            return {}


if __name__ == "__main__":
    # Quick test
    print("Testing Alpaca Interface...")
    try:
        interface = AlpacaInterface()
        account = interface.get_account()
        print(f"\nAccount Info: {account}")
        
        positions = interface.get_positions()
        print(f"\nOpen Positions: {len(positions)}")
        for pos in positions:
            print(f"  {pos['symbol']}: {pos['qty']} shares @ ${pos['current_price']:.2f}")
    except Exception as e:
        print(f"Error: {e}")

