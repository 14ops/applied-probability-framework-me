"""
Live Stock Trading Bot with Paper Trading and Real Execution
Integrates with Alpaca API for live trading
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Trading APIs
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: alpaca-trade-api not installed. Install with: pip install alpaca-trade-api")

import yfinance as yf

# ML
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import from our framework
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveTradingBot:
    """
    Live stock trading bot with ML prediction and risk management.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize live trading bot."""
        self.config = config or {}
        
        # Trading settings
        self.symbols = self.config.get('symbols', ['SPY', 'QQQ', 'AAPL'])
        self.initial_capital = self.config.get('initial_capital', 10000.0)
        self.position_size_pct = self.config.get('position_size_pct', 0.1)
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = self.config.get('take_profit_pct', 0.04)
        
        # Alpaca configuration
        self.use_alpaca = self.config.get('use_alpaca', False)
        self.paper_trading = self.config.get('paper_trading', True)
        
        if self.use_alpaca and ALPACA_AVAILABLE:
            self.api = self._init_alpaca()
        else:
            self.api = None
            logger.info("Running in simulation mode (no broker API)")
        
        # Model
        self.model = None
        self.feature_names = []
        self.is_trained = False
        
        # Portfolio tracking
        self.cash = self.initial_capital
        self.positions = {}  # {symbol: shares}
        self.trade_history = []
        
        # Logging
        self.log_dir = Path('logs/trading')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def _init_alpaca(self):
        """Initialize Alpaca API."""
        try:
            # Load credentials from environment or config
            api_key = self.config.get('alpaca_api_key', None)
            api_secret = self.config.get('alpaca_secret_key', None)
            
            if not api_key or not api_secret:
                logger.warning("Alpaca credentials not provided, running in simulation mode")
                return None
            
            base_url = 'https://paper-api.alpaca.markets' if self.paper_trading else 'https://api.alpaca.markets'
            
            api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
            
            # Test connection
            account = api.get_account()
            logger.info(f"Connected to Alpaca: {account.status}")
            
            return api
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca: {e}")
            return None
    
    def train_model(self, symbol: str, lookback_days: int = 365):
        """Train ML model for a symbol."""
        logger.info(f"Training model for {symbol}...")
        
        # Load data
        data = self._load_data(symbol, lookback_days)
        if data is None or len(data) < 100:
            logger.error(f"Insufficient data for {symbol}")
            return False
        
        # Create features
        X, y = self._create_features(data)
        
        if not SKLEARN_AVAILABLE:
            logger.error("sklearn not available")
            return False
        
        # Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        accuracy = self.model.score(X_test, y_test)
        self.feature_names = list(X.columns)
        self.is_trained = True
        
        logger.info(f"Model trained for {symbol}: {accuracy:.2%} accuracy")
        return True
    
    def _load_data(self, symbol: str, lookback_days: int):
        """Load historical data for a symbol."""
        end_date = datetime.now()
        start_date = datetime.fromordinal(end_date.toordinal() - lookback_days)
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            return data
        except Exception as e:
            logger.error(f"Data load error for {symbol}: {e}")
            return None
    
    def _create_features(self, data: pd.DataFrame) -> tuple:
        """Create feature matrix and target labels."""
        df = data.copy()
        
        # Target
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        
        # Features
        features_list = []
        
        # Moving averages
        for window in [7, 14, 30, 50]:
            df[f'MA{window}'] = df['Close'].rolling(window).mean()
            features_list.append(f'MA{window}')
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        features_list.append('RSI')
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        features_list.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
        
        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(14).std()
        features_list.append('Volatility')
        
        # Volume MA
        df['Volume_MA'] = df['Volume'].rolling(10).mean()
        features_list.append('Volume_MA')
        
        # Drop NaN
        df = df.dropna()
        
        X = df[features_list]
        y = df['Target']
        
        return X, y
    
    def predict_direction(self, symbol: str) -> tuple:
        """
        Predict next day direction for a symbol.
        
        Returns:
            (prediction, confidence) where prediction is 1 (up) or 0 (down)
        """
        if not self.is_trained:
            logger.error("Model not trained")
            return (0, 0.5)
        
        # Get latest data
        data = self._load_data(symbol, 90)
        if data is None:
            return (0, 0.5)
        
        # Create features
        X, _ = self._create_features(data)
        if len(X) == 0:
            return (0, 0.5)
        
        # Latest features
        latest = X.iloc[-1:].values
        
        # Predict
        prediction = self.model.predict(latest)[0]
        proba = self.model.predict_proba(latest)[0]
        confidence = proba[prediction]
        
        return (prediction, confidence)
    
    def check_risk_limits(self) -> bool:
        """Check if we've hit risk limits."""
        # Check daily loss limit
        daily_trades = [t for t in self.trade_history 
                       if t['date'] == datetime.now().date()]
        daily_pnl = sum(t['pnl'] for t in daily_trades)
        
        daily_loss_limit = -self.initial_capital * 0.03  # 3% max daily loss
        if daily_pnl < daily_loss_limit:
            logger.warning(f"Daily loss limit hit: ${daily_pnl:.2f}")
            return False
        
        return True
    
    def place_order(self, symbol: str, side: str, quantity: float):
        """
        Place an order via Alpaca or simulate.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
        """
        if self.api:
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=int(quantity),
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                logger.info(f"Order placed: {side} {quantity} {symbol}")
                return order
            except Exception as e:
                logger.error(f"Order error: {e}")
                return None
        else:
            # Simulate
            logger.info(f"[SIMULATION] Order placed: {side} {quantity} {symbol}")
            return {'simulated': True}
    
    def run_daily_trading(self):
        """Run daily trading routine."""
        logger.info("="*70)
        logger.info("LIVE TRADING BOT - DAILY ROUTINE")
        logger.info("="*70)
        
        # Check risk limits
        if not self.check_risk_limits():
            logger.warning("Risk limits exceeded, pausing trading")
            return
        
        # Train models for symbols
        for symbol in self.symbols:
            if not self.is_trained:
                self.train_model(symbol, lookback_days=365)
            
            # Get prediction
            direction, confidence = self.predict_direction(symbol)
            
            logger.info(f"{symbol}: Direction={direction}, Confidence={confidence:.2%}")
            
            # Trading logic
            if direction == 1 and confidence > 0.55:  # Buy signal
                self._execute_buy(symbol, confidence)
            elif direction == 0 and symbol in self.positions:
                self._execute_sell(symbol)
        
        # Save results
        self._save_results()
    
    def _execute_buy(self, symbol: str, confidence: float):
        """Execute a buy order."""
        # Calculate position size
        available_cash = self.cash
        position_value = available_cash * self.position_size_pct
        current_price = self._get_current_price(symbol)
        quantity = position_value / current_price if current_price > 0 else 0
        
        if quantity < 1:
            logger.warning(f"Not enough cash for {symbol}")
            return
        
        # Place order
        order = self.place_order(symbol, 'buy', quantity)
        if order:
            self.positions[symbol] = quantity
            self.cash -= (quantity * current_price)
            
            # Log
            self.trade_history.append({
                'date': datetime.now().date(),
                'symbol': symbol,
                'action': 'buy',
                'quantity': quantity,
                'price': current_price,
                'confidence': confidence
            })
    
    def _execute_sell(self, symbol: str):
        """Execute a sell order."""
        if symbol not in self.positions:
            return
        
        quantity = self.positions[symbol]
        current_price = self._get_current_price(symbol)
        
        # Place order
        order = self.place_order(symbol, 'sell', quantity)
        if order:
            pnl = (current_price * quantity) - (self.positions[symbol] * 
                                                self._get_entry_price(symbol))
            
            self.trade_history.append({
                'date': datetime.now().date(),
                'symbol': symbol,
                'action': 'sell',
                'quantity': quantity,
                'price': current_price,
                'pnl': pnl
            })
            
            self.cash += (quantity * current_price)
            del self.positions[symbol]
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price of a symbol."""
        if self.api:
            try:
                last_trade = self.api.get_latest_trade(symbol)
                return float(last_trade.price)
            except:
                return 0.0
        else:
            # Use yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')
            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
            return 0.0
    
    def _get_entry_price(self, symbol: str) -> float:
        """Get entry price for a position (simplified)."""
        # In practice, track entry prices
        return self._get_current_price(symbol)
    
    def _save_results(self):
        """Save trading results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"trading_session_{timestamp}.json"
        
        results = {
            'config': self.config,
            'initial_capital': self.initial_capital,
            'current_cash': self.cash,
            'positions': self.positions,
            'trade_history': self.trade_history
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Stock Trading Bot')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ'],
                       help='Stock symbols to trade')
    parser.add_argument('--capital', type=float, default=10000.0,
                       help='Initial capital')
    parser.add_argument('--paper', action='store_true', default=True,
                       help='Use paper trading')
    parser.add_argument('--alpaca-key', type=str, help='Alpaca API key')
    parser.add_argument('--alpaca-secret', type=str, help='Alpaca secret key')
    
    args = parser.parse_args()
    
    config = {
        'symbols': args.symbols,
        'initial_capital': args.capital,
        'paper_trading': args.paper,
        'alpaca_api_key': args.alpaca_key,
        'alpaca_secret_key': args.alpaca_secret,
        'use_alpaca': bool(args.alpaca_key and args.alpaca_secret)
    }
    
    bot = LiveTradingBot(config)
    bot.run_daily_trading()


if __name__ == "__main__":
    main()

