"""
Crypto/Stock Market Prediction Training Bot
Uses ML to predict price direction and simulate trading
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import csv

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


class TradingDataLoader:
    """Load and prepare trading data"""
    
    def __init__(self, symbol: str, lookback_days: int = 365):
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """Load historical data from yfinance"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            print(f"Loading data for {self.symbol} from {start_date.date()} to {end_date.date()}...")
            
            # For crypto, try with -USD suffix if needed
            if not self.symbol.endswith('USD') and not self.symbol.endswith('USDT'):
                symbol_with_suffix = f"{self.symbol}-USD"
                try:
                    ticker = yf.Ticker(symbol_with_suffix)
                    self.data = ticker.history(start=start_date, end=end_date)
                except:
                    ticker = yf.Ticker(self.symbol)
                    self.data = ticker.history(start=start_date, end=end_date)
            else:
                ticker = yf.Ticker(self.symbol.replace('USDT', '-USD'))
                self.data = ticker.history(start=start_date, end=end_date)
            
            if self.data.empty:
                raise ValueError(f"No data retrieved for {self.symbol}")
            
            print(f"Loaded {len(self.data)} days of data")
            return self.data
            
        except Exception as e:
            print(f"Error loading data for {self.symbol}: {e}")
            # Generate synthetic data for testing
            print("Generating synthetic data for testing...")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic price data for testing"""
        dates = pd.date_range(
            end=datetime.now(),
            periods=self.lookback_days,
            freq='D'
        )
        
        # Random walk with drift
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': np.random.uniform(1000000, 10000000, len(dates))
        }, index=dates)
        
        return df
    
    def create_features(self, config: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """Create feature matrix and target labels"""
        df = self.data.copy()
        
        # Create target: 1 if price goes up tomorrow, 0 if down
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df = df.dropna()
        
        # Feature engineering
        features_list = []
        
        # Moving averages
        if config.get('moving_average_7', True):
            df['MA7'] = df['Close'].rolling(window=7).mean()
            features_list.append('MA7')
        
        if config.get('moving_average_14', True):
            df['MA14'] = df['Close'].rolling(window=14).mean()
            features_list.append('MA14')
        
        if config.get('moving_average_30', True):
            df['MA30'] = df['Close'].rolling(window=30).mean()
            features_list.append('MA30')
        
        # RSI
        if config.get('rsi', True):
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=config.get('rsi_period', 14)).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=config.get('rsi_period', 14)).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            features_list.append('RSI')
        
        # MACD
        if config.get('macd', True):
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            features_list.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
        
        # Volume MA
        if config.get('volume_ma', True):
            df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
            features_list.append('Volume_MA')
        
        # Volatility
        if config.get('volatility', True):
            df['Volatility'] = df['Close'].pct_change().rolling(window=14).std()
            features_list.append('Volatility')
        
        # Drop NaN rows
        df = df.dropna()
        
        # Return features and target
        X = df[features_list]
        y = df['Target']
        
        return X, y


class TradingBot:
    """ML-based trading bot"""
    
    def __init__(self, model_type: str = "random_forest", model_params: Dict = None):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.feature_names = None
        self.is_trained = False
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the model"""
        if not SKLEARN_AVAILABLE:
            print("Error: sklearn not available. Cannot train model.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining {self.model_type} model...")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Create model
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 10),
                min_samples_split=self.model_params.get('min_samples_split', 5),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"  Training accuracy: {train_score:.2%}")
        print(f"  Test accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
        
        self.feature_names = list(X.columns)
        self.is_trained = True
        
        return accuracy
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)


class TradingSimulator:
    """Simulate trading with the trained model"""
    
    def __init__(self, initial_capital: float, position_size_pct: float,
                 stop_loss_pct: float, take_profit_pct: float, 
                 transaction_fee_pct: float):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.transaction_fee_pct = transaction_fee_pct
        
        self.position = 0  # Number of shares/crypto
        self.entry_price = 0
        self.trade_history = []
        self.portfolio_value_history = []
        
    def simulate(self, df: pd.DataFrame, bot: TradingBot, X: pd.DataFrame):
        """Run trading simulation"""
        print("\nRunning trading simulation...")
        
        for idx, row in df.iterrows():
            current_price = row['Close']
            current_features = X.loc[idx:idx]
            
            # Get prediction
            prediction = bot.predict(current_features)[0]
            proba = bot.predict_proba(current_features)[0]
            
            # Check if we have an open position
            if self.position > 0:
                # We're in a long position
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                
                # Check stop loss
                if pnl_pct <= -self.stop_loss_pct:
                    self._close_position(current_price, "stop_loss")
                
                # Check take profit
                elif pnl_pct >= self.take_profit_pct:
                    self._close_position(current_price, "take_profit")
                
                # Hold otherwise
            else:
                # No position - look for entry signal
                if prediction == 1 and proba[1] > 0.55:  # Buy signal with >55% confidence
                    self._open_position(current_price)
            
            # Record portfolio value
            portfolio_value = self.capital + (self.position * current_price if self.position > 0 else 0)
            self.portfolio_value_history.append({
                'date': idx,
                'value': portfolio_value,
                'price': current_price
            })
        
        # Close any open position at end
        if self.position > 0:
            final_price = df['Close'].iloc[-1]
            self._close_position(final_price, "end_of_simulation")
        
        # Calculate metrics
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        wins = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        losses = sum(1 for trade in self.trade_history if trade['pnl'] < 0)
        total_trades = len(self.trade_history)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        print(f"\n--- Trading Results ---")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.capital:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Wins: {wins}, Losses: {losses}")
        print(f"Win Rate: {win_rate:.2%}")
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'final_capital': self.capital
        }
    
    def _open_position(self, price: float):
        """Open a long position"""
        investment = self.capital * self.position_size_pct
        self.position = investment / price
        self.entry_price = price
        
        # Pay transaction fee
        fee = investment * self.transaction_fee_pct
        self.capital -= (investment + fee)
    
    def _close_position(self, price: float, reason: str):
        """Close the current position"""
        proceeds = self.position * price
        
        # Pay transaction fee
        fee = proceeds * self.transaction_fee_pct
        net_proceeds = proceeds - fee
        
        pnl = net_proceeds - (self.position * self.entry_price)
        
        self.trade_history.append({
            'entry_price': self.entry_price,
            'exit_price': price,
            'pnl': pnl,
            'return_pct': (pnl / (self.position * self.entry_price)) * 100,
            'reason': reason
        })
        
        self.capital += net_proceeds
        self.position = 0
        self.entry_price = 0


class TradingTrainingEngine:
    """Main training engine for trading bot"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize training engine"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.trading_config = self.config['trading_bot']
        self.logging_config = self.config['logging']
        
        # Create directories
        Path(self.logging_config['log_dir']).mkdir(exist_ok=True)
        Path(self.logging_config['results_dir']).mkdir(exist_ok=True)
        Path(self.logging_config['models_dir']).mkdir(exist_ok=True)
    
    def train_and_simulate(self, symbol: str):
        """Train model and run simulation for a symbol"""
        print("=" * 70)
        print(f"CRYPTO/STOCK TRADING BOT - {symbol}")
        print("=" * 70)
        
        # Load data
        loader = TradingDataLoader(symbol, self.trading_config['lookback_days'])
        data = loader.load_data()
        
        # Create features
        X, y = loader.create_features(self.trading_config['features'])
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target distribution:")
        print(y.value_counts())
        
        # Train bot
        bot = TradingBot(
            model_type=self.trading_config['model']['type'],
            model_params=self.trading_config['model']
        )
        
        accuracy = bot.train(X, y)
        
        # Run simulation
        simulator = TradingSimulator(
            initial_capital=self.trading_config['initial_capital'],
            position_size_pct=self.trading_config['position_size_pct'],
            stop_loss_pct=self.trading_config['stop_loss_pct'],
            take_profit_pct=self.trading_config['take_profit_pct'],
            transaction_fee_pct=self.trading_config['transaction_fee_pct']
        )
        
        results = simulator.simulate(data, bot, X)
        
        # Save results
        self._save_results(symbol, bot, simulator, results)
        
        return results
    
    def _save_results(self, symbol: str, bot: TradingBot, simulator: TradingSimulator, results: Dict):
        """Save training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.logging_config['results_dir'])
        
        # Save trade history
        if simulator.trade_history:
            trades_csv = results_dir / f"trades_{symbol}_{timestamp}.csv"
            pd.DataFrame(simulator.trade_history).to_csv(trades_csv, index=False)
            print(f"\nTrade history saved: {trades_csv}")
        
        # Save portfolio history
        portfolio_csv = results_dir / f"portfolio_{symbol}_{timestamp}.csv"
        pd.DataFrame(simulator.portfolio_value_history).to_csv(portfolio_csv, index=False)
        print(f"Portfolio history saved: {portfolio_csv}")
        
        # Save results summary
        summary = {
            'symbol': symbol,
            'timestamp': timestamp,
            'results': results,
            'config': self.trading_config
        }
        summary_json = results_dir / f"summary_{symbol}_{timestamp}.json"
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved: {summary_json}")


def main():
    """Main entry point"""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        if not config['trading_bot']['enabled']:
            print("Trading bot is disabled in config.json")
            return
        
        engine = TradingTrainingEngine()
        
        # Train for each symbol
        for symbol in config['trading_bot']['symbols']:
            try:
                engine.train_and_simulate(symbol)
                print("\n" + "=" * 70 + "\n")
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
    except FileNotFoundError:
        print("ERROR: config.json not found. Please create it first.")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

