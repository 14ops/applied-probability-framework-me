import pandas as pd
import numpy as np
import glob
import os
from typing import Dict

class EquitiesLoader:
    """
    Utility to load and prepare equities data for the backtesting environment.
    """
    @staticmethod
    def load_csv_dir(path: str, low_liq: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Loads all CSV files from a directory into a dictionary of DataFrames.
        Assumes CSVs have columns: date, open, high, low, close, volume.
        """
        data = {}
        all_files = glob.glob(os.path.join(path, "*.csv"))
        
        if not all_files:
            print(f"Warning: No CSV files found in {path}. Returning synthetic data.")
            return EquitiesLoader._create_synthetic_data()

        for filename in all_files:
            try:
                df = pd.read_csv(filename)
                
                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    print(f"Skipping {filename}: Missing required columns.")
                    continue
                
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                
                if not low_liq:
                    df['dollar_volume'] = df['close'] * df['volume']
                    avg_dollar_volume = df['dollar_volume'].mean()
                    if avg_dollar_volume < 1_000_000:
                        print(f"Skipping {filename}: Low average dollar volume (${avg_dollar_volume:,.0f}).")
                        continue
                
                symbol = os.path.basename(filename).replace('.csv', '')
                data[symbol] = df[['open', 'high', 'low', 'close', 'volume']]
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
                
        return data

    @staticmethod
    def _create_synthetic_data() -> Dict[str, pd.DataFrame]:
        """Creates a small synthetic dataset for testing when no data is found."""
        dates = pd.date_range(start='2023-01-01', periods=100)
        
        def create_df(base_price, vol_factor, seed):
            np.random.seed(seed)
            prices = base_price + np.cumsum(np.random.randn(100) * vol_factor)
            prices = np.maximum(prices, 1.0)
            
            df = pd.DataFrame({
                'open': prices,
                'high': prices * (1 + np.random.rand(100) * 0.01),
                'low': prices * (1 - np.random.rand(100) * 0.01),
                'close': prices * (1 + np.random.randn(100) * 0.005),
                'volume': np.random.randint(10000, 50000, 100)
            }, index=dates)
            df['close'] = df['close'].shift(-1).fillna(df['close'].iloc[-1])
            return df.iloc[:-1]

        data = {
            'AAPL': create_df(150, 0.5, 42),
            'MSFT': create_df(300, 0.8, 43),
            'GOOG': create_df(100, 0.3, 44)
        }
        return data
