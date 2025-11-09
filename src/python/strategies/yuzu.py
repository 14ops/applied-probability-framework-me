from .base_strategy import BaseStrategy
from typing import Dict, Any
import pandas as pd
from ..indicators.ta import atr

class Yuzu(BaseStrategy):
    """
    Yuzu: Volatility Breakout Strategy.
    Buys when the price breaks out of a recent range, confirmed by high ATR.
    """
    def __init__(self, symbol: str):
        super().__init__(symbol, "Yuzu - Volatility Breakout")
        self.atr_period = 14
        self.atr_multiplier = 2.0
        self.lookback = 20

    def decide_action(self, 
                      current_state: Dict[str, Any], 
                      historical_data: pd.DataFrame) -> Dict[str, Any]:
        
        # Ensure enough data for ATR calculation
        if len(historical_data) < self.lookback:
            return {'action': 'hold', 'size': 0.0}

        # Calculate ATR
        historical_data['ATR'] = atr(historical_data, self.atr_period)
        current_atr = historical_data['ATR'].iloc[-1]
        
        # Define the recent price range
        lookback_data = historical_data.iloc[-self.lookback:]
        recent_high = lookback_data['high'].max()
        recent_low = lookback_data['low'].min()
        
        current_close = historical_data['close'].iloc[-1]
        position = current_state['market_data'][self.symbol]['position']
        
        action = 'hold'
        size = 0.0
        
        # Breakout threshold: recent high + ATR * multiplier
        breakout_threshold = recent_high + current_atr * self.atr_multiplier
        
        # Logic for Yuzu
        if position == 0:
            # Buy on upward breakout
            if current_close > breakout_threshold:
                action = 'buy'
                size = 1.0
            
        elif position > 0:
            # Simple exit: sell if price drops below the recent low
            if current_close < recent_low:
                action = 'sell'
                size = 1.0
            else:
                action = 'hold'
                
        # Breakout strategies often use a trailing stop loss
        return {
            'action': action, 
            'size': size, 
            'tp': 0.15, # 15% take profit
            'sl': 0.05  # 5% stop loss
        }
