from .base_strategy import BaseStrategy
from typing import Dict, Any
import pandas as pd
from ..indicators.ta import ema

class Kazuya(BaseStrategy):
    """
    Kazuya: Defensive Trend Follower Strategy.
    Buys when the short-term EMA crosses above the long-term EMA (Golden Cross).
    Sells when the short-term EMA crosses below the long-term EMA (Death Cross).
    """
    def __init__(self, symbol: str):
        super().__init__(symbol, "Kazuya - Defensive Trend Follower")
        self.short_period = 10
        self.long_period = 50

    def decide_action(self, 
                      current_state: Dict[str, Any], 
                      historical_data: pd.DataFrame) -> Dict[str, Any]:
        
        # Ensure enough data for EMA calculation
        if len(historical_data) < self.long_period:
            return {'action': 'hold', 'size': 0.0}

        # Calculate EMAs
        historical_data['EMA_Short'] = ema(historical_data['close'], self.short_period)
        historical_data['EMA_Long'] = ema(historical_data['close'], self.long_period)
        
        # Check for cross-over/under in the last two periods
        # Need at least 2 rows for cross check
        if len(historical_data) < 2:
            return {'action': 'hold', 'size': 0.0}
            
        ema_short_prev = historical_data['EMA_Short'].iloc[-2]
        ema_long_prev = historical_data['EMA_Long'].iloc[-2]
        ema_short_curr = historical_data['EMA_Short'].iloc[-1]
        ema_long_curr = historical_data['EMA_Long'].iloc[-1]
        
        position = current_state['market_data'][self.symbol]['position']
        
        action = 'hold'
        size = 0.0
        
        # Golden Cross (Buy Signal)
        golden_cross = (ema_short_prev <= ema_long_prev) and (ema_short_curr > ema_long_curr)
        
        # Death Cross (Sell Signal)
        death_cross = (ema_short_prev >= ema_long_prev) and (ema_short_curr < ema_long_curr)
        
        # Logic for Kazuya
        if position == 0:
            if golden_cross:
                action = 'buy'
                size = 1.0
        elif position > 0:
            if death_cross:
                action = 'sell'
                size = 1.0
            else:
                action = 'hold'
                
        # Trend following is patient, so a wider stop-loss is acceptable
        return {
            'action': action, 
            'size': size, 
            'tp': None, # No fixed take-profit, ride the trend
            'sl': 0.10  # 10% stop loss
        }
