from .base_strategy import BaseStrategy
from typing import Dict, Any
import pandas as pd
from ..indicators.ta import sma

class Takeshi(BaseStrategy):
    """
    Takeshi: Aggressive Berserker Strategy (Simple SMA Crossover).
    Buys when the short-term SMA crosses above the long-term SMA.
    Sells when the short-term SMA crosses below the long-term SMA.
    """
    def __init__(self, symbol: str):
        super().__init__(symbol, "Takeshi - Aggressive Berserker (SMA Crossover)")
        self.short_period = 10
        self.long_period = 50

    def decide_action(self, 
                      current_state: Dict[str, Any], 
                      historical_data: pd.DataFrame) -> Dict[str, Any]:
        
        # Ensure enough data for SMA calculation
        if len(historical_data) < self.long_period:
            return {'action': 'hold', 'size': 0.0}

        # Calculate SMAs
        historical_data['SMA_Short'] = sma(historical_data['close'], self.short_period)
        historical_data['SMA_Long'] = sma(historical_data['close'], self.long_period)
        
        # Check for cross-over/under in the last two periods
        if len(historical_data) < 2:
            return {'action': 'hold', 'size': 0.0}
            
        sma_short_prev = historical_data['SMA_Short'].iloc[-2]
        sma_long_prev = historical_data['SMA_Long'].iloc[-2]
        sma_short_curr = historical_data['SMA_Short'].iloc[-1]
        sma_long_curr = historical_data['SMA_Long'].iloc[-1]
        
        position = current_state['market_data'][self.symbol]['position']
        
        action = 'hold'
        size = 0.0
        
        # Golden Cross (Buy Signal)
        golden_cross = (sma_short_prev <= sma_long_prev) and (sma_short_curr > sma_long_curr)
        
        # Death Cross (Sell Signal)
        death_cross = (sma_short_prev >= sma_long_prev) and (sma_short_curr < sma_long_curr)
        
        # Logic for Takeshi
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
                
        # Aggressive strategy with a moderate stop-loss
        return {
            'action': action, 
            'size': size, 
            'tp': None, # No fixed take-profit, ride the trend
            'sl': 0.05  # 5% stop loss
        }
