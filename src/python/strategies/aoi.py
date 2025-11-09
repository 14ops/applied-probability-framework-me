from .base_strategy import BaseStrategy
from typing import Dict, Any
import pandas as pd
from ..indicators.ta import bollinger_bands

class Aoi(BaseStrategy):
    """
    Aoi: Mean Reversion Specialist Strategy.
    Buys when price is below the lower Bollinger Band and sells when it hits the upper band.
    """
    def __init__(self, symbol: str):
        super().__init__(symbol, "Aoi - Mean Reversion Specialist")
        self.bb_period = 20
        self.bb_std_dev = 2.0

    def decide_action(self, 
                      current_state: Dict[str, Any], 
                      historical_data: pd.DataFrame) -> Dict[str, Any]:
        
        # Ensure enough data for Bollinger Bands calculation
        if len(historical_data) < self.bb_period:
            return {'action': 'hold', 'size': 0.0}

        # Calculate Bollinger Bands
        bb_df = bollinger_bands(historical_data['close'], self.bb_period, self.bb_std_dev)
        historical_data = historical_data.join(bb_df)
        
        current_close = historical_data['close'].iloc[-1]
        lower_band = historical_data['BB_LOWER'].iloc[-1]
        upper_band = historical_data['BB_UPPER'].iloc[-1]
        
        position = current_state['market_data'][self.symbol]['position']
        
        action = 'hold'
        size = 0.0
        
        # Logic for Aoi
        if position == 0:
            # No position: Buy when price is below the lower band (oversold)
            if current_close < lower_band:
                action = 'buy'
                size = 1.0 # Full allocation
                
        elif position > 0:
            # Has position: Sell when price hits the upper band (overbought/reversion complete)
            if current_close > upper_band:
                action = 'sell'
                size = 1.0 # Sell all
            else:
                action = 'hold'
                
        # Mean reversion strategies often use a wide stop-loss or no stop-loss, 
        # but a take-profit at the mean (SMA) is common. We'll use a simple TP.
        return {
            'action': action, 
            'size': size, 
            'tp': 0.05, # 5% take profit
            'sl': 0.10  # 10% stop loss
        }
