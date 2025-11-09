from .base_strategy import BaseStrategy
from typing import Dict, Any
import pandas as pd
from ..indicators.ta import rsi, bollinger_bands

class Lelouch(BaseStrategy):
    """
    Lelouch: Meta-Strategist Strategy.
    Combines two simple strategies (RSI and Bollinger Bands) and only acts 
    when both signals agree, leading to high-conviction trades.
    """
    def __init__(self, symbol: str):
        super().__init__(symbol, "Lelouch - Meta-Strategist")
        self.rsi_period = 14
        self.rsi_buy_threshold = 50
        self.rsi_sell_threshold = 50
        self.bb_period = 20
        self.bb_std_dev = 2.0

    def decide_action(self, 
                      current_state: Dict[str, Any], 
                      historical_data: pd.DataFrame) -> Dict[str, Any]:
        
        # Ensure enough data for both indicators
        if len(historical_data) < max(self.rsi_period, self.bb_period):
            return {'action': 'hold', 'size': 0.0}

        # 1. Calculate RSI
        historical_data['RSI'] = rsi(historical_data['close'], self.rsi_period)
        current_rsi = historical_data['RSI'].iloc[-1]
        
        # 2. Calculate Bollinger Bands
        bb_df = bollinger_bands(historical_data['close'], self.bb_period, self.bb_std_dev)
        historical_data = historical_data.join(bb_df)
        current_close = historical_data['close'].iloc[-1]
        lower_band = historical_data['BB_LOWER'].iloc[-1]
        upper_band = historical_data['BB_UPPER'].iloc[-1]
        
        position = current_state['market_data'][self.symbol]['position']
        
        action = 'hold'
        size = 0.0
        
        # Signal 1: RSI Buy/Sell (RSI > 50 is bullish, RSI < 50 is bearish)
        rsi_buy_signal = current_rsi > self.rsi_buy_threshold
        rsi_sell_signal = current_rsi < self.rsi_sell_threshold
        
        # Signal 2: Bollinger Band Buy/Sell (Price below lower band is buy, above upper band is sell)
        bb_buy_signal = current_close < lower_band
        bb_sell_signal = current_close > upper_band
        
        # Logic for Lelouch (High Conviction)
        if position == 0:
            # Buy only if both signals agree (RSI is bullish AND BB suggests oversold)
            if rsi_buy_signal and bb_buy_signal:
                action = 'buy'
                size = 1.0
                
        elif position > 0:
            # Sell only if both signals agree (RSI is bearish AND BB suggests overbought)
            if rsi_sell_signal and bb_sell_signal:
                action = 'sell'
                size = 1.0
            else:
                action = 'hold'
                
        # Meta-strategies are often more patient and use wider stops
        return {
            'action': action, 
            'size': size, 
            'tp': 0.20, # 20% take profit
            'sl': 0.15  # 15% stop loss
        }
