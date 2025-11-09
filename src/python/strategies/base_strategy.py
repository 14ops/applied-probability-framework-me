from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    All strategies must implement the decide_action method.
    """
    def __init__(self, symbol: str, name: str):
        self.symbol = symbol
        self.name = name
        self.position = 0 # Shares held

    @abstractmethod
    def decide_action(self, 
                      current_state: Dict[str, Any], 
                      historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the current state and historical data to decide on a trading action.
        
        Args:
            current_state: The current state of the environment.
            historical_data: The historical OHLCV data for the symbol up to the current step.
            
        Returns:
            A dictionary with the trading action and parameters:
            {'action': 'buy'|'sell'|'hold', 'size': float, 'tp': float|None, 'sl': float|None}
        """
        pass

    def update_position(self, shares_changed: float):
        """Updates the internal position tracking."""
        self.position += shares_changed
