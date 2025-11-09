import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

class EquitiesEnv:
    """
    A simple backtesting environment for equities trading.
    Simulates trading with slippage and commission costs.
    """
    def __init__(self, 
                 data: Dict[str, pd.DataFrame], 
                 starting_cash: float = 100000.0,
                 slippage_pct: float = 0.001,  # 0.1% slippage
                 commission_per_trade: float = 0.005): # $0.005 commission per share
        
        self.data = data
        self.symbols = list(data.keys())
        self.starting_cash = starting_cash
        self.slippage_pct = slippage_pct
        self.commission_per_trade = commission_per_trade
        
        self.reset()

    def reset(self) -> Dict[str, Any]:
        """Resets the environment to the initial state."""
        self.cash = self.starting_cash
        self.positions = {symbol: 0 for symbol in self.symbols}  # Shares held
        self.entry_price = {symbol: 0.0 for symbol in self.symbols} # Avg entry price
        self.current_step = 0
        self.max_steps = min(len(df) for df in self.data.values()) if self.data else 0
        self.history = []
        return self._get_state()

    def _get_state(self) -> Dict[str, Any]:
        """Returns the current market and portfolio state."""
        if self.current_step >= self.max_steps:
            return {}

        state = {
            'cash': self.cash,
            'equity': self.cash,
            'market_data': {}
        }
        
        total_market_value = 0
        for symbol in self.symbols:
            df = self.data[symbol]
            current_bar = df.iloc[self.current_step]
            current_price = current_bar['close']
            
            market_value = self.positions[symbol] * current_price
            total_market_value += market_value
            
            state['market_data'][symbol] = {
                'open': current_bar['open'],
                'high': current_bar['high'],
                'low': current_bar['low'],
                'close': current_price,
                'volume': current_bar['volume'],
                'position': self.positions[symbol]
            }
        
        state['equity'] = self.cash + total_market_value
        return state

    def _execute_trade(self, symbol: str, action: str, size: float) -> Tuple[float, float]:
        """Executes a trade and calculates costs."""
        df = self.data[symbol]
        if self.current_step + 1 >= len(df):
            return 0.0, 0.0 # Cannot trade on the last bar

        # Trade is executed at the next bar's open price +/- slippage
        trade_price = df.iloc[self.current_step + 1]['open']
        
        if action == 'buy':
            # Apply slippage (buy higher)
            final_price = trade_price * (1 + self.slippage_pct)
            
            # Calculate shares to buy (size is fraction of equity)
            max_shares = int((self.cash * size) / final_price)
            shares_to_buy = max_shares # For simplicity, buy max possible
            
            cost = shares_to_buy * final_price
            commission = shares_to_buy * self.commission_per_trade
            total_cost = cost + commission
            
            if total_cost > self.cash:
                # Not enough cash, adjust shares
                shares_to_buy = int((self.cash - commission) / final_price)
                total_cost = shares_to_buy * final_price + shares_to_buy * self.commission_per_trade
            
            if shares_to_buy > 0:
                # Update average entry price (weighted average)
                old_total_cost = self.positions[symbol] * self.entry_price[symbol]
                new_total_cost = old_total_cost + shares_to_buy * final_price
                new_total_shares = self.positions[symbol] + shares_to_buy
                
                self.entry_price[symbol] = new_total_cost / new_total_shares if new_total_shares > 0 else 0.0
                self.positions[symbol] += shares_to_buy
                self.cash -= total_cost
                return shares_to_buy, final_price
            
        elif action == 'sell':
            # Apply slippage (sell lower)
            final_price = trade_price * (1 - self.slippage_pct)
            
            # Calculate shares to sell (size is fraction of current position)
            shares_to_sell = int(self.positions[symbol] * size)
            
            if shares_to_sell > 0:
                commission = shares_to_sell * self.commission_per_trade
                revenue = shares_to_sell * final_price
                total_revenue = revenue - commission
                
                self.positions[symbol] -= shares_to_sell
                self.cash += total_revenue
                
                # If position is closed, reset entry price
                if self.positions[symbol] == 0:
                    self.entry_price[symbol] = 0.0
                
                return -shares_to_sell, final_price
                
        return 0.0, 0.0

    def step(self, actions: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Performs one step in the environment (one time period).
        actions: {'symbol': {'action': 'buy'|'sell'|'hold', 'size': float, 'tp': float|None, 'sl': float|None}}
        """
        if self.current_step >= self.max_steps:
            return self._get_state(), 0.0, True, {'msg': 'Episode finished'}

        initial_equity = self.cash + sum(self.positions[s] * self.data[s].iloc[self.current_step]['close'] for s in self.symbols)
        
        trade_info = {}
        for symbol, action_data in actions.items():
            action = action_data.get('action', 'hold').lower()
            size = action_data.get('size', 1.0) # 1.0 means full allocation/position
            
            if action in ['buy', 'sell']:
                shares_changed, price = self._execute_trade(symbol, action, size)
                trade_info[symbol] = {'shares': shares_changed, 'price': price, 'action': action}
            elif action == 'hold':
                pass # Do nothing
            
            # Simple Stop-Loss/Take-Profit check (executed at the next bar's close)
            # This is a simplification; real SL/TP would be checked intra-bar
            if self.current_step + 1 < self.max_steps and self.positions[symbol] > 0:
                next_close = self.data[symbol].iloc[self.current_step + 1]['close']
                
                # Check Stop Loss (SL)
                sl = action_data.get('sl')
                if sl is not None and next_close < self.entry_price[symbol] * (1 - sl):
                    # Execute full sell to stop loss
                    self._execute_trade(symbol, 'sell', 1.0)
                    trade_info[symbol]['sl_triggered'] = True
                    
                # Check Take Profit (TP)
                tp = action_data.get('tp')
                if tp is not None and next_close > self.entry_price[symbol] * (1 + tp):
                    # Execute full sell to take profit
                    self._execute_trade(symbol, 'sell', 1.0)
                    trade_info[symbol]['tp_triggered'] = True


        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        current_state = self._get_state()
        final_equity = current_state['equity']
        
        # PnL is the change in total equity
        pnl = final_equity - initial_equity
        
        # Reward is PnL for this step
        reward = pnl
        
        self.history.append({
            'step': self.current_step,
            'cash': self.cash,
            'equity': final_equity,
            'pnl': pnl,
            'trades': trade_info
        })

        return current_state, reward, done, {'trades': trade_info}

    def get_results(self) -> pd.DataFrame:
        """Returns the full trading history."""
        return pd.DataFrame(self.history)
