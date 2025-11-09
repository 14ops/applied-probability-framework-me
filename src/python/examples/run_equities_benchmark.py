import pandas as pd
import numpy as np
import os
from typing import Dict, List

# Import core components
from ..data.equities_loader import EquitiesLoader
from ..environments.equities_env import EquitiesEnv
from ..strategies.base_strategy import BaseStrategy

# Import all character strategies
from ..strategies.takeshi import Takeshi
from ..strategies.aoi import Aoi
from ..strategies.yuzu import Yuzu
from ..strategies.kazuya import Kazuya
from ..strategies.lelouch import Lelouch

# Helper function to run a single backtest
def run_backtest(env: EquitiesEnv, strategy: BaseStrategy) -> pd.DataFrame:
    """Runs a backtest for a single strategy and returns the results."""
    
    # Reset environment
    env.reset()
    
    # Run simulation loop
    done = False
    while not done:
        # Get current state
        state = env._get_state()
        
        # Get historical data for the strategy's symbol
        symbol = strategy.symbol
        historical_data = env.data[symbol].iloc[:env.current_step + 1]
        
        # Strategy decides action
        action_data = strategy.decide_action(state, historical_data)
        
        # Format action for the environment step
        actions = {symbol: action_data}
        
        # Take a step in the environment
        state, reward, done, info = env.step(actions)
        
        # Update strategy's internal position tracking
        if symbol in info['trades']:
            strategy.update_position(info['trades'][symbol]['shares'])
            
    return env.get_results()

def analyze_results(results: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
    """Calculates key performance metrics from the backtest results."""
    if results.empty:
        return {
            'Strategy': strategy_name,
            'Total Return (%)': 0.0,
            'Max Drawdown (%)': 0.0,
            'Sharpe Ratio': 0.0,
            'Trades': 0
        }
        
    # Calculate Total Return
    initial_equity = results['equity'].iloc[0]
    final_equity = results['equity'].iloc[-1]
    total_return = (final_equity / initial_equity - 1) * 100
    
    # Calculate Max Drawdown
    peak = results['equity'].cummax()
    drawdown = (results['equity'] - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    # Calculate Sharpe Ratio (simplified, assuming risk-free rate is 0)
    daily_returns = results['equity'].pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) # Annualized
    
    # Calculate total trades (simplified count of non-hold steps)
    trades = results['trades'].apply(lambda x: len(x) > 0).sum()
    
    return {
        'Strategy': strategy_name,
        'Total Return (%)': round(total_return, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Trades': trades
    }

def main():
    print("--- Applied Probability Framework: Equities MVP Benchmark ---")
    
    # 1. Load Data (using synthetic data if no CSVs are present)
    # NOTE: In a real scenario, you would place your OHLCV CSV files in a 'data' directory
    # For this example, we use the synthetic data created in EquitiesLoader
    data_path = os.path.join(os.getcwd(), 'data')
    market_data = EquitiesLoader.load_csv_dir(data_path)
    
    if not market_data:
        print("Error: No market data loaded. Exiting.")
        return

    # Select a symbol to trade (e.g., the first one loaded)
    symbol_to_trade = list(market_data.keys())[0]
    print(f"Loaded data for: {list(market_data.keys())}")
    print(f"Running backtests on symbol: {symbol_to_trade}")
    
    # 2. Initialize Environment
    env = EquitiesEnv(data=market_data, starting_cash=100000.0)
    
    # 3. Define Strategies
    strategies: List[BaseStrategy] = [
        Takeshi(symbol_to_trade),
        Aoi(symbol_to_trade),
        Yuzu(symbol_to_trade),
        Kazuya(symbol_to_trade),
        Lelouch(symbol_to_trade)
    ]
    
    # 4. Run Benchmarks
    all_results = []
    for strategy in strategies:
        print(f"\nRunning backtest for: {strategy.name}...")
        results_df = run_backtest(env, strategy)
        analysis = analyze_results(results_df, strategy.name)
        all_results.append(analysis)
        
    # 5. Display Results
    results_table = pd.DataFrame(all_results)
    print("\n--- Benchmark Results ---")
    print(results_table.to_markdown(index=False))
    
    # Optional: Save results to CSV
    results_table.to_csv('equities_benchmark_results.csv', index=False)
    print("\nResults saved to equities_benchmark_results.csv")

if __name__ == "__main__":
    # Ensure the examples directory exists
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    
    # To allow relative imports to work, we need to adjust the path
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    
    main()
