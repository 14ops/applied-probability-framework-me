"""
Main orchestration script to run both Mines and Trading training
"""

import argparse
import json
from pathlib import Path
import sys

from train_bot_simulator import MinesTrainingEngine
from train_trading_bot import TradingTrainingEngine


def main():
    parser = argparse.ArgumentParser(description='Train Mines and/or Trading bots')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['mines', 'trading', 'both'], default='both',
                       help='Which bots to run')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (fewer rounds)')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Please run this script from the project root directory.")
        return 1
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Quick mode adjustments
    if args.quick:
        print("Running in QUICK TEST mode...")
        if 'mines_bot' in config:
            config['mines_bot']['total_rounds'] = 100
        if 'trading_bot' in config:
            config['trading_bot']['lookback_days'] = 30
    
    # Save modified config for engines
    temp_config = Path('config_temp.json')
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        results = {}
        
        # Run Mines bot
        if args.mode in ['mines', 'both'] and config['mines_bot']['enabled']:
            print("\n" + "=" * 70)
            print("STARTING MINES BOT TRAINING")
            print("=" * 70)
            
            try:
                mines_engine = MinesTrainingEngine(str(temp_config))
                mines_engine.train()
                results['mines'] = 'success'
            except Exception as e:
                print(f"ERROR in Mines training: {e}")
                import traceback
                traceback.print_exc()
                results['mines'] = 'failed'
        
        # Run Trading bot
        if args.mode in ['trading', 'both'] and config['trading_bot']['enabled']:
            print("\n" + "=" * 70)
            print("STARTING TRADING BOT TRAINING")
            print("=" * 70)
            
            try:
                trading_engine = TradingTrainingEngine(str(temp_config))
                
                for symbol in config['trading_bot']['symbols']:
                    try:
                        trading_engine.train_and_simulate(symbol)
                        print("\n" + "=" * 70 + "\n")
                    except Exception as e:
                        print(f"ERROR processing {symbol}: {e}")
                        continue
                
                results['trading'] = 'success'
            except Exception as e:
                print(f"ERROR in Trading training: {e}")
                import traceback
                traceback.print_exc()
                results['trading'] = 'failed'
        
        # Summary
        print("\n" + "=" * 70)
        print("TRAINING SESSION COMPLETE")
        print("=" * 70)
        for mode, status in results.items():
            print(f"{mode.upper()}: {status}")
        print("=" * 70)
        
    finally:
        # Cleanup temp config
        if temp_config.exists():
            temp_config.unlink()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

