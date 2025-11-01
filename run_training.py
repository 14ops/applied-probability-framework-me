#!/usr/bin/env python
"""
Quick start script to run training for both bots
"""

import sys
import argparse
from pathlib import Path

# Add src/python to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'python'))

from orchestrate_training import main as orchestrate_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run dual-mode training')
    parser.add_argument('--quick', action='store_true', help='Quick test (100 rounds, 30 days)')
    parser.add_argument('--mode', choices=['mines', 'trading', 'both'], default='both',
                       help='Which bots to run')
    
    args = parser.parse_args()
    
    # Create simplified config if needed
    config_path = Path('config.json')
    if not config_path.exists():
        print("ERROR: config.json not found!")
        print("Please run this from the project root directory.")
        sys.exit(1)
    
    # Modify config for quick mode
    if args.quick:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if 'mines_bot' in config:
            config['mines_bot']['total_rounds'] = 100
        if 'trading_bot' in config:
            config['trading_bot']['lookback_days'] = 30
        
        # Save temp config
        temp_config = Path('config_temp.json')
        with open(temp_config, 'w') as f:
            json.dump(config, f, indent=2)
        
        import sys
        sys.argv = ['orchestrate_training.py', '--config', str(temp_config), '--mode', args.mode]
        
        try:
            orchestrate_main()
        finally:
            if temp_config.exists():
                temp_config.unlink()
    else:
        # Run normally
        import sys
        sys.argv = ['orchestrate_training.py', '--mode', args.mode]
        orchestrate_main()

