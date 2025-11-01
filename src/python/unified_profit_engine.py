"""
Unified Profit Engine
Master orchestrator combining Mines automation + Stock trading

This is the crown jewel: the unified system that coordinates both
profit generation streams into a single autonomous entity.
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import threading

# Import our bots
import sys
sys.path.insert(0, str(Path(__file__).parent))

from automation.rainbet_mines_bot import RainBetMinesBot
from trading.live_trading_bot import LiveTradingBot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedProfitEngine:
    """
    Master engine coordinating both Mines and Trading bots.
    
    Features:
    - Dual-mode profit generation
    - Capital allocation between bots
    - Performance monitoring and rebalancing
    - Unified logging and reporting
    - Auto-scaling based on performance
    """
    
    def __init__(self, config_path: str = "config_unified.json"):
        """Initialize the unified engine."""
        self.config = self._load_config(config_path)
        
        # Capital management
        self.total_capital = self.config.get('total_capital', 10000.0)
        self.mines_allocation_pct = self.config.get('mines_allocation_pct', 0.3)
        self.trading_allocation_pct = self.config.get('trading_allocation_pct', 0.7)
        
        # Initialize bots
        mines_config = self.config.get('mines_bot', {})
        mines_config['initial_bankroll'] = self.total_capital * self.mines_allocation_pct
        
        trading_config = self.config.get('trading_bot', {})
        trading_config['initial_capital'] = self.total_capital * self.trading_allocation_pct
        
        self.mines_bot = RainBetMinesBot(mines_config)
        self.trading_bot = LiveTradingBot(trading_config)
        
        # Trackers
        self.start_time = datetime.now()
        self.total_profit = 0.0
        self.mines_profit = 0.0
        self.trading_profit = 0.0
        
        self.session_log = []
        self.rebalance_points = []
        
        # State
        self.running = False
        self.lock = threading.Lock()
        
        # Logging
        self.log_dir = Path('logs/unified')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*70)
        logger.info("UNIFIED PROFIT ENGINE INITIALIZED")
        logger.info("="*70)
        logger.info(f"Total Capital: ${self.total_capital:.2f}")
        logger.info(f"Mines Allocation: ${mines_config['initial_bankroll']:.2f}")
        logger.info(f"Trading Allocation: ${trading_config['initial_capital']:.2f}")
        logger.info("="*70)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found: {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'total_capital': 10000.0,
            'mines_allocation_pct': 0.3,
            'trading_allocation_pct': 0.7,
            'rebalance_threshold': 0.15,  # Rebalance if allocation drifts >15%
            'performance_window_days': 7,
            
            'mines_bot': {
                'character': 'hybrid',
                'bet_amount': 10.0,
                'total_rounds': 100,
                'anti_detection': True
            },
            
            'trading_bot': {
                'symbols': ['SPY', 'QQQ', 'AAPL'],
                'position_size_pct': 0.1,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'paper_trading': True
            }
        }
    
    def run_mines_session(self):
        """Run a Mines bot session."""
        logger.info("Starting Mines bot session...")
        self.mines_bot.run()
        
        # Update profit tracking
        with self.lock:
            self.mines_profit = self.mines_bot.current_profit
            self.total_profit = self.mines_profit + self.trading_profit
        
        logger.info(f"Mines session complete. Profit: ${self.mines_profit:.2f}")
    
    def run_trading_session(self):
        """Run a trading bot session."""
        logger.info("Starting Trading bot session...")
        self.trading_bot.run_daily_trading()
        
        # Update profit tracking
        with self.lock:
            self.trading_profit = sum(t.get('pnl', 0) for t in self.trading_bot.trade_history)
            self.total_profit = self.mines_profit + self.trading_profit
        
        logger.info(f"Trading session complete. Profit: ${self.trading_profit:.2f}")
    
    def should_rebalance(self) -> bool:
        """Check if capital needs rebalancing."""
        current_mines_value = self.total_capital * self.mines_allocation_pct + self.mines_profit
        current_trading_value = self.total_capital * self.trading_allocation_pct + self.trading_profit
        total_value = current_mines_value + current_trading_value
        
        if total_value <= 0:
            return False
        
        mines_pct = current_mines_value / total_value
        trading_pct = current_trading_value / total_value
        
        target_mines = self.mines_allocation_pct
        target_trading = self.trading_allocation_pct
        
        drift = abs(mines_pct - target_mines)
        threshold = self.config.get('rebalance_threshold', 0.15)
        
        return drift > threshold
    
    def rebalance(self):
        """Rebalance capital between bots."""
        logger.info("Rebalancing capital allocation...")
        
        current_mines_value = self.total_capital * self.mines_allocation_pct + self.mines_profit
        current_trading_value = self.total_capital * self.trading_allocation_pct + self.trading_profit
        total_value = current_mines_value + current_trading_value
        
        if total_value <= 0:
            logger.warning("Cannot rebalance with zero/negative capital")
            return
        
        # Calculate new allocations
        new_mines_capital = total_value * self.mines_allocation_pct
        new_trading_capital = total_value * self.trading_allocation_pct
        
        # Update bot capital
        self.mines_bot.bankroll = new_mines_capital
        self.trading_bot.cash = new_trading_capital
        
        # Log rebalance
        self.rebalance_points.append({
            'timestamp': datetime.now(),
            'total_value': total_value,
            'mines_allocation': new_mines_capital,
            'trading_allocation': new_trading_capital
        })
        
        logger.info(f"Rebalanced: Mines=${new_mines_capital:.2f}, Trading=${new_trading_capital:.2f}")
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        hours = elapsed_time / 3600
        
        return {
            'total_profit': self.total_profit,
            'mines_profit': self.mines_profit,
            'trading_profit': self.trading_profit,
            'total_return': (self.total_profit / self.total_capital) * 100,
            'runtime_hours': hours,
            'hourly_profit': self.total_profit / hours if hours > 0 else 0,
            'rebalances': len(self.rebalance_points)
        }
    
    def run(self, duration_hours: float = 24.0):
        """
        Run the unified engine for a specified duration.
        
        Args:
            duration_hours: How long to run (default 24 hours)
        """
        logger.info("="*70)
        logger.info("STARTING UNIFIED PROFIT ENGINE")
        logger.info("="*70)
        logger.info(f"Duration: {duration_hours} hours")
        logger.info(f"Target: $10,000 profit")
        logger.info("="*70)
        
        self.running = True
        end_time = datetime.now().timestamp() + (duration_hours * 3600)
        
        # Initial calibration
        logger.info("Calibrating bots...")
        if not self.mines_bot.calibrate():
            logger.error("Mines bot calibration failed")
            return
        
        logger.info("Calibration complete. Starting engine in 5 seconds...")
        time.sleep(5)
        
        round_count = 0
        
        while self.running and time.time() < end_time:
            round_count += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"ROUND {round_count}")
            logger.info(f"{'='*70}")
            
            # Run both bots in parallel (separate threads)
            mines_thread = threading.Thread(target=self.run_mines_session)
            trading_thread = threading.Thread(target=self.run_trading_session)
            
            mines_thread.start()
            trading_thread.start()
            
            mines_thread.join()
            trading_thread.join()
            
            # Check performance
            metrics = self.get_performance_metrics()
            logger.info("\nPerformance Summary:")
            logger.info(f"Total Profit: ${metrics['total_profit']:.2f}")
            logger.info(f"Mines Profit: ${metrics['mines_profit']:.2f}")
            logger.info(f"Trading Profit: ${metrics['trading_profit']:.2f}")
            logger.info(f"Total Return: {metrics['total_return']:.1f}%")
            logger.info(f"Hourly Profit: ${metrics['hourly_profit']:.2f}")
            
            # Check if goal reached
            if self.total_profit >= 10000.0:
                logger.info("="*70)
                logger.info("🎉 $10,000 PROFIT TARGET REACHED!")
                logger.info("="*70)
                break
            
            # Rebalance if needed
            if self.should_rebalance():
                self.rebalance()
            
            # Inter-round pause
            logger.info("Pausing for 1 minute before next round...")
            time.sleep(60)
        
        # Final summary
        self._print_final_summary()
        self._save_results()
    
    def _print_final_summary(self):
        """Print final results summary."""
        metrics = self.get_performance_metrics()
        
        logger.info("="*70)
        logger.info("UNIFIED PROFIT ENGINE - SESSION COMPLETE")
        logger.info("="*70)
        logger.info(f"Total Profit: ${metrics['total_profit']:.2f}")
        logger.info(f"Mines Contribution: ${metrics['mines_profit']:.2f}")
        logger.info(f"Trading Contribution: ${metrics['trading_profit']:.2f}")
        logger.info(f"Total Return: {metrics['total_return']:.1f}%")
        logger.info(f"Runtime: {metrics['runtime_hours']:.1f} hours")
        logger.info(f"Rebalances: {metrics['rebalances']}")
        logger.info("="*70)
        
        # Goal status
        if metrics['total_profit'] >= 10000.0:
            logger.info("✅ $10,000 PROFIT TARGET: ACHIEVED")
        else:
            remaining = 10000.0 - metrics['total_profit']
            logger.info(f"⚠️  $10,000 PROFIT TARGET: ${remaining:.2f} remaining")
        logger.info("="*70)
    
    def _save_results(self):
        """Save comprehensive results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"unified_session_{timestamp}.json"
        
        metrics = self.get_performance_metrics()
        
        results = {
            'config': self.config,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'metrics': metrics,
            'rebalance_points': [
                {
                    'timestamp': r['timestamp'].isoformat(),
                    'total_value': r['total_value'],
                    'mines_allocation': r['mines_allocation'],
                    'trading_allocation': r['trading_allocation']
                }
                for r in self.rebalance_points
            ],
            'goal_status': {
                'target': 10000.0,
                'achieved': metrics['total_profit'],
                'success': metrics['total_profit'] >= 10000.0
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Profit Engine')
    parser.add_argument('--config', type=str, default='config_unified.json',
                       help='Configuration file')
    parser.add_argument('--duration', type=float, default=24.0,
                       help='Duration in hours')
    parser.add_argument('--capital', type=float, help='Initial capital')
    parser.add_argument('--mines-alloc', type=float, help='Mines allocation %')
    parser.add_argument('--trading-alloc', type=float, help='Trading allocation %')
    
    args = parser.parse_args()
    
    engine = UnifiedProfitEngine(config_path=args.config)
    
    # Override config if provided
    if args.capital:
        engine.total_capital = args.capital
    if args.mines_alloc:
        engine.mines_allocation_pct = args.mines_alloc / 100
    if args.trading_alloc:
        engine.trading_allocation_pct = args.trading_alloc / 100
    
    engine.run(duration_hours=args.duration)


if __name__ == "__main__":
    main()

