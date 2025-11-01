"""
Mines Bot Training Simulator
Simulates 10,000 rounds of Mines with evolving character strategies
"""

import json
import random
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import csv

# Import game components
from game_simulator import GameSimulator
from game.math import win_probability, payout_table_25_2, expected_value

# Note: Character strategies are defined but we'll use simplified decision logic
# They can be imported later if needed:
# from character_strategies.takeshi_kovacs import TakeshiKovacsStrategy
# from character_strategies.kazuya_kinoshita import KazuyaKinoshitaStrategy
# from character_strategies.lelouch_vi_britannia import LelouchViBritanniaStrategy


class CharacterAgent:
    """Represents a character agent with evolving parameters"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config.copy()
        self.profit_history = []
        self.wins = 0
        self.losses = 0
        self.total_profit = 0.0
        self.current_bankroll = config.get('initial_bankroll', 1000.0)
        self.initial_bankroll = self.current_bankroll
        
        # Tracking
        self.rounds_played = 0
        self.highest_profit = 0.0
        self.longest_win_streak = 0
        self.current_win_streak = 0
        
    def update(self, round_profit: float, win: bool):
        """Update agent based on round result"""
        self.rounds_played += 1
        self.total_profit += round_profit
        self.current_bankroll += round_profit
        self.profit_history.append(self.total_profit)
        
        if win:
            self.wins += 1
            self.current_win_streak += 1
            self.longest_win_streak = max(self.longest_win_streak, self.current_win_streak)
            
            # Positive reinforcement
            if round_profit > 0:
                self.config['risk_factor'] = min(1.0, self.config['risk_factor'] + self.config.get('learning_rate', 0.01))
                self.config['aggression_factor'] = min(1.0, self.config['aggression_factor'] + self.config.get('learning_rate', 0.01) * 0.5)
        else:
            self.losses += 1
            self.current_win_streak = 0
            
            # Negative reinforcement
            if round_profit < 0:
                self.config['risk_factor'] = max(0.0, self.config['risk_factor'] - self.config.get('learning_rate', 0.01) * 2)
                self.config['aggression_factor'] = max(0.0, self.config['aggression_factor'] - self.config.get('learning_rate', 0.01))
        
        self.highest_profit = max(self.highest_profit, self.total_profit)
    
    def should_continue(self) -> bool:
        """Check if agent should continue playing"""
        # Stop if bankroll depleted
        if self.current_bankroll <= 0:
            return False
        
        # Stop if hit stop loss threshold
        loss_ratio = abs(self.current_bankroll - self.initial_bankroll) / self.initial_bankroll
        if loss_ratio >= 0.5:  # 50% stop loss
            return False
            
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        win_rate = self.wins / self.rounds_played if self.rounds_played > 0 else 0
        return {
            'name': self.name,
            'total_profit': self.total_profit,
            'rounds_played': self.rounds_played,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'current_bankroll': self.current_bankroll,
            'highest_profit': self.highest_profit,
            'longest_win_streak': self.longest_win_streak,
            'risk_factor': self.config['risk_factor'],
            'aggression_factor': self.config['aggression_factor']
        }


class MinesTrainingEngine:
    """Main training engine for Mines bot"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize training engine with configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.mines_config = self.config['mines_bot']
        self.logging_config = self.config['logging']
        self.viz_config = self.config['visualization']
        
        # Create directories
        Path(self.logging_config['log_dir']).mkdir(exist_ok=True)
        Path(self.logging_config['results_dir']).mkdir(exist_ok=True)
        Path(self.logging_config['models_dir']).mkdir(exist_ok=True)
        
        # Initialize agents
        self.agents = {}
        for name, char_config in self.mines_config['characters'].items():
            self.agents[name] = CharacterAgent(name, {
                'initial_bankroll': self.mines_config['initial_bankroll'],
                **char_config
            })
        
        # Training state
        self.current_round = 0
        self.total_rounds = self.mines_config['total_rounds']
        self.profit_log = []
        self.decision_log = []
        
    def simulate_round(self, agent: CharacterAgent) -> Tuple[bool, float]:
        """
        Simulate a single round of Mines for an agent
        
        Returns:
            (win, profit) tuple
        """
        # Initialize game
        game = GameSimulator(
            board_size=self.mines_config['board_size'],
            mine_count=self.mines_config['mine_count'],
            seed=random.randint(0, 2**31)
        )
        game.initialize_game()
        
        bet_amount = self.mines_config['base_bet']
        cash_out_threshold = agent.config['cashout_threshold']
        payout_table = payout_table_25_2()
        
        clicks_made = 0
        max_clicks = self.mines_config['board_size'] ** 2 - self.mines_config['mine_count']
        
        # Decision loop
        while clicks_made < max_clicks:
            # Get valid actions (unrevealed cells)
            valid_actions = []
            for r in range(game.board_size):
                for c in range(game.board_size):
                    if not game.revealed[r][c]:
                        valid_actions.append((r, c))
            
            if not valid_actions:
                break
            
            # Decide action based on character strategy
            state = {
                'board_size': game.board_size,
                'mine_count': game.mine_count,
                'clicks_made': clicks_made,
                'current_payout': game.current_payout if hasattr(game, 'current_payout') else 0,
                'revealed': game.revealed,
                'max_clicks': max_clicks
            }
            
            # Simple decision logic (can be replaced with actual strategy)
            current_multiplier = clicks_made * 0.5 + 1.0 if clicks_made > 0 else 1.0
            
            # Decide: cash out or continue
            if clicks_made >= 3 and current_multiplier >= cash_out_threshold:
                # Cash out
                payout = current_multiplier * bet_amount
                profit = payout - bet_amount
                
                decision_info = {
                    'round': self.current_round,
                    'agent': agent.name,
                    'action': 'cashout',
                    'clicks': clicks_made,
                    'payout': current_multiplier,
                    'profit': profit,
                    'timestamp': datetime.now().isoformat()
                }
                self.decision_log.append(decision_info)
                
                return True, profit
            
            # Click a random cell
            chosen_cell = random.choice(valid_actions)
            success, result = game.click_cell(chosen_cell[0], chosen_cell[1])
            
            if result == "Mine":
                # Hit mine - lose
                decision_info = {
                    'round': self.current_round,
                    'agent': agent.name,
                    'action': 'click_mine',
                    'clicks': clicks_made + 1,
                    'payout': 0.0,
                    'profit': -bet_amount,
                    'timestamp': datetime.now().isoformat()
                }
                self.decision_log.append(decision_info)
                
                return False, -bet_amount
            
            clicks_made += 1
            
            # Check if won by clearing all
            if clicks_made >= max_clicks:
                final_multiplier = payout_table.get(clicks_made, 1.0)
                payout = final_multiplier * bet_amount
                profit = payout - bet_amount
                
                decision_info = {
                    'round': self.current_round,
                    'agent': agent.name,
                    'action': 'clear_all',
                    'clicks': clicks_made,
                    'payout': final_multiplier,
                    'profit': profit,
                    'timestamp': datetime.now().isoformat()
                }
                self.decision_log.append(decision_info)
                
                return True, profit
        
        # Should not reach here
        return False, -bet_amount
    
    def train(self):
        """Run training loop"""
        print("=" * 70)
        print("MINES BOT TRAINING ENGINE")
        print("=" * 70)
        print(f"Configuration: {self.mines_config['board_size']}x{self.mines_config['board_size']} board")
        print(f"Total rounds: {self.total_rounds}")
        print(f"Characters: {', '.join(self.agents.keys())}")
        print("=" * 70)
        print()
        
        start_time = time.time()
        
        while self.current_round < self.total_rounds:
            # Each round, all agents play
            for agent_name, agent in self.agents.items():
                if not agent.should_continue():
                    continue
                
                win, profit = self.simulate_round(agent)
                agent.update(profit, win)
                
                self.profit_log.append({
                    'round': self.current_round,
                    'agent': agent_name,
                    'profit': profit,
                    'total_profit': agent.total_profit
                })
            
            self.current_round += 1
            
            # Progress update
            if self.current_round % 1000 == 0:
                elapsed = time.time() - start_time
                progress = (self.current_round / self.total_rounds) * 100
                print(f"Progress: {self.current_round}/{self.total_rounds} ({progress:.1f}%)")
                print(f"Elapsed time: {elapsed:.1f}s")
                self.print_leaderboard()
                print()
        
        # Training complete
        elapsed = time.time() - start_time
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {elapsed:.1f}s")
        print()
        
        # Final results
        self.print_leaderboard()
        self.save_results()
    
    def print_leaderboard(self):
        """Print current leaderboard"""
        print("\n--- LEADERBOARD ---")
        sorted_agents = sorted(self.agents.values(), key=lambda a: a.total_profit, reverse=True)
        
        print(f"{'Rank':<6} {'Agent':<15} {'Profit':<12} {'Win%':<8} {'Rounds':<8} {'Bankroll':<12}")
        print("-" * 70)
        for rank, agent in enumerate(sorted_agents, 1):
            stats = agent.get_stats()
            win_rate = f"{stats['win_rate']:.1%}"
            print(f"{rank:<6} {stats['name']:<15} ${stats['total_profit']:<11.2f} {win_rate:<8} {stats['rounds_played']:<8} ${stats['current_bankroll']:<11.2f}")
    
    def save_results(self):
        """Save training results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.logging_config['results_dir'])
        
        # Save CSV log
        csv_path = results_dir / f"training_log_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['round', 'agent', 'profit', 'total_profit'])
            writer.writeheader()
            writer.writerows(self.profit_log)
        
        # Save decision log
        decision_csv = results_dir / f"decisions_{timestamp}.csv"
        if self.decision_log:
            with open(decision_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.decision_log[0].keys())
                writer.writeheader()
                writer.writerows(self.decision_log)
        
        # Save strategy evolution
        strategy_data = {}
        for agent_name, agent in self.agents.items():
            strategy_data[agent_name] = {
                'name': agent.name,
                'final_config': agent.config,
                'final_stats': agent.get_stats(),
                'profit_history': agent.profit_history
            }
        
        strategy_path = results_dir / f"strategy_evolution_{timestamp}.json"
        with open(strategy_path, 'w') as f:
            json.dump(strategy_data, f, indent=2)
        
        print(f"\nResults saved:")
        print(f"  - Training log: {csv_path}")
        print(f"  - Decision log: {decision_csv}")
        print(f"  - Strategy evolution: {strategy_path}")


def main():
    """Main entry point"""
    try:
        engine = MinesTrainingEngine()
        engine.train()
    except FileNotFoundError:
        print("ERROR: config.json not found. Please create it first.")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

