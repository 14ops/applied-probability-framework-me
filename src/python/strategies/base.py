"""
Strategy Base Interface

Formal plugin interface for Mines game strategies. All character strategies
must inherit from StrategyBase and implement the required methods.

This provides a standard interface for:
- Bet sizing decisions
- Click/cashout decisions
- Result tracking and learning
- State serialization
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import random
import json


class StrategyBase(ABC):
    """
    Abstract base class for all Mines game strategies.
    
    A strategy must implement:
    1. decide(state) - Make betting and action decisions
    2. on_result(result) - Process game outcomes
    
    Optional methods:
    - reset() - Reset strategy state between sessions
    - serialize() - Save strategy state
    - deserialize() - Load strategy state
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 rng: Optional[random.Random] = None):
        """
        Initialize strategy.
        
        Args:
            config: Configuration dictionary with strategy parameters
            rng: Random number generator for reproducibility
        """
        self.config = config or {}
        self.rng = rng or random.Random()
        
        # Standard parameters
        self.name = self.config.get('name', self.__class__.__name__)
        self.base_bet = self.config.get('base_bet', 10.0)
        self.max_bet = self.config.get('max_bet', 100.0)
        self.bankroll = self.config.get('initial_bankroll', 1000.0)
        
        # Tracking
        self.games_played = 0
        self.total_profit = 0.0
        self.win_streak = 0
        self.loss_streak = 0
        self.last_result = None
        
        # History
        self.result_history: List[Dict] = []
        
    @abstractmethod
    def decide(self, state: Dict[str, Any], 
               uncertainty_vector: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Make decision about bet amount and whether to click or cashout.
        
        This is the core strategy logic that must be implemented.
        
        Args:
            state: Current game state containing:
                - revealed: 2D array of revealed tiles
                - board: 2D array of tile values  
                - board_size: Size of board
                - mine_count: Number of mines
                - clicks_made: Number of clicks so far
                - current_multiplier: Current payout multiplier
                - bankroll: Available funds
                - game_active: Whether game is currently active
            uncertainty_vector: Optional list of (mean, variance) tuples per cell
                               for Bayesian uncertainty information
                
        Returns:
            Dictionary with decision:
                {
                    "action": "click" | "cashout" | "bet",
                    "bet": float (bet amount if action is "bet"),
                    "position": tuple (row, col) if action is "click",
                    "max_clicks": int (optional target number of clicks)
                }
                
        Example:
            >>> # To place a bet and start game
            >>> return {"action": "bet", "bet": 10.0, "max_clicks": 7}
            >>> 
            >>> # To click a specific position
            >>> return {"action": "click", "position": (2, 3)}
            >>>
            >>> # To cashout
            >>> return {"action": "cashout"}
        """
        pass
    
    @abstractmethod
    def on_result(self, result: Dict[str, Any]) -> None:
        """
        Process game result and update strategy state.
        
        This is called after each game completes, allowing the strategy
        to learn and adapt based on outcomes.
        
        Args:
            result: Game result containing:
                - win: bool (whether game was won)
                - payout: float (payout multiplier achieved)
                - clicks: int (number of safe clicks made)
                - profit: float (net profit/loss)
                - final_bankroll: float (bankroll after game)
                - bet_amount: float (amount that was bet)
                
        Example:
            >>> result = {
            ...     "win": True,
            ...     "payout": 2.12,
            ...     "clicks": 8,
            ...     "profit": 11.20,
            ...     "final_bankroll": 1011.20,
            ...     "bet_amount": 10.0
            ... }
            >>> strategy.on_result(result)
        """
        pass
    
    def reset(self) -> None:
        """
        Reset strategy state for new session.
        
        This is called between sessions or when explicitly requested.
        Should reset temporary state but may preserve learned parameters.
        """
        self.games_played = 0
        self.total_profit = 0.0
        self.win_streak = 0
        self.loss_streak = 0
        self.last_result = None
        self.result_history = []
    
    def update_tracking(self, result: Dict[str, Any]) -> None:
        """
        Update internal tracking statistics.
        
        Helper method that can be called from on_result() to maintain
        standard tracking across all strategies.
        
        Args:
            result: Game result dictionary
        """
        self.games_played += 1
        self.total_profit += result.get('profit', 0.0)
        self.bankroll = result.get('final_bankroll', self.bankroll)
        self.last_result = result
        self.result_history.append(result)
        
        # Update streaks
        if result.get('win', False):
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current strategy statistics.
        
        Returns:
            Dictionary with performance metrics:
                - name: Strategy name
                - games_played: Total games
                - total_profit: Net profit/loss
                - win_streak: Current win streak
                - loss_streak: Current loss streak
                - bankroll: Current bankroll
                - win_rate: Win percentage (if games > 0)
                - avg_profit: Average profit per game
        """
        wins = sum(1 for r in self.result_history if r.get('win', False))
        win_rate = (wins / self.games_played * 100) if self.games_played > 0 else 0.0
        avg_profit = (self.total_profit / self.games_played) if self.games_played > 0 else 0.0
        
        return {
            'name': self.name,
            'games_played': self.games_played,
            'total_profit': round(self.total_profit, 2),
            'win_streak': self.win_streak,
            'loss_streak': self.loss_streak,
            'bankroll': round(self.bankroll, 2),
            'win_rate': round(win_rate, 2),
            'avg_profit': round(avg_profit, 2),
            'wins': wins,
            'losses': self.games_played - wins,
        }
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize strategy state to dictionary.
        
        Can be overridden to save strategy-specific state.
        
        Returns:
            Dictionary containing strategy state
        """
        return {
            'name': self.name,
            'config': self.config,
            'games_played': self.games_played,
            'total_profit': self.total_profit,
            'win_streak': self.win_streak,
            'loss_streak': self.loss_streak,
            'bankroll': self.bankroll,
            'result_history': self.result_history[-100:],  # Keep last 100
        }
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """
        Load strategy state from dictionary.
        
        Can be overridden to load strategy-specific state.
        
        Args:
            data: Dictionary containing strategy state
        """
        self.games_played = data.get('games_played', 0)
        self.total_profit = data.get('total_profit', 0.0)
        self.win_streak = data.get('win_streak', 0)
        self.loss_streak = data.get('loss_streak', 0)
        self.bankroll = data.get('bankroll', self.config.get('initial_bankroll', 1000.0))
        self.result_history = data.get('result_history', [])
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save strategy state to JSON file.
        
        Args:
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(self.serialize(), f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load strategy state from JSON file.
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.deserialize(data)
    
    def __repr__(self) -> str:
        """String representation of strategy."""
        stats = self.get_statistics()
        return (f"{self.name}(games={stats['games_played']}, "
                f"profit={stats['total_profit']:.2f}, "
                f"win_rate={stats['win_rate']:.1f}%)")


class SimpleRandomStrategy(StrategyBase):
    """
    Simple random strategy for testing and baseline comparison.
    
    Makes random decisions for bet sizing and click positions.
    """
    
    def decide(self, state: Dict[str, Any], 
               uncertainty_vector: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
        """Make random decision."""
        if not state.get('game_active', False):
            # Place a bet
            bet = self.rng.uniform(self.base_bet, self.base_bet * 2)
            return {"action": "bet", "bet": bet}
        
        # In game - randomly click or cashout
        if self.rng.random() < 0.1:  # 10% chance to cashout
            return {"action": "cashout"}
        
        # Random click position
        board_size = state.get('board_size', 5)
        revealed = state.get('revealed', [])
        
        # Find unrevealed positions
        unrevealed = []
        for r in range(board_size):
            for c in range(board_size):
                if not revealed[r][c]:
                    unrevealed.append((r, c))
        
        if unrevealed:
            pos = self.rng.choice(unrevealed)
            return {"action": "click", "position": pos}
        
        return {"action": "cashout"}
    
    def on_result(self, result: Dict[str, Any]) -> None:
        """Update tracking."""
        self.update_tracking(result)


if __name__ == "__main__":
    # Test the interface
    strategy = SimpleRandomStrategy(config={'base_bet': 10.0, 'initial_bankroll': 1000.0})
    
    print(f"Created strategy: {strategy}")
    print(f"Statistics: {strategy.get_statistics()}")
    
    # Simulate a game result
    result = {
        'win': True,
        'payout': 2.12,
        'clicks': 8,
        'profit': 11.20,
        'final_bankroll': 1011.20,
        'bet_amount': 10.0
    }
    
    strategy.on_result(result)
    print(f"\nAfter 1 game: {strategy}")
    print(f"Statistics: {strategy.get_statistics()}")

