"""
Optimized Game Simulator

High-performance Mines Game simulator with:
- Vectorized operations
- Memory-efficient state representation
- Cached calculations
- Fast random number generation
- Optimized data structures
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class GameState:
    """Optimized game state representation."""
    board: np.ndarray  # 0 = empty, 1 = mine
    revealed: np.ndarray  # Boolean array
    mines: np.ndarray  # Mine positions as (row, col) pairs
    clicks_made: int
    current_payout: float
    game_over: bool
    won: bool
    board_size: int
    mine_count: int


class OptimizedGameSimulator:
    """
    High-performance Mines Game simulator.
    
    Performance optimizations:
    - NumPy arrays for all data structures
    - Vectorized mine placement and checking
    - Cached valid actions
    - Fast random number generation
    - Memory-efficient state representation
    """
    
    def __init__(self, board_size: int = 5, mine_count: int = 3, seed: Optional[int] = None):
        self.board_size = board_size
        self.mine_count = mine_count
        self.seed = seed
        
        # Initialize random number generator
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        
        # Game state
        self.state = GameState(
            board=np.zeros((board_size, board_size), dtype=np.int8),
            revealed=np.zeros((board_size, board_size), dtype=bool),
            mines=np.zeros((mine_count, 2), dtype=np.int16),
            clicks_made=0,
            current_payout=1.0,
            game_over=False,
            won=False,
            board_size=board_size,
            mine_count=mine_count
        )
        
        # Cached values
        self._valid_actions_cache = None
        self._valid_actions_dirty = True
        self._total_cells = board_size * board_size
        self._safe_cells = self._total_cells - mine_count
        
        # Performance tracking
        self._operation_times = {}
    
    def initialize_game(self) -> None:
        """Initialize a new game with optimized mine placement."""
        start_time = time.perf_counter()
        
        # Reset state
        self.state.board.fill(0)
        self.state.revealed.fill(False)
        self.state.clicks_made = 0
        self.state.current_payout = 1.0
        self.state.game_over = False
        self.state.won = False
        
        # Clear caches
        self._valid_actions_cache = None
        self._valid_actions_dirty = True
        
        # Place mines using vectorized operations
        self._place_mines_vectorized()
        
        # Track performance
        self._operation_times['initialize'] = time.perf_counter() - start_time
    
    def _place_mines_vectorized(self) -> None:
        """Place mines using vectorized operations."""
        # Generate all possible positions
        all_positions = np.array([
            (r, c) for r in range(self.board_size) for c in range(self.board_size)
        ])
        
        # Randomly select mine positions
        mine_indices = self.rng.choice(len(all_positions), size=self.mine_count, replace=False)
        self.state.mines = all_positions[mine_indices]
        
        # Set mine positions on board
        for row, col in self.state.mines:
            self.state.board[row, col] = 1
    
    def click_cell(self, row: int, col: int) -> Tuple[bool, str]:
        """
        Click a cell with optimized checking.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            (success, result_message)
        """
        start_time = time.perf_counter()
        
        # Validate coordinates
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False, "Invalid coordinates"
        
        # Check if already revealed
        if self.state.revealed[row, col]:
            return False, "Cell already revealed"
        
        # Mark as revealed
        self.state.revealed[row, col] = True
        self.state.clicks_made += 1
        
        # Check if mine
        if self.state.board[row, col] == 1:
            self.state.game_over = True
            self.state.won = False
            self.state.current_payout = 0.0
            self._valid_actions_dirty = True
            self._operation_times['click_mine'] = time.perf_counter() - start_time
            return True, "Mine"
        
        # Safe cell - update payout
        self.state.current_payout += 0.5  # Simplified payout increase
        
        # Check win condition
        if self.state.clicks_made >= self._safe_cells:
            self.state.game_over = True
            self.state.won = True
            self._valid_actions_dirty = True
            self._operation_times['click_safe'] = time.perf_counter() - start_time
            return True, "All safe cells revealed"
        
        # Invalidate valid actions cache
        self._valid_actions_dirty = True
        self._operation_times['click_safe'] = time.perf_counter() - start_time
        return True, "Safe"
    
    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """Get valid actions with caching."""
        if not self._valid_actions_dirty and self._valid_actions_cache is not None:
            return self._valid_actions_cache
        
        # Calculate valid actions using vectorized operations
        unrevealed_mask = ~self.state.revealed
        valid_positions = np.argwhere(unrevealed_mask)
        self._valid_actions_cache = [(int(pos[0]), int(pos[1])) for pos in valid_positions]
        self._valid_actions_dirty = False
        
        return self._valid_actions_cache
    
    def get_state(self) -> Dict[str, Any]:
        """Get current game state as dictionary."""
        return {
            'board': self.state.board.copy(),
            'revealed': self.state.revealed.copy(),
            'mines': self.state.mines.copy(),
            'clicks_made': self.state.clicks_made,
            'current_payout': self.state.current_payout,
            'game_over': self.state.game_over,
            'won': self.state.won,
            'board_size': self.state.board_size,
            'mine_count': self.state.mine_count,
            'valid_actions': self.get_valid_actions()
        }
    
    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute an action and return new state.
        
        Args:
            action: (row, col) tuple
            
        Returns:
            (new_state, reward, done, info)
        """
        if action is None:
            # Cash out
            reward = self.state.current_payout if self.state.clicks_made > 0 else 0.0
            self.state.game_over = True
            self.state.won = True
            return self.get_state(), reward, True, {'cashed_out': True}
        
        row, col = action
        success, message = self.click_cell(row, col)
        
        if not success:
            return self.get_state(), 0.0, True, {'error': message}
        
        # Calculate reward
        if self.state.game_over:
            if self.state.won:
                reward = self.state.current_payout
            else:
                reward = 0.0
        else:
            reward = 0.0  # No immediate reward for safe clicks
        
        return self.get_state(), reward, self.state.game_over, {'message': message}
    
    def reset(self) -> Dict[str, Any]:
        """Reset the game and return initial state."""
        self.initialize_game()
        return self.get_state()
    
    def cash_out(self) -> Tuple[bool, str]:
        """Cash out with current payout."""
        if self.state.game_over:
            return False, "Game already over"
        
        self.state.game_over = True
        self.state.won = True
        self._valid_actions_dirty = True
        return True, f"Cashed out with {self.state.current_payout:.2f}x"
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get simulation metadata."""
        return {
            'board_size': self.board_size,
            'mine_count': self.mine_count,
            'clicks_made': self.state.clicks_made,
            'current_payout': self.state.current_payout,
            'won': self.state.won,
            'operation_times': self._operation_times.copy()
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        total_time = sum(self._operation_times.values())
        return {
            'total_operations_time': total_time,
            'operations_per_second': len(self._operation_times) / max(total_time, 1e-6),
            'avg_operation_time': np.mean(list(self._operation_times.values())) if self._operation_times else 0.0,
            'operation_times': self._operation_times.copy()
        }


class BatchGameSimulator:
    """
    Batch simulator for running multiple games in parallel.
    
    Optimized for running many simulations simultaneously.
    """
    
    def __init__(self, board_size: int = 5, mine_count: int = 3, batch_size: int = 1000):
        self.board_size = board_size
        self.mine_count = mine_count
        self.batch_size = batch_size
        
        # Pre-allocate arrays for batch operations
        self.total_cells = board_size * board_size
        self.safe_cells = self.total_cells - mine_count
        
    def simulate_batch(self, num_games: int, strategy_factory: callable) -> np.ndarray:
        """
        Simulate a batch of games.
        
        Args:
            num_games: Number of games to simulate
            strategy_factory: Factory function for creating strategies
            
        Returns:
            Array of rewards for each game
        """
        rewards = np.zeros(num_games)
        
        for i in range(num_games):
            # Create simulator and strategy
            sim = OptimizedGameSimulator(self.board_size, self.mine_count)
            strategy = strategy_factory()
            
            # Run simulation
            state = sim.reset()
            done = False
            
            while not done:
                valid_actions = sim.get_valid_actions()
                if not valid_actions:
                    break
                
                action = strategy.select_action(state, valid_actions)
                state, reward, done, _ = sim.step(action)
            
            rewards[i] = reward
        
        return rewards
    
    def simulate_batch_vectorized(self, num_games: int, strategy_factory: callable) -> np.ndarray:
        """
        Vectorized batch simulation (experimental).
        
        This is a simplified version - full vectorization would require
        more complex strategy implementations.
        """
        # For now, fall back to regular batch simulation
        return self.simulate_batch(num_games, strategy_factory)


def create_optimized_simulator_factory(board_size: int = 5, mine_count: int = 3):
    """Factory function for creating optimized simulators."""
    def factory(seed: Optional[int] = None):
        return OptimizedGameSimulator(board_size, mine_count, seed)
    return factory


def benchmark_simulator_performance(num_games: int = 1000) -> Dict[str, float]:
    """
    Benchmark simulator performance.
    
    Args:
        num_games: Number of games to simulate
        
    Returns:
        Performance metrics
    """
    from character_strategies.senku_ishigami import SenkuIshigamiStrategy
    
    # Create simulator and strategy
    sim = OptimizedGameSimulator()
    strategy = SenkuIshigamiStrategy()
    
    # Benchmark
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    rewards = []
    for _ in range(num_games):
        state = sim.reset()
        strategy.reset()
        done = False
        
        while not done:
            valid_actions = sim.get_valid_actions()
            if not valid_actions:
                break
            action = strategy.select_action(state, valid_actions)
            state, reward, done, _ = sim.step(action)
        
        rewards.append(reward)
    
    end_time = time.perf_counter()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Calculate metrics
    duration = end_time - start_time
    memory_used = end_memory - start_memory
    
    return {
        'total_duration': duration,
        'games_per_second': num_games / duration,
        'memory_used_mb': memory_used,
        'memory_per_game_mb': memory_used / num_games,
        'avg_reward': np.mean(rewards),
        'win_rate': np.mean([r > 0 for r in rewards])
    }
