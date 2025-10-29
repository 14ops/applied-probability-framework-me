"""
Deep Reinforcement Learning Strategy for Mines Game

This strategy uses a trained DQN agent to make decisions in the mines game.
It wraps the DQNAgent from the drl package to fit the StrategyBase interface.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import random
from .base import StrategyBase
from ..drl.drl_agent import DQNAgent
from ..drl.drl_environment import MinesEnv


class DRLStrategy(StrategyBase):
    """
    Deep Reinforcement Learning strategy using DQN.
    
    This strategy uses a pre-trained DQN agent to make decisions about
    when to click tiles and when to cash out. The agent learns optimal
    policies through reinforcement learning.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 rng: Optional[random.Random] = None):
        """
        Initialize DRL strategy.
        
        Args:
            config: Configuration dictionary
            rng: Random number generator
        """
        super().__init__(config, rng)
        
        # DRL-specific configuration
        self.n_cells = config.get('n_cells', 25)  # 5x5 board
        self.episodes_trained = config.get('episodes_trained', 2000)
        self.agent = None
        self.env = None
        
        # Initialize agent if not provided
        if 'agent' not in config:
            self._initialize_agent()
        else:
            self.agent = config['agent']
    
    def _initialize_agent(self):
        """Initialize the DQN agent and train it."""
        try:
            from ..drl.drl_training import train_dqn
            
            # Create environment
            self.env = MinesEnv(n_cells=self.n_cells)
            state_dim = self.env.encode_state().shape[0]
            action_dim = self.n_cells + 1  # cells + cashout
            
            # Train agent
            self.agent = train_dqn(
                episodes=self.episodes_trained,
                n_cells=self.n_cells
            )
            
        except ImportError as e:
            print(f"Warning: Could not import DRL modules: {e}")
            print("DRLStrategy will use random actions as fallback")
            self.agent = None
    
    def _state_to_env_format(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Convert game state to environment format.
        
        Args:
            state: Game state dictionary
            
        Returns:
            Environment state array
        """
        if self.env is None:
            self.env = MinesEnv(n_cells=self.n_cells)
        
        # Convert revealed tiles to environment format
        board_size = state.get('board_size', 5)
        revealed = state.get('revealed', [])
        
        # Create visible array (1=revealed safe, 0=unrevealed, -1=mine)
        visible = np.zeros(self.n_cells, dtype=np.int8)
        
        for r in range(board_size):
            for c in range(board_size):
                if r < len(revealed) and c < len(revealed[r]) and revealed[r][c]:
                    idx = r * board_size + c
                    if idx < self.n_cells:
                        visible[idx] = 1  # Safe revealed
        
        # Set environment state
        self.env.visible = visible
        self.env.remaining_mines = state.get('mine_count', 2)
        self.env.clicks = state.get('clicks_made', 0)
        self.env.current_mult = state.get('current_multiplier', 1.0)
        self.env.bankroll_delta = 0.0  # Not used in current env
        self.env.done = False
        
        return self.env.encode_state()
    
    def decide(self, state: Dict[str, Any], 
               uncertainty_vector: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Make decision using DQN agent.
        
        Args:
            state: Current game state
            
        Returns:
            Decision dictionary
        """
        if not state.get('game_active', False):
            # Place a bet
            bet = self.base_bet
            return {"action": "bet", "bet": bet}
        
        # If no agent available, use random fallback
        if self.agent is None:
            return self._random_fallback(state)
        
        try:
            # Convert state to environment format
            env_state = self._state_to_env_format(state)
            
            # Get valid actions
            valid_actions = self.env.valid_actions()
            valid_mask = [False] * (self.n_cells + 1)
            for action in valid_actions:
                valid_mask[action] = True
            
            # Get action from agent
            action = self.agent.act(env_state, valid_mask=valid_mask)
            
            if action == self.n_cells:
                # Cash out
                return {"action": "cashout"}
            else:
                # Click position
                board_size = state.get('board_size', 5)
                row = action // board_size
                col = action % board_size
                return {"action": "click", "position": (row, col)}
                
        except Exception as e:
            print(f"Warning: DRL agent error: {e}")
            return self._random_fallback(state)
    
    def _random_fallback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to random strategy if DRL fails."""
        if not state.get('game_active', False):
            return {"action": "bet", "bet": self.base_bet}
        
        # Random decision
        if self.rng.random() < 0.1:  # 10% chance to cashout
            return {"action": "cashout"}
        
        # Random click
        board_size = state.get('board_size', 5)
        revealed = state.get('revealed', [])
        
        unrevealed = []
        for r in range(board_size):
            for c in range(board_size):
                if r < len(revealed) and c < len(revealed[r]) and not revealed[r][c]:
                    unrevealed.append((r, c))
        
        if unrevealed:
            pos = self.rng.choice(unrevealed)
            return {"action": "click", "position": pos}
        
        return {"action": "cashout"}
    
    def on_result(self, result: Dict[str, Any]) -> None:
        """Update strategy state."""
        self.update_tracking(result)
        
        # Optional: Update agent with experience
        # This could be implemented to continue learning during play
        if self.agent is not None and hasattr(self.agent, 'buf'):
            # Convert result to experience and add to replay buffer
            # This is a simplified version - in practice you'd need more state info
            pass
