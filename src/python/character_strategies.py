"""
Character-based strategies for the Mines game.

Each strategy implements a unique decision-making philosophy based on
anime/game characters as documented in docs/guides/character_strategies.md
"""

import random
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from src.python.core.base_strategy import BaseStrategy


class TakeshiStrategy(BaseStrategy):
    """
    Takeshi Kovacs - The Aggressive Berserker
    
    High-risk, high-reward aggressive play that escalates betting
    when perceived advantage is high.
    """
    
    def __init__(self, name: str = "Takeshi Kovacs", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.aggression_threshold = self.config.get('aggression_threshold', 0.5)
        self.aggression_factor = self.config.get('aggression_factor', 1.5)
    
    def select_action(self, state: Any, valid_actions: Any) -> Any:
        """Select highest EV cell aggressively when advantage is high."""
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Calculate perceived advantage based on revealed cells vs remaining mines
        if hasattr(state, 'revealed_count') and hasattr(state, 'total_cells') and hasattr(state, 'num_mines'):
            revealed = state.revealed_count
            total = state.total_cells
            mines = state.num_mines
            remaining_cells = total - revealed
            if remaining_cells > 0:
                mine_density = (mines / remaining_cells)
                perceived_advantage = 1.0 - mine_density
            else:
                perceived_advantage = 0.0
        else:
            perceived_advantage = 0.5
        
        # If high advantage, pick the best cell aggressively
        if perceived_advantage > self.aggression_threshold:
            # Pick randomly from top candidates (simulating EV calculation)
            return random.choice(valid_actions[:min(3, len(valid_actions))])
        else:
            # Conservative pick
            return random.choice(valid_actions)


class LelouchStrategy(BaseStrategy):
    """
    Lelouch vi Britannia - The Strategic Mastermind
    
    Long-term strategic planning with psychological advantage and
    state-space exploration.
    """
    
    def __init__(self, name: str = "Lelouch vi Britannia", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.strategic_depth = self.config.get('strategic_depth', 3)
        self.move_history = []
    
    def select_action(self, state: Any, valid_actions: Any) -> Any:
        """Select action based on strategic impact and future states."""
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Avoid patterns - don't repeat recent moves
        available = [a for a in valid_actions if a not in self.move_history[-5:]]
        if not available:
            available = valid_actions
        
        # Prefer corner/edge cells for strategic information
        if hasattr(valid_actions[0], '__iter__') and len(valid_actions[0]) == 2:
            # Actions are (row, col) tuples
            corners_edges = [a for a in available if self._is_strategic_position(a, state)]
            if corners_edges:
                action = random.choice(corners_edges)
            else:
                action = random.choice(available)
        else:
            action = random.choice(available)
        
        self.move_history.append(action)
        return action
    
    def _is_strategic_position(self, action: Tuple[int, int], state: Any) -> bool:
        """Check if position is strategically valuable (corner or edge)."""
        if not hasattr(state, 'board_size'):
            return False
        
        row, col = action
        board_size = state.board_size
        
        # Corners and edges
        is_corner = (row in [0, board_size-1]) and (col in [0, board_size-1])
        is_edge = (row in [0, board_size-1]) or (col in [0, board_size-1])
        
        return is_corner or is_edge


class KazuyaStrategy(BaseStrategy):
    """
    Kazuya Kinoshita - The Conservative Survivor
    
    Extreme risk aversion with strict safety thresholds and
    capital preservation focus.
    """
    
    def __init__(self, name: str = "Kazuya Kinoshita", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.max_tolerable_risk = self.config.get('max_risk', 0.20)  # 20% max risk
        self.clicks_made = 0
    
    def select_action(self, state: Any, valid_actions: Any) -> Any:
        """Select the safest available cell."""
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Calculate risk level
        if hasattr(state, 'revealed_count') and hasattr(state, 'num_mines'):
            remaining_cells = len(valid_actions)
            if remaining_cells > 0:
                risk_per_cell = state.num_mines / remaining_cells
                
                # If risk is too high, suggest cashing out (return None or first action)
                if risk_per_cell > self.max_tolerable_risk and self.clicks_made > 2:
                    # In a real implementation, this would trigger cash out
                    # For now, pick the safest (but this is still risky)
                    pass
        
        self.clicks_made += 1
        
        # Always pick the "safest" cell (in practice, center cells are often safer)
        if hasattr(valid_actions[0], '__iter__') and len(valid_actions[0]) == 2:
            # Prefer center cells
            if hasattr(state, 'board_size'):
                center = state.board_size // 2
                center_cells = [a for a in valid_actions 
                               if abs(a[0] - center) <= 1 and abs(a[1] - center) <= 1]
                if center_cells:
                    return random.choice(center_cells)
        
        return random.choice(valid_actions)
    
    def reset(self) -> None:
        """Reset click counter for new game."""
        super().reset()
        self.clicks_made = 0


class SenkuStrategy(BaseStrategy):
    """
    Senku Ishigami - The Analytical Scientist
    
    Pure data-driven EV maximization with precise probability
    calculations and systematic exploration.
    """
    
    def __init__(self, name: str = "Senku Ishigami", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.probability_map = {}
    
    def select_action(self, state: Any, valid_actions: Any) -> Any:
        """Select action with highest expected value based on probabilities."""
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Calculate EV for each action
        action_evs = []
        for action in valid_actions:
            ev = self._calculate_expected_value(action, state)
            action_evs.append((action, ev))
        
        # Sort by EV and pick the highest
        action_evs.sort(key=lambda x: x[1], reverse=True)
        return action_evs[0][0]
    
    def _calculate_expected_value(self, action: Any, state: Any) -> float:
        """Calculate expected value for an action."""
        # Base probability calculation
        if hasattr(state, 'revealed_count') and hasattr(state, 'total_cells') and hasattr(state, 'num_mines'):
            remaining_cells = state.total_cells - state.revealed_count
            if remaining_cells > 0:
                prob_safe = 1.0 - (state.num_mines / remaining_cells)
            else:
                prob_safe = 0.0
        else:
            prob_safe = 0.5
        
        # Payout multiplier (simplified - would be based on actual game state)
        payout = 1.5
        
        # EV = (prob_safe * payout) - (prob_mine * loss)
        ev = (prob_safe * payout) - ((1 - prob_safe) * 1.0)
        
        return ev


class RintaroOkabeStrategy(BaseStrategy):
    """
    Rintaro Okabe - The Mad Scientist
    
    Meta-game manipulation with game theory, worldline convergence,
    and psychological pressure tactics.
    """
    
    def __init__(self, name: str = "Rintaro Okabe", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.worldline_history = []
        self.lab_member_votes = {}
    
    def select_action(self, state: Any, valid_actions: Any) -> Any:
        """Select action based on worldline convergence and game theory."""
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Simulate lab member consensus
        takeshi_vote = random.choice(valid_actions)  # Aggressive
        kazuya_vote = random.choice(valid_actions)  # Conservative
        senku_vote = random.choice(valid_actions)  # Analytical
        
        # Okabe's intuition: occasionally make "mad" choices
        if random.random() < 0.3:  # 30% chance of "Reading Steiner" intuition
            # Pick an unexpected action to shift worldlines
            return random.choice(valid_actions)
        
        # Otherwise, use a weighted consensus
        consensus_candidates = [takeshi_vote, kazuya_vote, senku_vote]
        action = random.choice(consensus_candidates)
        
        self.worldline_history.append(action)
        return action


class HybridStrategy(BaseStrategy):
    """
    Hybrid Strategy - The Ultimate Fusion
    
    Dynamic blending of Senku (analytical) and Lelouch (strategic)
    based on game phase and complexity.
    """
    
    def __init__(self, name: str = "Hybrid Strategy", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.senku = SenkuStrategy("Senku_Internal", config)
        self.lelouch = LelouchStrategy("Lelouch_Internal", config)
        self.game_phase = 'early'
    
    def select_action(self, state: Any, valid_actions: Any) -> Any:
        """Dynamically blend Senku and Lelouch strategies."""
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Determine game phase
        if hasattr(state, 'revealed_count') and hasattr(state, 'total_cells'):
            progress = state.revealed_count / state.total_cells
            if progress < 0.3:
                self.game_phase = 'early'
            elif progress < 0.7:
                self.game_phase = 'mid'
            else:
                self.game_phase = 'late'
        
        # Early game: use Senku (analytical)
        if self.game_phase == 'early':
            return self.senku.select_action(state, valid_actions)
        # Late game: use Lelouch (strategic)
        elif self.game_phase == 'late':
            return self.lelouch.select_action(state, valid_actions)
        # Mid game: blend both
        else:
            if random.random() < 0.5:
                return self.senku.select_action(state, valid_actions)
            else:
                return self.lelouch.select_action(state, valid_actions)

