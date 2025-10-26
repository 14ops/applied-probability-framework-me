"""
Adaptive Takeshi Strategy - Q-Learning Enabled Version
This is an example of how to enable Q-learning for character strategies.
"""

from core.adaptive_strategy import AdaptiveLearningStrategy
from core.plugin_system import register_strategy
import random


@register_strategy("adaptive_takeshi")
@register_strategy("takeshi_learning")
class AdaptiveTakeshiStrategy(AdaptiveLearningStrategy):
    """
    Takeshi Kovacs strategy with Q-Learning enabled.
    
    Features:
    - Q-Learning matrix for state-action values
    - Experience replay buffer
    - Parameter evolution (optional)
    - Saves/loads learning progress
    """
    
    def __init__(self, config: dict = None):
        """Initialize adaptive Takeshi strategy."""
        if config is None:
            config = {}
        
        # Define evolvable parameters (optional)
        parameter_ranges = {
            'aggression_threshold': (0.3, 0.7),
            'aggression_factor': (1.0, 2.0),
        }
        
        # Initialize with Q-learning enabled
        super().__init__(
            name="Adaptive Takeshi Kovacs",
            config=config,
            use_q_learning=True,  # Enable Q-learning!
            use_experience_replay=True,  # Enable experience replay!
            use_parameter_evolution=False,  # Optional: enable parameter evolution
            parameter_ranges=parameter_ranges
        )
        
        # Takeshi-specific parameters
        self.aggression_threshold = config.get('aggression_threshold', 0.5)
        self.aggression_factor = config.get('aggression_factor', 1.5)
        self.move_history = []
    
    def select_action_heuristic(self, state: dict, valid_actions: list):
        """
        Heuristic action selection (used when not relying on Q-learning).
        
        This is the original Takeshi strategy logic.
        """
        if not valid_actions:
            return None
        
        # Analyze the board
        board = state.get('board', [])
        revealed = state.get('revealed', [])
        board_size = state.get('board_size', 5)
        
        # Calculate scores for each action
        cell_scores = []
        for action in valid_actions:
            row, col = action
            score = self._calculate_cell_score(row, col, board, revealed, board_size)
            cell_scores.append((action, score))
        
        # Sort by score (highest first for aggressive play)
        cell_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Aggressive: take the best available move
        for action, score in cell_scores:
            if action not in self.move_history[-3:]:  # Don't repeat last 3 moves
                self.move_history.append(action)
                return action
        
        # Fallback to best move
        return cell_scores[0][0]
    
    def _calculate_cell_score(self, row, col, board, revealed, board_size):
        """Calculate a score for a cell based on surrounding revealed cells."""
        score = 0
        safe_neighbors = 0
        total_neighbors = 0
        
        # Check all 8 neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    total_neighbors += 1
                    if revealed[nr][nc]:
                        if board[nr][nc] != -1:  # Not a mine
                            safe_neighbors += 1
                            if board[nr][nc] > 0:
                                score += 2
        
        # Calculate safety ratio
        if total_neighbors > 0:
            safety_ratio = safe_neighbors / total_neighbors
            score += safety_ratio * 3
        
        # Prefer edge/corner cells
        if row in [0, board_size-1] or col in [0, board_size-1]:
            score += 1
        if (row in [0, board_size-1]) and (col in [0, board_size-1]):
            score += 1
        
        return score
    
    def reset(self):
        """Reset episode-specific state."""
        super().reset()
        self.move_history = []


# Optional: Create other adaptive versions
@register_strategy("adaptive_senku")
@register_strategy("senku_learning")
class AdaptiveSenkuStrategy(AdaptiveLearningStrategy):
    """Senku Ishigami strategy with Q-Learning enabled."""
    
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        
        super().__init__(
            name="Adaptive Senku Ishigami",
            config=config,
            use_q_learning=True,
            use_experience_replay=True,
            use_parameter_evolution=False
        )
    
    def select_action_heuristic(self, state: dict, valid_actions: list):
        """Pure EV-based selection (Senku's analytical approach)."""
        if not valid_actions:
            return None
        
        # Calculate expected value for each action
        # For now, random selection (can be enhanced)
        return random.choice(valid_actions)


if __name__ == "__main__":
    print("Adaptive strategies with Q-learning registered!")
    print("Available strategies:")
    print("  - adaptive_takeshi")
    print("  - takeshi_learning")
    print("  - adaptive_senku")
    print("  - senku_learning")

