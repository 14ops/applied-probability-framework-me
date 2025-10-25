"""
Auto-registration module for all built-in simulators and strategies.
This module should be imported at startup to populate the plugin registry.
"""

from core.plugin_system import register_simulator, register_strategy
from core.base_simulator import BaseSimulator
from core.base_strategy import BaseStrategy
from game_simulator import GameSimulator
from strategies import BasicStrategy
from bayesian import BetaEstimator
import random
import numpy as np


# Wrap GameSimulator to match BaseSimulator interface
@register_simulator("mines")
class MinesSimulator(BaseSimulator):
    """Mines game simulator wrapper for plugin system."""
    
    def __init__(self, config: dict = None):
        """
        Initialize Mines simulator.
        
        Args:
            config: Configuration dict with board_size, mine_count, etc.
        """
        if config is None:
            config = {}
        
        # Extract seed from config
        seed = config.get('seed', None)
        
        # Call parent with seed and config separately
        super().__init__(seed=seed, config=config)
        
        board_size = self.config.get('board_size', 5)
        mine_count = self.config.get('mine_count', 3)
        self.game = GameSimulator(board_size=board_size, mine_count=mine_count, seed=seed)
        self.game.initialize_game()
    
    def reset(self) -> dict:
        """Reset the game state."""
        self.game.initialize_game()
        return self.get_state()
    
    def step(self, action: tuple) -> tuple:
        """
        Execute an action in the simulator.
        
        Args:
            action: Tuple of (row, col) to click
            
        Returns:
            Tuple of (state, reward, done, info)
        """
        r, c = action
        success, result = self.game.click_cell(r, c)
        
        # Determine reward and done status
        if result == "Mine":
            reward = -1.0
            done = True
        elif self.game.cells_clicked >= self.game.max_clicks:
            reward = self.game.current_payout
            done = True
            self.game.win = True
        else:
            reward = 0.0
            done = False
        
        state = self.get_state()
        info = {
            'result': result,
            'cells_clicked': self.game.cells_clicked,
            'current_payout': self.game.current_payout
        }
        
        return state, reward, done, info
    
    def get_state(self) -> dict:
        """Get current game state."""
        return {
            'revealed': self.game.revealed.copy(),
            'board_size': self.game.board_size,
            'mine_count': self.game.mine_count,
            'cells_clicked': self.game.cells_clicked,
            'max_clicks': self.game.max_clicks,
            'current_payout': self.game.current_payout
        }
    
    def get_valid_actions(self) -> list:
        """Get list of valid actions (unrevealed cells)."""
        actions = []
        for r in range(self.game.board_size):
            for c in range(self.game.board_size):
                if not self.game.revealed[r][c]:
                    actions.append((r, c))
        return actions
    
    def is_terminal(self) -> bool:
        """Check if game is in terminal state."""
        return (self.game.cells_clicked >= self.game.max_clicks or 
                any((r, c) in self.game.mines for r in range(self.game.board_size) 
                    for c in range(self.game.board_size) if self.game.revealed[r][c]))
    
    def get_state_representation(self):
        """Get numerical representation of the state for ML models."""
        import numpy as np
        # Return the revealed board as a flat array
        return self.game.revealed.flatten().astype(float)


# Wrap BasicStrategy to match BaseStrategy interface
@register_strategy("basic")
@register_strategy("conservative")
class ConservativeStrategy(BaseStrategy):
    """Conservative strategy using Bayesian estimation."""
    
    def __init__(self, config: dict = None):
        """Initialize conservative strategy."""
        super().__init__(name="Conservative", config=config or {})
        self.estimator = BetaEstimator(a=1.0, b=1.0)
    
    def select_action(self, state: dict, valid_actions: list):
        """
        Select next action conservatively.
        
        Args:
            state: Current game state
            valid_actions: List of valid (row, col) actions
            
        Returns:
            Selected action tuple (row, col)
        """
        if not valid_actions:
            return None
        
        # Simple conservative strategy: pick randomly from valid actions
        # In a more sophisticated version, this would use probability estimates
        return random.choice(valid_actions)
    
    def update(self, state: dict, action: tuple, reward: float, next_state: dict, done: bool):
        """
        Update strategy based on outcome.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state
            done: Whether episode is done
        """
        # Update Bayesian estimator
        if done:
            if reward > 0:
                self.estimator.update(win=True)
            else:
                self.estimator.update(win=False)
    
    def get_info(self) -> dict:
        """Get strategy information."""
        return {
            'name': self.name,
            'estimated_win_prob': self.estimator.mean(),
            'confidence_interval': self.estimator.ci()
        }


# Register additional common strategies
@register_strategy("random")
class RandomStrategy(BaseStrategy):
    """Random action selection strategy."""
    
    def __init__(self, config: dict = None):
        """Initialize random strategy."""
        super().__init__(name="Random", config=config or {})
    
    def select_action(self, state: dict, valid_actions: list):
        """Select random action from valid actions."""
        if not valid_actions:
            return None
        return random.choice(valid_actions)
    
    def update(self, state: dict, action: tuple, reward: float, next_state: dict, done: bool):
        """Random strategy doesn't learn."""
        pass
    
    def get_info(self) -> dict:
        """Get strategy information."""
        return {'name': self.name}


@register_strategy("aggressive")
class AggressiveStrategy(BaseStrategy):
    """Aggressive strategy that takes more risks."""
    
    def __init__(self, config: dict = None):
        """Initialize aggressive strategy."""
        super().__init__(name="Aggressive", config=config or {})
        self.estimator = BetaEstimator(a=2.0, b=1.0)  # More optimistic prior
    
    def select_action(self, state: dict, valid_actions: list):
        """Select action aggressively."""
        if not valid_actions:
            return None
        # In real implementation, would prioritize high-risk/high-reward cells
        return random.choice(valid_actions)
    
    def update(self, state: dict, action: tuple, reward: float, next_state: dict, done: bool):
        """Update strategy based on outcome."""
        if done:
            if reward > 0:
                self.estimator.update(win=True)
            else:
                self.estimator.update(win=False)
    
    def get_info(self) -> dict:
        """Get strategy information."""
        return {
            'name': self.name,
            'estimated_win_prob': self.estimator.mean(),
            'confidence_interval': self.estimator.ci()
        }


# Character-based strategies from docs/guides/character_strategies.md

@register_strategy("takeshi")
@register_strategy("takeshi_kovacs")
class TakeshiStrategy(BaseStrategy):
    """Takeshi Kovacs - The Aggressive Berserker"""

    def __init__(self, config: dict = None):
        super().__init__(name="Takeshi Kovacs", config=config or {})
        self.aggression_threshold = self.config.get('aggression_threshold', 0.5)
        self.aggression_factor = self.config.get('aggression_factor', 1.5)
        self.move_history = []

    def select_action(self, state: dict, valid_actions: list):
        if not valid_actions:
            return None

        # Analyze the board to find the best moves
        board = state.get('board', [])
        revealed = state.get('revealed', [])
        board_size = state.get('board_size', 5)

        # Calculate mine probabilities for each cell
        cell_scores = []
        for action in valid_actions:
            row, col = action
            score = self._calculate_cell_score(row, col, board, revealed, board_size)
            cell_scores.append((action, score))

        # Sort by score (highest first for aggressive play)
        cell_scores.sort(key=lambda x: x[1], reverse=True)

        # Aggressive: take the best available move
        # But avoid repeating recent moves to avoid patterns
        for action, score in cell_scores:
            if action not in self.move_history[-3:]:  # Don't repeat last 3 moves
                self.move_history.append(action)
                return action

        # Fallback to best move if all recent moves are the same
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
                        # Revealed neighbor - check if it's safe or has mine info
                        if board[nr][nc] != -1:  # Not a mine
                            safe_neighbors += 1
                            # If it's a number, it's especially good (gives information)
                            if board[nr][nc] > 0:
                                score += 2

        # Calculate safety ratio
        if total_neighbors > 0:
            safety_ratio = safe_neighbors / total_neighbors
            score += safety_ratio * 3  # Base score from safety

        # Prefer edge/corner cells (less constrained)
        if row in [0, board_size-1] or col in [0, board_size-1]:
            score += 1
        if (row in [0, board_size-1]) and (col in [0, board_size-1]):
            score += 1  # Corner bonus

        return score

    def update(self, state: dict, action: tuple, reward: float, next_state: dict, done: bool):
        pass

    def reset(self):
        super().reset()
        self.move_history = []


@register_strategy("lelouch")
@register_strategy("lelouch_vi_britannia")
class LelouchStrategy(BaseStrategy):
    """Lelouch vi Britannia - The Strategic Mastermind"""

    def __init__(self, config: dict = None):
        super().__init__(name="Lelouch vi Britannia", config=config or {})
        self.move_history = []
        self.game_phase = 'early'
        self.strategic_targets = []

    def select_action(self, state: dict, valid_actions: list):
        if not valid_actions:
            return None

        board = state.get('board', [])
        revealed = state.get('revealed', [])
        board_size = state.get('board_size', 5)
        clicks_made = state.get('clicks_made', 0)

        # Determine game phase
        total_cells = board_size * board_size
        progress = clicks_made / total_cells if total_cells > 0 else 0
        if progress < 0.3:
            self.game_phase = 'early'
        elif progress < 0.7:
            self.game_phase = 'mid'
        else:
            self.game_phase = 'late'

        # Strategic analysis based on phase
        if self.game_phase == 'early':
            return self._early_game_strategy(board, revealed, valid_actions, board_size)
        elif self.game_phase == 'late':
            return self._late_game_strategy(board, revealed, valid_actions, board_size)
        else:
            return self._mid_game_strategy(board, revealed, valid_actions, board_size)

    def _early_game_strategy(self, board, revealed, valid_actions, board_size):
        """Early game: Focus on information gathering."""
        # Prefer corner cells first (maximum information potential)
        corners = [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]
        for corner in corners:
            if corner in valid_actions and corner not in self.move_history[-2:]:
                self.move_history.append(corner)
                return corner

        # Then edge cells
        edges = []
        for r in range(board_size):
            for c in range(board_size):
                if (r in [0, board_size-1] or c in [0, board_size-1]) and (r, c) in valid_actions:
                    edges.append((r, c))

        if edges and edges[0] not in self.move_history[-2:]:
            self.move_history.append(edges[0])
            return edges[0]

        return self._select_best_move(board, revealed, valid_actions, board_size)

    def _late_game_strategy(self, board, revealed, valid_actions, board_size):
        """Late game: Focus on probability and safety."""
        return self._select_safest_move(board, revealed, valid_actions, board_size)

    def _mid_game_strategy(self, board, revealed, valid_actions, board_size):
        """Mid game: Balance between information and safety."""
        return self._select_best_move(board, revealed, valid_actions, board_size)

    def _select_best_move(self, board, revealed, valid_actions, board_size):
        """Select move based on comprehensive analysis."""
        best_score = -1
        best_action = valid_actions[0]

        for action in valid_actions:
            if action in self.move_history[-2:]:
                continue  # Avoid recent moves

            score = self._calculate_strategic_score(action, board, revealed, board_size)
            if score > best_score:
                best_score = score
                best_action = action

        self.move_history.append(best_action)
        return best_action

    def _select_safest_move(self, board, revealed, valid_actions, board_size):
        """Select the safest possible move."""
        best_score = float('inf')
        best_action = valid_actions[0]

        for action in valid_actions:
            if action in self.move_history[-2:]:
                continue

            score = self._calculate_risk_score(action, board, revealed, board_size)
            if score < best_score:
                best_score = score
                best_action = action

        self.move_history.append(best_action)
        return best_action

    def _calculate_strategic_score(self, action, board, revealed, board_size):
        """Calculate strategic value of a move."""
        row, col = action
        score = 0

        # Information potential (revealed neighbors with numbers)
        info_potential = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size and revealed[nr][nc]:
                    if board[nr][nc] > 0:  # Number cell
                        info_potential += board[nr][nc]

        score += info_potential * 2

        # Position strategic value
        if (row in [0, board_size-1]) and (col in [0, board_size-1]):
            score += 3  # Corner
        elif row in [0, board_size-1] or col in [0, board_size-1]:
            score += 2  # Edge

        # Avoid clustering (strategic distribution)
        cluster_penalty = 0
        for past_move in self.move_history[-3:]:
            if abs(row - past_move[0]) <= 1 and abs(col - past_move[1]) <= 1:
                cluster_penalty += 1
        score -= cluster_penalty * 0.5

        return score

    def _calculate_risk_score(self, action, board, revealed, board_size):
        """Calculate risk level of a move."""
        row, col = action
        risk = 0

        # Check adjacent revealed cells for mine constraints
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size and revealed[nr][nc]:
                    if board[nr][nc] == 0:  # Empty cell
                        risk -= 0.5  # Lower risk
                    elif board[nr][nc] > 0:  # Number cell
                        # Calculate if this position is likely to have a mine
                        mines_around = board[nr][nc]
                        hidden_around = 0
                        for dr2 in [-1, 0, 1]:
                            for dc2 in [-1, 0, 1]:
                                nr2, nc2 = nr + dr2, nc + dc2
                                if 0 <= nr2 < board_size and 0 <= nc2 < board_size:
                                    if not revealed[nr2][nc2] and not (nr2 == row and nc2 == col):
                                        hidden_around += 1

                        if hidden_around > 0:
                            mine_probability = mines_around / hidden_around
                            risk += mine_probability * 2

        return risk

    def update(self, state: dict, action: tuple, reward: float, next_state: dict, done: bool):
        pass

    def reset(self):
        super().reset()
        self.move_history = []
        self.game_phase = 'early'
        self.strategic_targets = []


@register_strategy("kazuya")
@register_strategy("kazuya_kinoshita")
class KazuyaStrategy(BaseStrategy):
    """Kazuya Kinoshita - The Conservative Survivor"""

    def __init__(self, config: dict = None):
        super().__init__(name="Kazuya Kinoshita", config=config or {})
        self.max_tolerable_risk = self.config.get('max_risk', 0.20)
        self.clicks_made = 0
        self.safety_history = []

    def select_action(self, state: dict, valid_actions: list):
        if not valid_actions:
            return None

        board = state.get('board', [])
        revealed = state.get('revealed', [])
        board_size = state.get('board_size', 5)
        mines = state.get('mine_count', 3)

        # Calculate overall risk level
        remaining_cells = len(valid_actions)
        if remaining_cells > 0:
            overall_risk = mines / remaining_cells
        else:
            overall_risk = 1.0

        # If overall risk is too high, be extremely conservative
        if overall_risk > self.max_tolerable_risk * 1.5 and self.clicks_made > 1:
            # Look for the absolutely safest move possible
            return self._find_safest_move(board, revealed, valid_actions, board_size)

        # Normal conservative play
        return self._conservative_analysis(board, revealed, valid_actions, board_size, mines)

    def _find_safest_move(self, board, revealed, valid_actions, board_size):
        """Find the absolutely safest move based on neighbor analysis."""
        safest_score = float('inf')
        safest_move = valid_actions[0]

        for action in valid_actions:
            row, col = action
            risk_score = self._calculate_detailed_risk(row, col, board, revealed, board_size)

            if risk_score < safest_score:
                safest_score = risk_score
                safest_move = action

        self.safety_history.append(safest_move)
        return safest_move

    def _conservative_analysis(self, board, revealed, valid_actions, board_size, mines):
        """Conservative analysis prioritizing safety."""
        best_safety = -1
        best_move = valid_actions[0]

        for action in valid_actions:
            # Skip if we recently clicked near this area (avoid clustering)
            if any(abs(action[0] - past[0]) <= 1 and abs(action[1] - past[1]) <= 1
                   for past in self.safety_history[-2:]):
                continue

            safety_score = self._calculate_safety_score(action, board, revealed, board_size, mines)

            if safety_score > best_safety:
                best_safety = safety_score
                best_move = action

        self.safety_history.append(best_move)
        return best_move

    def _calculate_safety_score(self, action, board, revealed, board_size, mines):
        """Calculate how safe a move appears to be."""
        row, col = action
        safety_score = 0

        # Check all 8 neighbors for safety indicators
        safe_neighbors = 0
        dangerous_neighbors = 0
        info_neighbors = 0

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if revealed[nr][nc]:
                        if board[nr][nc] == -1:  # Mine neighbor!
                            dangerous_neighbors += 2  # Very dangerous
                        elif board[nr][nc] == 0:  # Empty neighbor
                            safe_neighbors += 1
                        else:  # Number neighbor (gives information)
                            info_neighbors += 1
                            # Check if this number suggests mines nearby
                            mines_marked = self._count_marked_mines_around(nr, nc, board, revealed, board_size)
                            if mines_marked < board[nr][nc]:
                                dangerous_neighbors += 1  # Likely mine nearby

        # Calculate safety ratio
        total_neighbor_info = safe_neighbors + dangerous_neighbors + info_neighbors
        if total_neighbor_info > 0:
            safety_ratio = (safe_neighbors + info_neighbors * 0.5) / total_neighbor_info
            safety_score = safety_ratio * 10  # Scale it up
        else:
            safety_score = 5  # Neutral score for no neighbor info

        # Prefer center positions (statistically safer)
        center_distance = abs(row - board_size//2) + abs(col - board_size//2)
        center_bonus = (board_size - center_distance) * 0.5
        safety_score += center_bonus

        # Penalty for being too close to recent moves
        recent_penalty = 0
        for past in self.safety_history[-3:]:
            distance = abs(row - past[0]) + abs(col - past[1])
            if distance <= 1:
                recent_penalty += 1
        safety_score -= recent_penalty * 2

        return safety_score

    def _calculate_detailed_risk(self, row, col, board, revealed, board_size):
        """Calculate detailed risk for a specific cell."""
        risk = 0

        # Check each neighbor for mine implications
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size and revealed[nr][nc]:
                    if board[nr][nc] > 0:  # Number cell
                        # Count hidden cells around this number
                        hidden_count = 0
                        for dr2 in [-1, 0, 1]:
                            for dc2 in [-1, 0, 1]:
                                nr2, nc2 = nr + dr2, nc + dc2
                                if 0 <= nr2 < board_size and 0 <= nc2 < board_size:
                                    if not revealed[nr2][nc2]:
                                        hidden_count += 1

                        if hidden_count > 0:
                            mine_probability = board[nr][nc] / hidden_count
                            # If this cell is one of the hidden cells, add risk
                            if mine_probability > 0.5:
                                risk += mine_probability * 3

        return risk

    def _count_marked_mines_around(self, row, col, board, revealed, board_size):
        """Count how many mines are already marked around a cell."""
        # This would need access to marked mines, but we don't have that info
        # So we'll estimate based on revealed mines
        mine_count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if revealed[nr][nc] and board[nr][nc] == -1:
                        mine_count += 1
        return mine_count

    def update(self, state: dict, action: tuple, reward: float, next_state: dict, done: bool):
        # Adjust risk tolerance based on performance
        if done and reward > 0:
            # Successful game - can be slightly less conservative
            self.max_tolerable_risk = min(0.35, self.max_tolerable_risk + 0.01)
        elif done and reward < 0:
            # Failed game - be more conservative
            self.max_tolerable_risk = max(0.05, self.max_tolerable_risk - 0.02)

    def reset(self):
        super().reset()
        self.clicks_made = 0
        self.safety_history = []


@register_strategy("senku")
@register_strategy("senku_ishigami")
class SenkuStrategy(BaseStrategy):
    """Senku Ishigami - The Analytical Scientist"""

    def __init__(self, config: dict = None):
        super().__init__(name="Senku Ishigami", config=config or {})
        self.probability_cache = {}
        self.information_theory = []

    def select_action(self, state: dict, valid_actions: list):
        if not valid_actions:
            return None

        board = state.get('board', [])
        revealed = state.get('revealed', [])
        board_size = state.get('board_size', 5)
        mines = state.get('mine_count', 3)

        # Clear cache for new analysis
        self.probability_cache = {}

        # Calculate comprehensive EV for each action
        action_analysis = []
        for action in valid_actions:
            ev_score = self._calculate_scientific_ev(action, board, revealed, board_size, mines)
            info_value = self._calculate_information_value(action, board, revealed, board_size)
            combined_score = ev_score + info_value * 0.3  # Weight information gain

            action_analysis.append((action, combined_score, ev_score, info_value))

        # Sort by combined score
        action_analysis.sort(key=lambda x: x[1], reverse=True)

        # Log analysis for learning
        best_action = action_analysis[0][0]
        self.information_theory.append({
            'best_ev': action_analysis[0][2],
            'best_info': action_analysis[0][3],
            'total_options': len(valid_actions)
        })

        return best_action

    def _calculate_scientific_ev(self, action, board, revealed, board_size, mines):
        """Calculate scientifically accurate expected value."""
        row, col = action

        # Use Bayesian inference based on revealed neighbors
        prior_mine_probability = mines / (board_size * board_size)

        # Update based on neighbor information
        neighbor_evidence = 0
        evidence_count = 0

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size and revealed[nr][nc]:
                    evidence_count += 1
                    if board[nr][nc] == -1:  # Mine neighbor
                        neighbor_evidence += 0.8  # Strong evidence of mine
                    elif board[nr][nc] == 0:  # Empty neighbor
                        neighbor_evidence -= 0.3  # Evidence against mine
                    else:  # Number neighbor
                        # Analyze the number constraint
                        constraint = self._analyze_number_constraint(nr, nc, board, revealed, board_size)
                        if constraint == 'mine_likely':
                            neighbor_evidence += 0.5
                        elif constraint == 'safe_likely':
                            neighbor_evidence -= 0.5

        # Bayesian update
        if evidence_count > 0:
            evidence_strength = neighbor_evidence / evidence_count
            # Simple Bayesian update: posterior = prior * likelihood
            likelihood = 1.0 + evidence_strength
            posterior_mine_prob = min(0.95, prior_mine_probability * likelihood)
        else:
            posterior_mine_prob = prior_mine_probability

        prob_safe = 1.0 - posterior_mine_prob

        # Calculate payout based on current game state
        total_cells = board_size * board_size
        remaining_safe = total_cells - mines - sum(row.count(-1) for row in board if -1 in row)
        if remaining_safe > 0:
            payout_multiplier = total_cells / remaining_safe
        else:
            payout_multiplier = 1.0

        # Apply house edge
        payout_multiplier *= 0.97

        # EV calculation
        ev = (prob_safe * payout_multiplier) - (posterior_mine_prob * 1.0)

        return ev

    def _analyze_number_constraint(self, row, col, board, revealed, board_size):
        """Analyze what a number cell tells us about mine placement."""
        number = board[row][col]
        if number <= 0:
            return 'neutral'

        # Count hidden cells around this number
        hidden_cells = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if not revealed[nr][nc]:
                        hidden_cells.append((nr, nc))

        hidden_count = len(hidden_cells)

        if hidden_count == 0:
            return 'satisfied'

        # Check if this number is already satisfied by marked mines
        # (We don't have marked mines, so estimate)
        revealed_mines_around = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if revealed[nr][nc] and board[nr][nc] == -1:
                        revealed_mines_around += 1

        if revealed_mines_around >= number:
            return 'satisfied'

        remaining_mines_needed = number - revealed_mines_around
        mine_probability = remaining_mines_needed / hidden_count if hidden_count > 0 else 0

        if mine_probability > 0.7:
            return 'mine_likely'
        elif mine_probability < 0.3:
            return 'safe_likely'
        else:
            return 'uncertain'

    def _calculate_information_value(self, action, board, revealed, board_size):
        """Calculate information value of revealing a cell."""
        row, col = action

        # Count unhidden neighbors
        unhidden_neighbors = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if not revealed[nr][nc]:
                        unhidden_neighbors += 1

        # More unhidden neighbors = more information potential
        info_value = unhidden_neighbors * 0.5

        # Edge/corner positions might give different information
        if (row in [0, board_size-1]) and (col in [0, board_size-1]):
            info_value += 0.5  # Corner might reveal more

        return info_value

    def update(self, state: dict, action: tuple, reward: float, next_state: dict, done: bool):
        # Senku learns from each outcome to improve probability estimates
        if done:
            # Analyze what we learned from this game
            board = state.get('board', [])
            revealed = state.get('revealed', [])

            # Count information gained
            total_cells = len(board) * len(board[0]) if board else 0
            revealed_cells = sum(row.count(True) for row in revealed) if revealed else 0
            information_ratio = revealed_cells / total_cells if total_cells > 0 else 0

            # Adjust future calculations based on success/failure
            if reward > 0:
                # Successful - our probability estimates were good
                self.probability_cache['confidence'] = self.probability_cache.get('confidence', 1.0) * 1.02
            else:
                # Failed - need to be more conservative
                self.probability_cache['confidence'] = self.probability_cache.get('confidence', 1.0) * 0.98

    def reset(self):
        super().reset()
        self.probability_cache = {}
        self.information_theory = []


@register_strategy("okabe")
@register_strategy("rintaro_okabe")
class RintaroOkabeStrategy(BaseStrategy):
    """Rintaro Okabe - The Mad Scientist"""

    def __init__(self, config: dict = None):
        super().__init__(name="Rintaro Okabe", config=config or {})
        self.worldline_history = []
        self.steiner_intuitions = 0
        self.consensus_weights = {'takeshi': 1.0, 'kazuya': 1.0, 'senku': 1.0}
        self.meta_learning = []

    def select_action(self, state: dict, valid_actions: list):
        if not valid_actions:
            return None

        board = state.get('board', [])
        revealed = state.get('revealed', [])
        board_size = state.get('board_size', 5)

        # Multi-dimensional analysis
        action_scores = []

        for action in valid_actions:
            # Game theory analysis
            game_theory_score = self._game_theory_analysis(action, board, revealed, board_size)

            # Worldline convergence analysis
            convergence_score = self._worldline_convergence(action, state)

            # Psychological pressure analysis
            pressure_score = self._psychological_analysis(action, state)

            # Lab member consensus (simulated)
            consensus_score = self._lab_consensus(action, valid_actions)

            # Meta-analysis: how well do our predictions match reality?
            meta_score = self._meta_analysis(action, state)

            # Combine all factors with adaptive weights
            total_score = (
                game_theory_score * 0.3 +
                convergence_score * 0.25 +
                pressure_score * 0.2 +
                consensus_score * 0.15 +
                meta_score * 0.1
            )

            action_scores.append((action, total_score, game_theory_score, convergence_score))

        # Sort by total score
        action_scores.sort(key=lambda x: x[1], reverse=True)

        # "Reading Steiner" - occasional intuition override
        if random.random() < 0.25:  # 25% chance of intuition
            self.steiner_intuitions += 1
            # Choose a move that might shift the "worldline"
            selected = action_scores[0][0]
            self.worldline_history.append(selected)
            return selected

        # Normal strategic selection
        best_action = action_scores[0][0]
        self.worldline_history.append(best_action)

        # Update meta learning
        self.meta_learning.append({
            'game_theory': action_scores[0][2],
            'convergence': action_scores[0][3],
            'worldline_length': len(self.worldline_history)
        })

        return best_action

    def _game_theory_analysis(self, action, board, revealed, board_size):
        """Game theory optimal move analysis."""
        row, col = action
        score = 0

        # Analyze local game state for optimal play
        safe_neighbors = 0
        risky_neighbors = 0

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if revealed[nr][nc]:
                        if board[nr][nc] == -1:  # Mine
                            risky_neighbors += 1
                        elif board[nr][nc] == 0:  # Safe
                            safe_neighbors += 1
                        else:  # Number - information
                            score += board[nr][nc] * 0.5

        # Risk-reward balance
        if safe_neighbors + risky_neighbors > 0:
            safety_ratio = safe_neighbors / (safe_neighbors + risky_neighbors)
            score += safety_ratio * 2

        # Prefer moves that maintain optionality
        edge_bonus = 0
        if row in [0, board_size-1] or col in [0, board_size-1]:
            edge_bonus = 1
        if (row in [0, board_size-1]) and (col in [0, board_size-1]):
            edge_bonus = 2  # Corner

        score += edge_bonus

        return score

    def _worldline_convergence(self, action, state):
        """Analyze how this move affects future game states."""
        row, col = action
        board_size = state.get('board_size', 5)
        clicks_made = state.get('clicks_made', 0)

        # Estimate future information gain
        future_potential = 0

        # Count unrevealed neighbors
        unrevealed_neighbors = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if not state.get('revealed', [[False]] * board_size)[nr][nc]:
                        unrevealed_neighbors += 1

        future_potential = unrevealed_neighbors * 0.7

        # Game phase consideration
        total_cells = board_size * board_size
        progress = clicks_made / total_cells if total_cells > 0 else 0

        if progress < 0.3:
            future_potential *= 1.2  # Early game - information is valuable
        elif progress > 0.7:
            future_potential *= 0.8  # Late game - safety is more important

        # Avoid moves that lead to obvious traps
        trap_penalty = self._detect_potential_traps(row, col, state)
        future_potential -= trap_penalty

        return future_potential

    def _psychological_analysis(self, action, state):
        """Psychological pressure and pattern analysis."""
        row, col = action

        # Check if this move breaks patterns
        pattern_score = 0

        # Avoid clustering (don't click too close to recent moves)
        recent_moves = self.worldline_history[-5:]
        clustering_penalty = 0
        for past_row, past_col in recent_moves:
            distance = abs(row - past_row) + abs(col - past_col)
            if distance <= 1:
                clustering_penalty += 1
        pattern_score -= clustering_penalty * 0.3

        # Prefer moves that create uncertainty
        board_size = state.get('board_size', 5)
        if (row in [0, board_size-1]) or (col in [0, board_size-1]):
            pattern_score += 0.5  # Edge moves are less predictable

        # Time-based variation (don't be too predictable)
        if len(self.worldline_history) % 3 == 0:
            pattern_score += 0.2  # Occasional "random" moves

        return pattern_score

    def _lab_consensus(self, action, valid_actions):
        """Simulate lab member consensus."""
        # Simple simulation of different strategies
        aggressive_moves = [valid_actions[i] for i in range(min(3, len(valid_actions)))]
        conservative_moves = [valid_actions[-i-1] for i in range(min(3, len(valid_actions)))]
        analytical_moves = valid_actions[len(valid_actions)//2:len(valid_actions)//2+3]

        # Weighted vote
        consensus = 0
        if action in aggressive_moves:
            consensus += self.consensus_weights['takeshi']
        if action in conservative_moves:
            consensus += self.consensus_weights['kazuya']
        if action in analytical_moves:
            consensus += self.consensus_weights['senku']

        return consensus

    def _meta_analysis(self, action, state):
        """Meta-analysis of our prediction accuracy."""
        # Simple meta-learning based on past performance
        if len(self.meta_learning) < 5:
            return 0  # Not enough data

        # Check if our recent predictions have been accurate
        recent_performance = sum(m['game_theory'] for m in self.meta_learning[-5:])
        avg_performance = recent_performance / 5

        # Adjust confidence based on past accuracy
        if avg_performance > 2:
            return 1.0  # High confidence
        elif avg_performance > 1:
            return 0.5  # Medium confidence
        else:
            return -0.5  # Low confidence - try different approach

    def _detect_potential_traps(self, row, col, state):
        """Detect potential trap situations."""
        trap_risk = 0
        board = state.get('board', [])
        revealed = state.get('revealed', [])
        board_size = state.get('board_size', 5)

        # Check for number cells that suggest mines nearby
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size and revealed[nr][nc]:
                    number = board[nr][nc]
                    if number > 0:
                        # Count hidden cells around this number
                        hidden_around = 0
                        for dr2 in [-1, 0, 1]:
                            for dc2 in [-1, 0, 1]:
                                nr2, nc2 = nr + dr2, nc + dc2
                                if 0 <= nr2 < board_size and 0 <= nc2 < board_size:
                                    if not revealed[nr2][nc2]:
                                        hidden_around += 1

                        if hidden_around > 0:
                            mines_per_hidden = number / hidden_around
                            if mines_per_hidden > 0.6:  # High probability of mine nearby
                                trap_risk += mines_per_hidden

        return trap_risk

    def update(self, state: dict, action: tuple, reward: float, next_state: dict, done: bool):
        # Update consensus weights based on performance
        if done:
            if reward > 0:
                # Success - reinforce strategies that contributed
                for move in [action]:
                    # This would need more sophisticated tracking
                    pass
            else:
                # Failure - adjust weights
                if len(self.worldline_history) > 3:
                    # Penalize recent strategy choices
                    pass

    def reset(self):
        super().reset()
        self.worldline_history = []
        self.steiner_intuitions = 0
        self.meta_learning = []


@register_strategy("hybrid")
class HybridStrategy(BaseStrategy):
    """Hybrid Strategy - The Ultimate Fusion"""

    def __init__(self, config: dict = None):
        super().__init__(name="Hybrid Strategy", config=config or {})
        self.senku = SenkuStrategy(config)
        self.lelouch = LelouchStrategy(config)
        self.game_phase = 'early'
        self.performance_history = []
        self.senku_weight = 0.5
        self.lelouch_weight = 0.5

    def select_action(self, state: dict, valid_actions: list):
        if not valid_actions:
            return None

        # Determine game phase with more sophisticated analysis
        cells_clicked = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        total_cells = board_size ** 2
        progress = cells_clicked / total_cells if total_cells > 0 else 0

        # Advanced phase determination based on multiple factors
        mine_count = state.get('mine_count', 3)
        remaining_cells = len(valid_actions)
        risk_level = mine_count / remaining_cells if remaining_cells > 0 else 1.0

        if progress < 0.25:
            self.game_phase = 'early'
        elif progress < 0.75:
            self.game_phase = 'mid'
        else:
            self.game_phase = 'late'

        # Get recommendations from both strategies
        senku_action = self.senku.select_action(state, valid_actions)
        lelouch_action = self.lelouch.select_action(state, valid_actions)

        if not senku_action or not lelouch_action:
            return senku_action or lelouch_action or valid_actions[0]

        # Advanced blending based on performance and situation
        blend_factor = self._calculate_blend_factor(state, risk_level)

        # If strategies agree, use their consensus
        if senku_action == lelouch_action:
            return senku_action

        # Otherwise, use weighted decision
        if random.random() < blend_factor:
            # Use Senku's analytical approach
            return senku_action
        else:
            # Use Lelouch's strategic approach
            return lelouch_action

    def _calculate_blend_factor(self, state, risk_level):
        """Calculate how much to weight Senku vs Lelouch based on current situation."""
        # Base weights from performance history
        recent_performance = self.performance_history[-10:] if self.performance_history else []
        if recent_performance:
            avg_reward = sum(p['reward'] for p in recent_performance) / len(recent_performance)
            if avg_reward > 0.5:
                self.senku_weight = min(0.8, self.senku_weight + 0.02)
            elif avg_reward < -0.5:
                self.lelouch_weight = min(0.8, self.lelouch_weight + 0.02)

        # Adjust based on game phase
        if self.game_phase == 'early':
            return max(0.7, self.senku_weight)  # Favor analysis early
        elif self.game_phase == 'late':
            return max(0.7, self.lelouch_weight)  # Favor strategy late
        else:
            return 0.5  # Balance in mid game

    def update(self, state: dict, action: tuple, reward: float, next_state: dict, done: bool):
        """Update both sub-strategies and track hybrid performance."""
        self.senku.update(state, action, reward, next_state, done)
        self.lelouch.update(state, action, reward, next_state, done)

        if done:
            self.performance_history.append({
                'game_phase': self.game_phase,
                'reward': reward,
                'senku_weight': self.senku_weight,
                'lelouch_weight': self.lelouch_weight
            })

            # Keep only recent history
            if len(self.performance_history) > 20:
                self.performance_history = self.performance_history[-20:]

    def reset(self):
        super().reset()
        self.senku.reset()
        self.lelouch.reset()
        self.performance_history = []


# This will execute when the module is imported
def register_all_plugins():
    """Register all built-in plugins. Called automatically on import."""
    # Plugins are registered via decorators, so just importing this module
    # will register everything
    pass


# Auto-register on import
register_all_plugins()

