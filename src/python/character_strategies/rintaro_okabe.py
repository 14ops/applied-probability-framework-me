"""
Rintaro Okabe - The Mad Scientist

Core Philosophy: "I am the mad scientist, Hououin Kyouma!"
Meta-game manipulation with worldline convergence and game theory.
"""

import random
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from core.adaptive_strategy import AdaptiveLearningStrategy


class RintaroOkabeStrategy(AdaptiveLearningStrategy):
    """
    The Mad Scientist - Worldline convergence with game-theoretic exploitation + AI Evolution.
    
    Key Features:
    - Game Theory Optimal (GTO) moves
    - Worldline convergence probability
    - Psychological pressure and bluffing
    - Lab Member Consensus System
    - Timeline manipulation (adaptive learning)
    - Q-Learning Matrix for temporal learning
    - Experience Replay across worldlines
    - Evolution of lab member personas
    """
    
    register = True
    plugin_name = "okabe"
    
    def __init__(self, name: str = "Rintaro Okabe - Mad Scientist", config: Optional[Dict[str, Any]] = None):
        # Define evolvable parameters (lab member personas)
        parameter_ranges = {
            'delta': (0.1, 0.5),  # GTO weight
            'epsilon': (0.1, 0.3),  # Psychological weight
            'worldline_weight': (0.3, 0.7),
            'mayuri_risk': (0.1, 0.3),
            'mayuri_aggression': (0.2, 0.4),
            'daru_risk': (0.6, 0.9),
            'daru_aggression': (0.7, 0.9),
            'kurisu_risk': (0.4, 0.6),
            'kurisu_aggression': (0.4, 0.6),
        }
        
        # Initialize adaptive learning with evolution
        super().__init__(
            name=name,
            config=config,
            use_q_learning=True,
            use_experience_replay=True,
            use_parameter_evolution=True,
            parameter_ranges=parameter_ranges
        )
        
        # Worldline parameters (evolvable)
        self.delta = config.get('delta', 0.3) if config else 0.3  # GTO weight
        self.epsilon = config.get('epsilon', 0.2) if config else 0.2  # Psychological weight
        self.worldline_weight = config.get('worldline_weight', 0.5) if config else 0.5
        
        # Lab Member personas (different strategic archetypes - evolvable)
        self.lab_members = {
            'mayuri': {
                'risk_tolerance': config.get('mayuri_risk', 0.2) if config else 0.2,
                'aggression': config.get('mayuri_aggression', 0.3) if config else 0.3
            },  # Conservative
            'daru': {
                'risk_tolerance': config.get('daru_risk', 0.7) if config else 0.7,
                'aggression': config.get('daru_aggression', 0.8) if config else 0.8
            },    # Aggressive
            'kurisu': {
                'risk_tolerance': config.get('kurisu_risk', 0.5) if config else 0.5,
                'aggression': config.get('kurisu_aggression', 0.5) if config else 0.5
            },  # Balanced/Logical
        }
        
        # Timeline tracking (now includes cross-worldline learning)
        self.worldline_history = []
        self.divergence_meter = 0.0
        
        # Q-learning weight (Reading Steiner allows blending learned and heuristic)
        self.q_learning_weight = config.get('q_learning_weight', 0.4) if config else 0.4
        
    def select_action_heuristic(self, state: Any, valid_actions: Any) -> Any:
        """
        Heuristic action selection (override of AdaptiveLearningStrategy).
        Uses worldline convergence and multi-persona consensus.
        
        Args:
            state: Current game state
            valid_actions: List of valid (row, col) tuples
            
        Returns:
            (row, col) tuple for selected action
        """
        if not valid_actions:
            return None
        
        # Detect opponent/system behavior
        opponent_behavior = self._detect_opponent_behavior(state)
        
        # Check for bluff opportunities
        if opponent_behavior.get('is_bluff', False):
            return self._exploit_bluff(state, valid_actions)
        
        # Check worldline convergence
        desired_worldline = self._calculate_desired_worldline(state, valid_actions)
        
        if desired_worldline['convergence_probability'] > 0.7:
            return desired_worldline['action']
        
        # Get Lab Member Consensus
        consensus_action = self._lab_member_consensus(state, valid_actions)
        
        if consensus_action:
            return consensus_action
        
        # Default to GTO play
        return self._game_theory_optimal_move(state, valid_actions)
    
    def _game_theory_optimal_move(self, state: Dict, valid_actions: List) -> Tuple[int, int]:
        """Calculate Game Theory Optimal move."""
        # GTO in Mines: balance between exploitation and exploration
        # Prioritize cells with highest EV but add randomness for unexploitability
        
        best_cell = None
        best_score = float('-inf')
        
        for cell in valid_actions:
            ev = self._calculate_ev(state, cell, valid_actions)
            
            # Add GTO randomness
            gto_score = ev + random.gauss(0, 0.1)
            
            if gto_score > best_score:
                best_score = gto_score
                best_cell = cell
        
        return best_cell if best_cell else valid_actions[0]
    
    def _calculate_worldline_convergence(self, state: Dict, cell: Tuple[int, int],
                                        valid_actions: List) -> float:
        """
        Calculate how much this action increases probability of desired future state.
        """
        # Desired future state: high profit, low risk remaining
        clicks_made = state.get('clicks_made', 0)
        board_size = state.get('board_size', 5)
        mine_count = state.get('mine_count', 3)
        
        total_cells = board_size * board_size
        safe_cells = total_cells - mine_count
        
        # How close are we to clearing the board?
        progress = clicks_made / safe_cells
        
        # This cell's contribution to convergence
        p_safe = self._calculate_safety_probability(state, cell, valid_actions)
        
        # High safety + good progress = high convergence
        convergence = p_safe * (1.0 + progress)
        
        return convergence
    
    def _calculate_desired_worldline(self, state: Dict, valid_actions: List) -> Dict:
        """
        Identify the action that leads to the most desirable worldline.
        """
        best_action = None
        best_convergence = 0.0
        
        for cell in valid_actions:
            convergence = self._calculate_worldline_convergence(state, cell, valid_actions)
            
            if convergence > best_convergence:
                best_convergence = convergence
                best_action = cell
        
        return {
            'action': best_action,
            'convergence_probability': best_convergence
        }
    
    def _detect_opponent_behavior(self, state: Dict) -> Dict:
        """
        Detect patterns in the game system's behavior.
        """
        # Analyze recent worldline history for patterns
        if len(self.worldline_history) < 5:
            return {'is_bluff': False}
        
        # Check for suspicious patterns (simplified)
        recent_outcomes = [w['outcome'] for w in self.worldline_history[-5:]]
        
        # If too many losses in a row, might be a "bluff" (unfavorable RNG)
        loss_count = sum(1 for outcome in recent_outcomes if outcome == 'loss')
        
        is_bluff = loss_count >= 4  # Suspiciously high loss rate
        
        return {'is_bluff': is_bluff}
    
    def _exploit_bluff(self, state: Dict, valid_actions: List) -> Tuple[int, int]:
        """
        Exploit detected bluff by making a counter-intuitive move.
        """
        # If system is "bluffing" with bad RNG, be more aggressive
        # Choose a high-risk, high-reward cell
        
        best_cell = None
        best_potential = 0.0
        
        for cell in valid_actions:
            ev = self._calculate_ev(state, cell, valid_actions)
            risk = self._calculate_risk(state, cell, valid_actions)
            
            # High risk, high reward
            potential = ev * risk  # Paradoxical: higher risk = higher potential
            
            if potential > best_potential:
                best_potential = potential
                best_cell = cell
        
        return best_cell if best_cell else valid_actions[0]
    
    def _lab_member_consensus(self, state: Dict, valid_actions: List) -> Optional[Tuple[int, int]]:
        """
        Simulate consensus among lab members (different strategic personas).
        """
        votes = {}
        
        for member_name, persona in self.lab_members.items():
            # Each member votes based on their persona
            vote = self._member_vote(state, valid_actions, persona)
            if vote:
                votes[vote] = votes.get(vote, 0) + 1
        
        if not votes:
            return None
        
        # Return cell with most votes
        consensus_cell = max(votes, key=votes.get)
        
        # Only return if there's strong consensus (at least 2 votes)
        if votes[consensus_cell] >= 2:
            return consensus_cell
        
        return None
    
    def _member_vote(self, state: Dict, valid_actions: List, persona: Dict) -> Optional[Tuple[int, int]]:
        """Individual lab member's vote based on their persona."""
        risk_tolerance = persona['risk_tolerance']
        aggression = persona['aggression']
        
        best_cell = None
        best_score = float('-inf')
        
        for cell in valid_actions:
            ev = self._calculate_ev(state, cell, valid_actions)
            risk = self._calculate_risk(state, cell, valid_actions)
            
            # Score based on persona
            if risk > risk_tolerance:
                continue  # Too risky for this persona
            
            score = ev * aggression
            
            if score > best_score:
                best_score = score
                best_cell = cell
        
        return best_cell
    
    def _calculate_ev(self, state: Dict, cell: Tuple[int, int], valid_actions: List) -> float:
        """Calculate expected value."""
        p_safe = self._calculate_safety_probability(state, cell, valid_actions)
        clicks_made = state.get('clicks_made', 0)
        
        payout = 1.0 + (clicks_made * 0.1)
        loss = 1.0
        
        return (p_safe * payout) - ((1 - p_safe) * loss)
    
    def _calculate_safety_probability(self, state: Dict, cell: Tuple[int, int],
                                     valid_actions: List) -> float:
        """Calculate probability that cell is safe."""
        mine_count = state.get('mine_count', 3)
        unrevealed_count = len(valid_actions)
        
        return 1.0 - (mine_count / max(unrevealed_count, 1))
    
    def _calculate_risk(self, state: Dict, cell: Tuple[int, int], valid_actions: List) -> float:
        """Calculate risk level."""
        return 1.0 - self._calculate_safety_probability(state, cell, valid_actions)
    
    def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """Update learning matrices, worldline history, and divergence meter."""
        # Call adaptive strategy update (handles Q-learning, replay, evolution)
        super().update(state, action, reward, next_state, done)
        
        # Record worldline outcome
        outcome = 'win' if reward > 0 else 'loss'
        self.worldline_history.append({
            'action': action,
            'outcome': outcome,
            'reward': reward
        })
        
        # Update divergence meter (how far from ideal worldline)
        if outcome == 'loss':
            self.divergence_meter += 0.1
        else:
            self.divergence_meter = max(0, self.divergence_meter - 0.05)
    
    def _apply_evolved_parameters(self, params: Dict[str, float]) -> None:
        """Apply evolved parameters to strategy and lab members."""
        super()._apply_evolved_parameters(params)
        
        # Update worldline parameters
        self.delta = params.get('delta', self.delta)
        self.epsilon = params.get('epsilon', self.epsilon)
        self.worldline_weight = params.get('worldline_weight', self.worldline_weight)
        
        # Update lab member personas
        self.lab_members['mayuri']['risk_tolerance'] = params.get('mayuri_risk', 
                                                                   self.lab_members['mayuri']['risk_tolerance'])
        self.lab_members['mayuri']['aggression'] = params.get('mayuri_aggression',
                                                               self.lab_members['mayuri']['aggression'])
        self.lab_members['daru']['risk_tolerance'] = params.get('daru_risk',
                                                                 self.lab_members['daru']['risk_tolerance'])
        self.lab_members['daru']['aggression'] = params.get('daru_aggression',
                                                             self.lab_members['daru']['aggression'])
        self.lab_members['kurisu']['risk_tolerance'] = params.get('kurisu_risk',
                                                                   self.lab_members['kurisu']['risk_tolerance'])
        self.lab_members['kurisu']['aggression'] = params.get('kurisu_aggression',
                                                               self.lab_members['kurisu']['aggression'])
    
    def reset(self) -> None:
        """Reset for new game - but keep worldline history for learning."""
        super().reset()
        # Keep worldline_history for cross-game learning
        # This is Okabe's "Reading Steiner" - memory across timelines



