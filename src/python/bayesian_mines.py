"""
Bayesian Mines Module with pgmpy for Mine Probability Inference

This module implements sophisticated Bayesian inference for mine probability
estimation using probabilistic graphical models (pgmpy). It provides advanced
probabilistic reasoning for the Mines Game with constraint satisfaction and
uncertainty quantification.

Features:
- Probabilistic graphical models for mine inference
- Constraint satisfaction for revealed cell information
- Uncertainty quantification and confidence intervals
- Real-time probability updates
- Integration with DRL agent for risk assessment
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import logging
from collections import defaultdict, deque
import itertools
from datetime import datetime

try:
    import pgmpy
    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination, BeliefPropagation
    from pgmpy.factors.discrete import DiscreteFactor
    HAS_PGMPY = True
except ImportError:
    HAS_PGMPY = False
    logging.warning("pgmpy not available. Install with: pip install pgmpy")

from bayesian import BetaEstimator, VectorizedBetaEstimator


@dataclass
class MineConstraint:
    """Constraint from a revealed cell with number."""
    cell_position: Tuple[int, int]
    number: int
    adjacent_cells: List[Tuple[int, int]]
    satisfied: bool = False


@dataclass
class ProbabilityEstimate:
    """Probability estimate for a cell."""
    cell_position: Tuple[int, int]
    mine_probability: float
    confidence: float
    evidence_strength: float
    last_updated: datetime


class BayesianMinesInference:
    """
    Bayesian inference system for mine probability estimation.
    
    Uses probabilistic graphical models to infer mine probabilities
    based on revealed cell constraints and game state information.
    """
    
    def __init__(self, board_size: int = 5, mine_count: int = 3):
        self.board_size = board_size
        self.mine_count = mine_count
        self.total_cells = board_size * board_size
        self.safe_cells = self.total_cells - mine_count
        
        # Game state
        self.revealed_cells: Set[Tuple[int, int]] = set()
        self.mine_positions: Set[Tuple[int, int]] = set()
        self.constraints: List[MineConstraint] = []
        self.probability_estimates: Dict[Tuple[int, int], ProbabilityEstimate] = {}
        
        # Bayesian network components
        self.bayesian_network = None
        self.inference_engine = None
        self._build_bayesian_network()
        
        # Performance tracking
        self.inference_count = 0
        self.inference_times = deque(maxlen=1000)
        
        # Uncertainty quantification
        self.uncertainty_estimators = {
            'global': BetaEstimator(1.0, 1.0),
            'local': BetaEstimator(1.0, 1.0),
            'constraint': BetaEstimator(1.0, 1.0)
        }
    
    def _build_bayesian_network(self):
        """Build the Bayesian network for mine inference."""
        if not HAS_PGMPY:
            logging.warning("pgmpy not available, using simplified inference")
            return
        
        try:
            # Create Bayesian network
            self.bayesian_network = BayesianNetwork()
            
            # Add nodes for each cell
            for r in range(self.board_size):
                for c in range(self.board_size):
                    cell_name = f"cell_{r}_{c}"
                    self.bayesian_network.add_node(cell_name)
            
            # Add constraint nodes
            for i in range(self.total_cells):
                constraint_name = f"constraint_{i}"
                self.bayesian_network.add_node(constraint_name)
            
            # Add edges based on adjacency
            for r in range(self.board_size):
                for c in range(self.board_size):
                    cell_name = f"cell_{r}_{c}"
                    adjacent_cells = self._get_adjacent_cells(r, c)
                    
                    for adj_r, adj_c in adjacent_cells:
                        adj_cell_name = f"cell_{adj_r}_{adj_c}"
                        self.bayesian_network.add_edge(cell_name, adj_cell_name)
            
            # Create inference engine
            self.inference_engine = VariableElimination(self.bayesian_network)
            
        except Exception as e:
            logging.error(f"Failed to build Bayesian network: {e}")
            self.bayesian_network = None
            self.inference_engine = None
    
    def _get_adjacent_cells(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get adjacent cells for a given position."""
        adjacent = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    adjacent.append((nr, nc))
        return adjacent
    
    def update_game_state(self, 
                         revealed_cells: Set[Tuple[int, int]],
                         mine_positions: Set[Tuple[int, int]],
                         constraints: List[MineConstraint]):
        """Update the game state and recalculate probabilities."""
        self.revealed_cells = revealed_cells.copy()
        self.mine_positions = mine_positions.copy()
        self.constraints = constraints.copy()
        
        # Recalculate all probability estimates
        self._recalculate_probabilities()
    
    def _recalculate_probabilities(self):
        """Recalculate mine probabilities for all unrevealed cells."""
        start_time = datetime.now()
        
        # Get unrevealed cells
        unrevealed_cells = set()
        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r, c) not in self.revealed_cells and (r, c) not in self.mine_positions:
                    unrevealed_cells.add((r, c))
        
        # Calculate probabilities for each unrevealed cell
        for cell_pos in unrevealed_cells:
            probability = self._calculate_cell_probability(cell_pos)
            confidence = self._calculate_confidence(cell_pos)
            evidence_strength = self._calculate_evidence_strength(cell_pos)
            
            self.probability_estimates[cell_pos] = ProbabilityEstimate(
                cell_position=cell_pos,
                mine_probability=probability,
                confidence=confidence,
                evidence_strength=evidence_strength,
                last_updated=datetime.now()
            )
        
        # Track performance
        inference_time = (datetime.now() - start_time).total_seconds()
        self.inference_times.append(inference_time)
        self.inference_count += 1
    
    def _calculate_cell_probability(self, cell_pos: Tuple[int, int]) -> float:
        """Calculate mine probability for a specific cell."""
        if HAS_PGMPY and self.bayesian_network is not None:
            return self._bayesian_inference(cell_pos)
        else:
            return self._simplified_inference(cell_pos)
    
    def _bayesian_inference(self, cell_pos: Tuple[int, int]) -> float:
        """Use Bayesian network for probability inference."""
        try:
            # Prepare evidence
            evidence = {}
            
            # Add revealed cells as evidence
            for r, c in self.revealed_cells:
                cell_name = f"cell_{r}_{c}"
                evidence[cell_name] = 0  # 0 = no mine
            
            # Add mine positions as evidence
            for r, c in self.mine_positions:
                cell_name = f"cell_{r}_{c}"
                evidence[cell_name] = 1  # 1 = mine
            
            # Query probability
            cell_name = f"cell_{cell_pos[0]}_{cell_pos[1]}"
            query = self.inference_engine.query(variables=[cell_name], evidence=evidence)
            
            # Extract probability
            if cell_name in query.variables:
                prob_table = query.get_value(cell_name)
                if 1 in prob_table.values:
                    return float(prob_table.get_value({cell_name: 1}))
                else:
                    return 0.0
            else:
                return 0.5  # Default if query fails
                
        except Exception as e:
            logging.warning(f"Bayesian inference failed for {cell_pos}: {e}")
            return self._simplified_inference(cell_pos)
    
    def _simplified_inference(self, cell_pos: Tuple[int, int]) -> float:
        """Simplified probability inference without pgmpy."""
        # Count remaining mines
        remaining_mines = self.mine_count - len(self.mine_positions)
        
        # Count unrevealed cells
        unrevealed_count = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r, c) not in self.revealed_cells and (r, c) not in self.mine_positions:
                    unrevealed_count += 1
        
        if unrevealed_count == 0:
            return 0.0
        
        # Base probability
        base_prob = remaining_mines / unrevealed_count
        
        # Adjust based on constraints
        constraint_adjustment = self._calculate_constraint_adjustment(cell_pos)
        
        # Final probability
        final_prob = base_prob * constraint_adjustment
        
        return max(0.0, min(1.0, final_prob))
    
    def _calculate_constraint_adjustment(self, cell_pos: Tuple[int, int]) -> float:
        """Calculate constraint-based adjustment for cell probability."""
        adjustment = 1.0
        
        # Check constraints that affect this cell
        for constraint in self.constraints:
            if cell_pos in constraint.adjacent_cells:
                # Calculate how this constraint affects the cell
                constraint_effect = self._analyze_constraint_effect(cell_pos, constraint)
                adjustment *= constraint_effect
        
        return adjustment
    
    def _analyze_constraint_effect(self, 
                                  cell_pos: Tuple[int, int], 
                                  constraint: MineConstraint) -> float:
        """Analyze how a constraint affects a cell's mine probability."""
        if constraint.satisfied:
            return 1.0  # No adjustment if constraint is already satisfied
        
        # Count adjacent unrevealed cells
        unrevealed_adjacent = []
        for adj_pos in constraint.adjacent_cells:
            if (adj_pos not in self.revealed_cells and 
                adj_pos not in self.mine_positions):
                unrevealed_adjacent.append(adj_pos)
        
        if not unrevealed_adjacent:
            return 1.0
        
        # Calculate constraint satisfaction probability
        required_mines = constraint.number
        available_cells = len(unrevealed_adjacent)
        
        if available_cells < required_mines:
            return 0.0  # Impossible to satisfy
        
        # Calculate probability that this cell is a mine given the constraint
        if cell_pos in unrevealed_adjacent:
            # This cell could be one of the required mines
            prob_this_cell_mine = required_mines / available_cells
            return 1.0 + prob_this_cell_mine  # Boost probability
        else:
            return 1.0  # No direct effect
    
    def _calculate_confidence(self, cell_pos: Tuple[int, int]) -> float:
        """Calculate confidence in the probability estimate."""
        # Count evidence sources
        evidence_count = 0
        
        # Count adjacent revealed cells
        for adj_pos in self._get_adjacent_cells(cell_pos[0], cell_pos[1]):
            if adj_pos in self.revealed_cells:
                evidence_count += 1
        
        # Count constraints affecting this cell
        constraint_count = 0
        for constraint in self.constraints:
            if cell_pos in constraint.adjacent_cells:
                constraint_count += 1
        
        # Calculate confidence based on evidence
        total_evidence = evidence_count + constraint_count
        max_evidence = 8 + len(self.constraints)  # Max possible evidence
        
        confidence = min(1.0, total_evidence / max_evidence)
        
        # Update Bayesian estimator
        self.uncertainty_estimators['local'].update(
            wins=1 if confidence > 0.5 else 0,
            losses=1 if confidence <= 0.5 else 0
        )
        
        return confidence
    
    def _calculate_evidence_strength(self, cell_pos: Tuple[int, int]) -> float:
        """Calculate strength of evidence for this cell."""
        evidence_strength = 0.0
        
        # Check adjacent revealed cells
        for adj_pos in self._get_adjacent_cells(cell_pos[0], cell_pos[1]):
            if adj_pos in self.revealed_cells:
                # Find constraint for this revealed cell
                for constraint in self.constraints:
                    if constraint.cell_position == adj_pos:
                        # Calculate evidence strength based on constraint
                        unrevealed_adjacent = [
                            pos for pos in constraint.adjacent_cells
                            if pos not in self.revealed_cells and pos not in self.mine_positions
                        ]
                        
                        if unrevealed_adjacent:
                            # Evidence strength is inverse of uncertainty
                            uncertainty = len(unrevealed_adjacent) / constraint.number
                            evidence_strength += 1.0 / (1.0 + uncertainty)
        
        return min(1.0, evidence_strength)
    
    def get_cell_probability(self, cell_pos: Tuple[int, int]) -> float:
        """Get mine probability for a specific cell."""
        if cell_pos in self.probability_estimates:
            return self.probability_estimates[cell_pos].mine_probability
        else:
            return 0.5  # Default probability
    
    def get_cell_confidence(self, cell_pos: Tuple[int, int]) -> float:
        """Get confidence in probability estimate for a cell."""
        if cell_pos in self.probability_estimates:
            return self.probability_estimates[cell_pos].confidence
        else:
            return 0.0
    
    def get_safest_cells(self, num_cells: int = 5) -> List[Tuple[int, int]]:
        """Get the safest cells (lowest mine probability)."""
        if not self.probability_estimates:
            return []
        
        # Sort by mine probability (ascending)
        sorted_cells = sorted(
            self.probability_estimates.items(),
            key=lambda x: x[1].mine_probability
        )
        
        return [cell_pos for cell_pos, _ in sorted_cells[:num_cells]]
    
    def get_riskiest_cells(self, num_cells: int = 5) -> List[Tuple[int, int]]:
        """Get the riskiest cells (highest mine probability)."""
        if not self.probability_estimates:
            return []
        
        # Sort by mine probability (descending)
        sorted_cells = sorted(
            self.probability_estimates.items(),
            key=lambda x: x[1].mine_probability,
            reverse=True
        )
        
        return [cell_pos for cell_pos, _ in sorted_cells[:num_cells]]
    
    def get_uncertainty_map(self) -> np.ndarray:
        """Get uncertainty map for all cells."""
        uncertainty_map = np.zeros((self.board_size, self.board_size))
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r, c) in self.probability_estimates:
                    uncertainty_map[r, c] = 1.0 - self.probability_estimates[(r, c)].confidence
                elif (r, c) in self.revealed_cells:
                    uncertainty_map[r, c] = 0.0  # Known safe
                elif (r, c) in self.mine_positions:
                    uncertainty_map[r, c] = 0.0  # Known mine
                else:
                    uncertainty_map[r, c] = 1.0  # Unknown
        
        return uncertainty_map
    
    def get_probability_map(self) -> np.ndarray:
        """Get probability map for all cells."""
        prob_map = np.zeros((self.board_size, self.board_size))
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r, c) in self.probability_estimates:
                    prob_map[r, c] = self.probability_estimates[(r, c)].mine_probability
                elif (r, c) in self.mine_positions:
                    prob_map[r, c] = 1.0  # Known mine
                else:
                    prob_map[r, c] = 0.0  # Known safe or unknown
        
        return prob_map
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the inference system."""
        if not self.inference_times:
            return {}
        
        return {
            'total_inferences': self.inference_count,
            'avg_inference_time': np.mean(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'bayesian_network_available': self.bayesian_network is not None,
            'uncertainty_estimates': {
                name: estimator.mean() for name, estimator in self.uncertainty_estimators.items()
            }
        }
    
    def export_probability_data(self, filepath: str):
        """Export probability estimates to file."""
        import json
        
        data = {
            'board_size': self.board_size,
            'mine_count': self.mine_count,
            'revealed_cells': list(self.revealed_cells),
            'mine_positions': list(self.mine_positions),
            'probability_estimates': {
                f"{pos[0]}_{pos[1]}": {
                    'mine_probability': est.mine_probability,
                    'confidence': est.confidence,
                    'evidence_strength': est.evidence_strength,
                    'last_updated': est.last_updated.isoformat()
                }
                for pos, est in self.probability_estimates.items()
            },
            'constraints': [
                {
                    'cell_position': constraint.cell_position,
                    'number': constraint.number,
                    'adjacent_cells': constraint.adjacent_cells,
                    'satisfied': constraint.satisfied
                }
                for constraint in self.constraints
            ],
            'performance_stats': self.get_performance_stats(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logging.info(f"Probability data exported to {filepath}")


class ConstraintSatisfactionSolver:
    """
    Constraint satisfaction solver for mine constraints.
    
    Uses advanced constraint satisfaction techniques to solve
    mine placement problems and validate solutions.
    """
    
    def __init__(self, board_size: int = 5):
        self.board_size = board_size
        self.constraints: List[MineConstraint] = []
        self.solutions: List[Set[Tuple[int, int]]] = []
    
    def add_constraint(self, constraint: MineConstraint):
        """Add a constraint to the solver."""
        self.constraints.append(constraint)
    
    def solve_constraints(self, 
                         available_cells: Set[Tuple[int, int]],
                         mine_count: int) -> List[Set[Tuple[int, int]]]:
        """
        Solve mine placement constraints.
        
        Args:
            available_cells: Cells where mines can be placed
            mine_count: Number of mines to place
            
        Returns:
            List of valid mine placement solutions
        """
        solutions = []
        
        # Generate all possible mine combinations
        for mine_combination in itertools.combinations(available_cells, mine_count):
            mine_set = set(mine_combination)
            
            # Check if this combination satisfies all constraints
            if self._check_constraint_satisfaction(mine_set):
                solutions.append(mine_set)
        
        self.solutions = solutions
        return solutions
    
    def _check_constraint_satisfaction(self, mine_positions: Set[Tuple[int, int]]) -> bool:
        """Check if mine positions satisfy all constraints."""
        for constraint in self.constraints:
            if not self._check_single_constraint(mine_positions, constraint):
                return False
        return True
    
    def _check_single_constraint(self, 
                                mine_positions: Set[Tuple[int, int]], 
                                constraint: MineConstraint) -> bool:
        """Check if a single constraint is satisfied."""
        # Count mines adjacent to the constraint cell
        adjacent_mines = 0
        for adj_pos in constraint.adjacent_cells:
            if adj_pos in mine_positions:
                adjacent_mines += 1
        
        # Check if constraint is satisfied
        return adjacent_mines == constraint.number
    
    def get_constraint_satisfaction_probability(self, 
                                              cell_pos: Tuple[int, int],
                                              available_cells: Set[Tuple[int, int]],
                                              mine_count: int) -> float:
        """Calculate probability that a cell is a mine given constraints."""
        if not self.solutions:
            self.solve_constraints(available_cells, mine_count)
        
        if not self.solutions:
            return 0.5  # Default if no solutions
        
        # Count solutions where this cell is a mine
        mine_count = sum(1 for solution in self.solutions if cell_pos in solution)
        
        return mine_count / len(self.solutions)


def create_bayesian_mines_inference(board_size: int = 5, 
                                   mine_count: int = 3) -> BayesianMinesInference:
    """Factory function for creating Bayesian mines inference."""
    return BayesianMinesInference(board_size, mine_count)


def create_constraint_solver(board_size: int = 5) -> ConstraintSatisfactionSolver:
    """Factory function for creating constraint solver."""
    return ConstraintSatisfactionSolver(board_size)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create inference system
    inference = create_bayesian_mines_inference(board_size=5, mine_count=3)
    
    # Example game state
    revealed_cells = {(0, 0), (0, 1), (1, 0)}
    mine_positions = {(2, 2)}
    
    # Example constraints
    constraints = [
        MineConstraint(
            cell_position=(0, 0),
            number=1,
            adjacent_cells=[(0, 1), (1, 0), (1, 1)]
        ),
        MineConstraint(
            cell_position=(0, 1),
            number=2,
            adjacent_cells=[(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]
        )
    ]
    
    # Update game state
    inference.update_game_state(revealed_cells, mine_positions, constraints)
    
    # Get probability estimates
    print("Probability estimates:")
    for pos, est in inference.probability_estimates.items():
        print(f"Cell {pos}: P(mine) = {est.mine_probability:.3f}, "
              f"confidence = {est.confidence:.3f}")
    
    # Get safest cells
    safest = inference.get_safest_cells(3)
    print(f"Safest cells: {safest}")
    
    # Export data
    inference.export_probability_data("bayesian_mines_data.json")
    
    # Performance stats
    stats = inference.get_performance_stats()
    print(f"Performance stats: {stats}")