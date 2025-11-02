from typing import List, Tuple
import numpy as np

class BayesianMineModel:
    """
    Lightweight placeholder: returns (mean, var) for 'safe' probability per cell.
    Replace with pgmpy-based inference later; keep API stable.
    """
    def __init__(self, n_cells=25):
        self.n = n_cells

    def infer_safety(self, visible: np.ndarray) -> List[Tuple[float, float]]:
        # naive uniform prior over unopened cells
        unopened = (visible == 0)
        k = unopened.sum()
        if k == 0:
            return [(1.0, 0.0)] * self.n
        mean = 1.0 - (2 / max(1, k))  # pretend 2 mines spread over unopened
        var  = min(0.25, 0.5 / max(1, k))
        out = []
        for i in range(self.n):
            out.append((mean if unopened[i] else 1.0, var if unopened[i] else 0.0))
        return out
