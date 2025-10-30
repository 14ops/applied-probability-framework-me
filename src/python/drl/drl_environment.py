from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import numpy as np

@dataclass
class StepResult:
    next_state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]

class MinesEnv:
    """
    Minimal RL wrapper for 'Mines-like' simulation states the project already uses.
    State: [visible board encoding, remaining_mines, clicks_this_round, current_mult, bankroll_delta]
    Actions: 0..(N_cells-1) for clicking unrevealed cells, and action == N_cells for CASH_OUT.
    """
    def __init__(self, n_cells: int = 25):
        self.n_cells = n_cells
        self.reset()

    def encode_state(self) -> np.ndarray:
        # Basic placeholder: vector of length n_cells + 4 meta features.
        meta = np.array([self.remaining_mines, self.clicks, self.current_mult, self.bankroll_delta], dtype=np.float32)
        return np.concatenate([self.visible.astype(np.float32), meta])

    def reset(self) -> np.ndarray:
        self.visible = np.zeros(self.n_cells, dtype=np.int8)  # 0=unopened, 1=safe_open, -1=mine (not shown to agent)
        self.remaining_mines = 2
        self.clicks = 0
        self.current_mult = 1.0
        self.bankroll_delta = 0.0
        self.done = False
        return self.encode_state()

    def valid_actions(self) -> List[int]:
        unopened = [i for i,v in enumerate(self.visible) if v == 0]
        return unopened + [self.n_cells]  # last action = cash out

    def step(self, action: int) -> StepResult:
        if self.done:
            return StepResult(self.encode_state(), 0.0, True, {"err": "episode_done"})
        reward = 0.0
        info = {}
        if action == self.n_cells:
            # CASH OUT: pay current_mult - 1 as reward proxy
            reward = max(0.0, self.current_mult - 1.0)
            self.done = True
            return StepResult(self.encode_state(), reward, True, info)
        # Click cell
        self.clicks += 1
        # Placeholder: deterministic safe click for now; real project should plug true mine logic here
        self.visible[action] = 1
        self.current_mult *= 1.04  # simple rising mult; replace with real payout curve your project uses
        return StepResult(self.encode_state(), 0.01, False, info)
