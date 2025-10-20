import random
import numpy as np


class GameSimulator:
    def __init__(self, board_size=5, mine_count=3, seed=None):
        self.board_size = board_size
        self.mine_count = mine_count
        self.seed = seed
        self.win = False
        self.current_payout = 0.0
        self.mines = set()
        self.revealed = np.full((board_size, board_size), False)
        self.board = np.full((board_size, board_size), 0)  # 0 for empty, 1 for mine
        self.cells_clicked = 0
        self.max_clicks = (board_size * board_size) - mine_count
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def initialize_game(self):
        self.mines = set()
        self.revealed = np.full((self.board_size, self.board_size), False)
        self.board = np.full((self.board_size, self.board_size), 0)
        self.win = False
        self.current_payout = 0.0
        self.cells_clicked = 0

        # Place mines
        all_cells = [
            (r, c) for r in range(self.board_size) for c in range(self.board_size)
        ]
        self.mines = set(self.rng.sample(all_cells, self.mine_count))
        for r, c in self.mines:
            self.board[r][c] = 1

    def click_cell(self, r, c):
        if not (0 <= r < self.board_size and 0 <= c < self.board_size):
            return False, "Invalid coordinates"
        if self.revealed[r][c]:
            return False, "Cell already revealed"

        self.revealed[r][c] = True
        self.cells_clicked += 1

        if (r, c) in self.mines:
            self.win = False
            self.current_payout = 0.0
            return True, "Mine"
        else:
            # Simulate payout increase for each safe click (simplified)
            self.current_payout += 0.5  # Example: each safe click increases payout
            if self.cells_clicked >= self.max_clicks:
                self.win = True
                return True, "All safe cells revealed"
            return True, "Safe"

    def cash_out(self):
        if self.cells_clicked > 0 and not self.win:
            self.win = True  # Consider cashing out a 'win' if no mine hit
        return True, "Cashed out"

    def run_simulation(self, strategies):
        print("Running simulation...")
        for i in range(10):
            print(f"Turn {i+1}")
            for strategy in strategies:
                strategy.apply()
        print("Simulation finished.")
