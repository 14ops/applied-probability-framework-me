import random
import numpy as np
from scipy.stats import beta

"""Module for advanced probability estimation techniques, including Bayesian updating."""


class GameSimulator:
    """A simplified game simulator for demonstration purposes."""

    def __init__(self, board_size=5, mine_count=5):
        self.board_size = board_size
        self.mine_count = mine_count
        self.board = []
        self.revealed = []
        self.mines = set()
        self.current_payout = 1.0
        self.game_over = False
        self.win = False
        self.cells_clicked = 0

    def initialize_game(self):
        """Initializes a new game round."""
        self.board = [
            [0 for _ in range(self.board_size)] for _ in range(self.board_size)
        ]
        self.revealed = [
            [False for _ in range(self.board_size)] for _ in range(self.board_size)
        ]
        self.mines = set()
        self.current_payout = 1.0
        self.game_over = False
        self.win = False
        self.cells_clicked = 0

        all_cells = [
            (r, c) for r in range(self.board_size) for c in range(self.board_size)
        ]
        self.mines = set(random.sample(all_cells, self.mine_count))

    def click_cell(self, r, c):
        """Simulates clicking a cell on the board."""
        if self.game_over or self.revealed[r][c]:
            return False, "Invalid click"

        self.revealed[r][c] = True
        self.cells_clicked += 1

        if (r, c) in self.mines:
            self.game_over = True
            self.win = False
            return True, "Mine"

        self.current_payout *= 1.1
        non_mine_cells_to_reveal = (self.board_size * self.board_size) - self.mine_count
        if self.cells_clicked == non_mine_cells_to_reveal:
            self.game_over = True
            self.win = True
        return True, "Safe"

    def cash_out(self):
        """Simulates cashing out from the game."""
        if not self.game_over:
            self.game_over = True
            self.win = True
            return True, "Cashed out"
        return False, "Game already over"


class BayesianClickEstimator:
    """Estimates the probability of a safe click using Bayesian updating."""

    def __init__(self, a=1, b=1, decay_factor=1.0):
        self.a = a  # Prior for wins (safe clicks)
        self.b = b  # Prior for losses (mine clicks)
        self.decay_factor = decay_factor  # Factor for exponential decay

    def update(self, is_safe_click):
        """Updates the estimator with the result of a click."""
        # Apply exponential decay to old data before updating
        self.a *= self.decay_factor
        self.b *= self.decay_factor

        if is_safe_click:
            self.a += 1
        else:
            self.b += 1

    def mean(self):
        """Calculates the expected probability of a safe click."""
        return self.a / (self.a + self.b)

    def ci(self, alpha=0.05):
        """Calculates the (1-alpha) confidence interval for the Beta distribution."""
        lower = beta.ppf(alpha / 2, self.a, self.b)
        upper = beta.ppf(1 - alpha / 2, self.a, self.b)
        return lower, upper


if __name__ == "__main__":
    # Simulate a game where we try to estimate the probability of a safe click
    # Let\"s assume the true probability of a safe click is around 0.8 (20% chance of hitting a mine)
    _true_mine_prob = 0.2  # Renamed to _true_mine_prob to avoid Pylint warning
    _board_size = 5  # Renamed to _board_size to avoid Pylint warning
    _mine_count = int(_board_size * _board_size * _true_mine_prob)
    if _mine_count == 0:
        _mine_count = 1  # Ensure at least one mine

    estimator = BayesianClickEstimator(
        a=1, b=1, decay_factor=0.99
    )  # Start with a neutral prior, slight decay
    print(f"Initial estimate: Mean={estimator.mean():.2%}, CI={estimator.ci()}")

    _num_clicks = 100  # Renamed to _num_clicks to avoid Pylint warning
    for i in range(_num_clicks):
        # Simulate a click outcome based on true mine probability
        is_safe = random.random() > _true_mine_prob
        estimator.update(is_safe)

        if (i + 1) % 10 == 0:
            mean_prob = estimator.mean()
            lower_ci, upper_ci = estimator.ci()
            print(
                f"After {i+1} clicks: Mean={mean_prob:.2%}, CI=({lower_ci:.2%}, {upper_ci:.2%})"
            )

    print(
        "\nBayesian estimation complete. The estimator has adapted its belief about the safe click probability, much like a seasoned detective refining their suspects list with each new piece of evidence!"
    )

    # Demonstrate Importance Sampling (conceptual for now, full implementation is complex)
    print("\n--- Importance Sampling (Conceptual) ---")
    print(
        "Importance sampling would involve biasing simulations towards rare events (e.g., hitting a mine on the first click) and then re-weighting the results to get an unbiased estimate of their probability. This is like a specialized scout focusing on high-risk areas to find hidden dangers more efficiently."
    )

    # Demonstrate Variance Reduction Techniques (conceptual for now)
    print("\n--- Variance Reduction Techniques (Conceptual) ---")
    print(
        "Techniques like Antithetic Variates (running pairs of simulations with inverse random numbers) or Control Variates (using analytical approximations to reduce variance) would be applied here to make simulations more efficient and accurate. This is like optimizing your training regimen to get stronger results with less effort."
    )


