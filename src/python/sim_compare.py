import json
import time
import os
from pathlib import Path
import random
import numpy as np
from scipy.stats import kstest

from game_simulator import GameSimulator
from strategies import BasicStrategy

# Assuming GameSimulator and strategy classes are available in the src directory
# For this example, we'll use a simplified GameSimulator and BasicStrategy


class GameSimulator:
    def __init__(self, board_size=5, mine_count=5, seed=None):
        self.board_size = board_size
        self.mine_count = mine_count
        self.board = []
        self.revealed = []
        self.mines = set()
        self.current_payout = 1.0
        self.game_over = False
        self.win = False
        self.cells_clicked = 0
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random

    def initialize_game(self):
        if self.seed is not None:
            self.rng.seed(self.seed)

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
        self.mines = set(self.rng.sample(all_cells, self.mine_count))

    def click_cell(self, r, c):
        if self.game_over or self.revealed[r][c]:
            return False, "Invalid click"

        self.revealed[r][c] = True
        self.cells_clicked += 1

        if (r, c) in self.mines:
            self.game_over = True
            self.win = False
            return True, "Mine"
        else:
            self.current_payout *= 1.1  # Simplified payout
            non_mine_cells_to_reveal = (
                self.board_size * self.board_size
            ) - self.mine_count
            if self.cells_clicked == non_mine_cells_to_reveal:
                self.game_over = True
                self.win = True
            return True, "Safe"

    def cash_out(self):
        if not self.game_over:
            self.game_over = True
            self.win = True
            return True, "Cashed out"
        return False, "Game already over"


class BasicStrategy:
    def __init__(self):
        self.name = "BasicStrategy"

    def apply(self, simulator):
        unrevealed_cells = []
        for r in range(simulator.board_size):
            for c in range(simulator.board_size):
                if not simulator.revealed[r][c]:
                    unrevealed_cells.append((r, c))

        if unrevealed_cells:
            r, c = simulator.rng.choice(unrevealed_cells)
            simulator.click_cell(r, c)
            if simulator.cells_clicked > 1 and simulator.rng.random() < 0.3:
                simulator.cash_out()
        else:
            if not simulator.game_over:
                simulator.cash_out()


# Logger (simplified for direct use)
LOG_FILE = Path("logs/sessions.jsonl")


def read_logged_sessions():
    sessions = []
    if not LOG_FILE.exists():
        return sessions
    with LOG_FILE.open("r") as f:
        for line in f:
            try:
                sessions.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON line: {line.strip()}")
    return sessions


def run_simulated_session(session_data):
    # Re-run simulation based on logged parameters
    board_size = session_data.get("grid_size", 5)
    mine_count = session_data.get("mine_count", 5)
    seed = session_data.get("seed", None)  # Assuming seed is logged for reproducibility
    strategy_name = session_data.get("strategy", "BasicStrategy")

    simulator = GameSimulator(board_size=board_size, mine_count=mine_count, seed=seed)
    simulator.initialize_game()

    # Instantiate the correct strategy based on name
    # In a real scenario, you'd have a factory or mapping for all strategies
    if strategy_name == "BasicStrategy":
        strategy = BasicStrategy()
    else:
        # Fallback or raise error for unknown strategies
        strategy = BasicStrategy()

    # Replay clicks or run strategy until game over
    # For simplicity, we'll just run the strategy once for a simulated outcome
    # A more detailed comparison would replay each click from the log
    strategy.apply(simulator)

    return {
        "win": simulator.win,
        "payout": simulator.current_payout if simulator.win else 0.0,
        "cells_clicked": simulator.cells_clicked,
    }


def compare_sim_to_live():
    logged_sessions = read_logged_sessions()
    if not logged_sessions:
        print("No logged sessions found to compare.")
        return

    live_payouts = []
    sim_payouts = []
    live_wins = []
    sim_wins = []

    print(f"Comparing {len(logged_sessions)} logged sessions...")

    for session in logged_sessions:
        # Extract actual live outcome
        live_wins.append(session.get("win", False))
        live_payouts.append(
            session.get("payout", 0.0) if session.get("win", False) else 0.0
        )

        # Run simulated outcome
        sim_result = run_simulated_session(session)
        sim_wins.append(sim_result["win"])
        sim_payouts.append(sim_result["payout"])

    # Convert to numpy arrays for easier calculation
    live_payouts_np = np.array(live_payouts)
    sim_payouts_np = np.array(sim_payouts)
    live_wins_np = np.array(live_wins)
    sim_wins_np = np.array(sim_wins)

    # Calculate differences
    mean_diff = np.mean(live_payouts_np) - np.mean(sim_payouts_np)
    var_diff = np.var(live_payouts_np) - np.var(sim_payouts_np)
    skew_diff = (
        np.mean((live_payouts_np - np.mean(live_payouts_np)) ** 3)
        / np.std(live_payouts_np) ** 3
        if np.std(live_payouts_np) > 0
        else 0
    ) - (
        np.mean((sim_payouts_np - np.mean(sim_payouts_np)) ** 3)
        / np.std(sim_payouts_np) ** 3
        if np.std(sim_payouts_np) > 0
        else 0
    )

    # Kolmogorov-Smirnov test for distribution similarity
    # Null hypothesis: two samples are drawn from the same continuous distribution
    ks_statistic, p_value = kstest(live_payouts_np, sim_payouts_np)

    print("\n--- Simulation vs. Live Comparison Results ---")
    print(f"Mean Payout (Live): {np.mean(live_payouts_np):.4f}")
    print(f"Mean Payout (Simulated): {np.mean(sim_payouts_np):.4f}")
    print(f"Difference in Means: {mean_diff:.4f}")
    print(f"Difference in Variances: {var_diff:.4f}")
    print(f"Difference in Skewness: {skew_diff:.4f}")
    print(f"Kolmogorov-Smirnov Test P-value: {p_value:.4f}")

    if p_value > 0.05:  # Common significance level
        print(
            "Conclusion: The simulated and live payout distributions are statistically similar (p > 0.05). Your simulator is a faithful echo of reality, like a perfect genjutsu!"
        )
    else:
        print(
            "Conclusion: The simulated and live payout distributions are statistically different (p <= 0.05). It seems there's a discrepancy between your simulation and reality. Time to recalibrate your chakra flow!"
        )

    # Also compare win rates
    live_win_rate = np.mean(live_wins_np)
    sim_win_rate = np.mean(sim_wins_np)
    print(f"\nLive Win Rate: {live_win_rate:.2%}")
    print(f"Simulated Win Rate: {sim_win_rate:.2%}")
    print(f"Difference in Win Rates: {(live_win_rate - sim_win_rate):.2%}")


if __name__ == "__main__":
    # Create a dummy log file for demonstration
    if not LOG_FILE.parent.exists():
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Generate some dummy session data for demonstration
    dummy_sessions = []
    for i in range(100):
        seed = i  # Use seed for reproducibility
        sim = GameSimulator(seed=seed)
        sim.initialize_game()
        strategy = BasicStrategy()
        strategy.apply(sim)
        dummy_sessions.append(
            {
                "ts": time.time(),
                "grid_size": sim.board_size,
                "mine_count": sim.mine_count,
                "seed": seed,
                "strategy": strategy.name,
                "win": sim.win,
                "payout": sim.current_payout if sim.win else 0.0,
                "cells_clicked": sim.cells_clicked,
            }
        )

    with LOG_FILE.open("w") as f:
        for session in dummy_sessions:
            f.write(json.dumps(session) + "\n")

    compare_sim_to_live()

    # Demonstrate a scenario where distributions might differ
    print("\n" + "=" * 50 + "\n")
    print(
        "Demonstrating a scenario with differing distributions (e.g., a bug in sim or strategy change)"
    )
    dummy_sessions_diff = []
    for i in range(100):
        seed = i + 100  # Different seed
        sim = GameSimulator(seed=seed)
        sim.initialize_game()
        strategy = BasicStrategy()
        # Introduce a bias in the 'live' data for demonstration
        sim.win = random.random() < 0.6  # Higher win rate in 'live' for this demo
        sim.current_payout = 1.5 if sim.win else 0.0
        sim.cells_clicked = 3 if sim.win else 1

        dummy_sessions_diff.append(
            {
                "ts": time.time(),
                "grid_size": sim.board_size,
                "mine_count": sim.mine_count,
                "seed": seed,
                "strategy": strategy.name,
                "win": sim.win,
                "payout": sim.current_payout if sim.win else 0.0,
                "cells_clicked": sim.cells_clicked,
            }
        )

    with LOG_FILE.open("w") as f:
        for session in dummy_sessions_diff:
            f.write(json.dumps(session) + "\n")

    compare_sim_to_live()
