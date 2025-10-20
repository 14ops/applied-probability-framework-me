import json
import os
import time
import random
import numpy as np
from datetime import datetime


# Placeholder for game_simulator and strategies if they are not directly importable
# In a real scenario, these would be proper imports from the project structure
class GameSimulator:
    def __init__(self, board_size=5, mine_count=5):
        self.board_size = board_size
        self.mine_count = mine_count
        self.board = []
        self.revealed = []
        self.mines = set()
        self.current_payout = 1.0
        self.bankroll = 1000.0
        self.initial_bet = 10.0
        self.game_over = False
        self.win = False
        self.cells_clicked = 0

    def initialize_game(self):
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

        # Place mines
        all_cells = [
            (r, c) for r in range(self.board_size) for c in range(self.board_size)
        ]
        self.mines = set(random.sample(all_cells, self.mine_count))

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
            # Simulate payout increase (simplified)
            self.current_payout *= 1.1  # Example payout increase
            # Check if all non-mine cells are revealed
            non_mine_cells_to_reveal = (
                self.board_size * self.board_size
            ) - self.mine_count
            if self.cells_clicked == non_mine_cells_to_reveal:
                self.game_over = True
                self.win = True
            return True, "Safe"

    def cash_out(self):
        if not self.game_over:
            self.bankroll += self.initial_bet * self.current_payout
            self.game_over = True
            self.win = True
            return True, "Cashed out"
        return False, "Game already over"


class BasicStrategy:
    def apply(self, simulator):
        # Simple strategy: click a random unrevealed cell
        unrevealed_cells = []
        for r in range(simulator.board_size):
            for c in range(simulator.board_size):
                if not simulator.revealed[r][c]:
                    unrevealed_cells.append((r, c))

        if unrevealed_cells:
            r, c = random.choice(unrevealed_cells)
            simulator.click_cell(r, c)
            # Decide to cash out randomly after a few clicks to simulate a win
            if (
                simulator.cells_clicked > 1 and random.random() < 0.3
            ):  # 30% chance to cash out after 1+ clicks
                simulator.cash_out()
        else:
            # If no unrevealed cells, and game is not over, it means all non-mines were clicked
            if not simulator.game_over:
                simulator.cash_out()


class DataLogger:
    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_session(self, session_data):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"session_log_{timestamp}.json")
        with open(log_file, "w") as f:
            json.dump(session_data, f, indent=4)
        print(f"Session logged to {log_file}")
        return log_file


class DataComparator:
    def __init__(self):
        pass

    def compare_distributions(self, real_data, simulated_data, metrics=None):
        if metrics is None:
            metrics = ["win_rate", "payout_distribution", "streak_lengths"]

        comparison_results = {}

        # Example: Compare win rates
        if "win_rate" in metrics:
            real_win_rate = np.mean([s["win"] for s in real_data if "win" in s])
            sim_win_rate = np.mean([s["win"] for s in simulated_data if "win" in s])
            comparison_results["win_rate_diff"] = real_win_rate - sim_win_rate
            win_rate_diff_val = comparison_results["win_rate_diff"]
            print(
                f"Win Rate - Real: {real_win_rate:.2%}, Simulated: {sim_win_rate:.2%}, Diff: {win_rate_diff_val:.2%}"
            )

        # Example: Compare average payouts
        if "payout_distribution" in metrics:
            real_payouts = [s["final_payout"] for s in real_data if "final_payout" in s]
            sim_payouts = [
                s["final_payout"] for s in simulated_data if "final_payout" in s
            ]
            if real_payouts and sim_payouts:
                real_avg_payout = np.mean(real_payouts)
                sim_avg_payout = np.mean(sim_payouts)
                comparison_results["avg_payout_diff"] = real_avg_payout - sim_avg_payout
                avg_payout_diff_val = comparison_results["avg_payout_diff"]
                print(
                    f"Avg Payout - Real: {real_avg_payout:.2f}, Simulated: {sim_avg_payout:.2f}, Diff: {avg_payout_diff_val:.2f}"
                )

        # Placeholder for more complex comparisons (e.g., streak lengths, latency, UI frame)
        # This would require more detailed data logging and statistical methods.

        return comparison_results


def run_data_collection_and_comparison(
    num_real_sessions=10, num_simulated_sessions=100
):
    logger = DataLogger()
    comparator = DataComparator()

    print("\n--- Simulating 'Real' Game Sessions ---")
    real_game_sessions = []
    for i in range(num_real_sessions):
        sim = GameSimulator()
        sim.initialize_game()
        strategy = BasicStrategy()  # Using a basic strategy for \'real\' play
        session_history = []
        while not sim.game_over:
            # Simulate human-like delays and clicks
            time.sleep(
                random.uniform(0.01, 0.05)
            )  # Reduced delay for faster simulation
            strategy.apply(sim)
            session_history.append(
                {
                    "board_state": sim.board,
                    "revealed": sim.revealed,
                    "mines": list(sim.mines),
                    "current_payout": sim.current_payout,
                    "game_over": sim.game_over,
                    "win": sim.win,
                }
            )

        final_payout = sim.current_payout if sim.win else 0.0
        real_game_sessions.append(
            {
                "session_id": f"real_game_{i}",
                "board_size": sim.board_size,
                "mine_count": sim.mine_count,
                "win": sim.win,
                "final_payout": final_payout,
                "history": session_history,
            }
        )
        print(f"'Real' Session {i+1}: Win={sim.win}, Payout={final_payout:.2f}")

    print("\n--- Running Simulated Sessions (for comparison) ---")
    simulated_sessions = []
    for i in range(num_simulated_sessions):
        sim = GameSimulator()
        sim.initialize_game()
        strategy = BasicStrategy()  # Using the same basic strategy for simulation
        session_history = []
        while not sim.game_over:
            strategy.apply(sim)
            session_history.append(
                {
                    "board_state": sim.board,
                    "revealed": sim.revealed,
                    "mines": list(sim.mines),
                    "current_payout": sim.current_payout,
                    "game_over": sim.game_over,
                    "win": sim.win,
                }
            )
        final_payout = sim.current_payout if sim.win else 0.0
        simulated_sessions.append(
            {
                "session_id": f"simulated_game_{i}",
                "board_size": sim.board_size,
                "mine_count": sim.mine_count,
                "win": sim.win,
                "final_payout": final_payout,
                "history": session_history,
            }
        )
        # print(f"Simulated Session {i+1}: Win={sim.win}, Payout={final_payout:.2f}") # Suppress for brevity

    print("\n--- Logging 'Real' Game Sessions ---")
    logged_real_sessions_file = logger.log_session(real_game_sessions)

    print("\n--- Comparing Real vs. Simulated Distributions ---")
    comparison_results = comparator.compare_distributions(
        real_game_sessions, simulated_sessions
    )
    print("Comparison Results:", comparison_results)

    # Placeholder for calibration: Adjust model parameters based on comparison_results
    # For instance, if sim_win_rate is consistently lower than real_win_rate,
    # you might adjust the simulator\\\'s internal probabilities or strategy parameters.
    print("\n--- Model Calibration (Placeholder) ---")
    print("Based on the comparison, model parameters would be adjusted here.")
    print(
        "For example, if simulated win rate is consistently off, adjust game_simulator parameters or strategy logic."
    )

    return comparison_results


if __name__ == "__main__":
    run_data_collection_and_comparison()
