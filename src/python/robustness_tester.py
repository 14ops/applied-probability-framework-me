import json
import os
import random
import numpy as np
from datetime import datetime

# Assuming GameSimulator and BasicStrategy are available or defined as in data_logger_and_comparator.py
# For this script, we'll use the simplified versions from data_logger_and_comparator.py


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
            self.current_payout *= 1.1
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
        unrevealed_cells = []
        for r in range(simulator.board_size):
            for c in range(simulator.board_size):
                if not simulator.revealed[r][c]:
                    unrevealed_cells.append((r, c))

        if unrevealed_cells:
            r, c = random.choice(unrevealed_cells)
            simulator.click_cell(r, c)
            if simulator.cells_clicked > 1 and random.random() < 0.3:
                simulator.cash_out()
        else:
            if not simulator.game_over:
                simulator.cash_out()


class RobustnessTester:
    def __init__(self, simulator_class=GameSimulator, strategy_class=BasicStrategy):
        self.simulator_class = simulator_class
        self.strategy_class = strategy_class

    def run_k_fold_backtesting(self, historical_sessions, k_folds=5):
        print(f"\n--- Running {k_folds}-Fold Backtesting ---")
        session_indices = list(range(len(historical_sessions)))
        random.shuffle(session_indices)
        fold_size = len(session_indices) // k_folds

        fold_results = []

        for i in range(k_folds):
            test_indices = session_indices[i * fold_size : (i + 1) * fold_size]
            train_indices = [idx for idx in session_indices if idx not in test_indices]

            test_set = [historical_sessions[idx] for idx in test_indices]
            train_set = [historical_sessions[idx] for idx in train_indices]

            # In a real scenario, strategy parameters would be optimized on train_set here
            # For this basic example, we just run the strategy on the test set
            print(
                f"  Fold {i+1}/{k_folds}: Testing on {len(test_set)} sessions, Training on {len(train_set)} sessions (placeholder)"
            )

            fold_win_rates = []
            fold_payouts = []

            for session_data in test_set:
                # Replay or simulate based on session_data
                # For simplicity, we'll just use the recorded win/payout from historical_sessions
                fold_win_rates.append(1 if session_data["win"] else 0)
                fold_payouts.append(session_data["final_payout"])

            if fold_win_rates:
                fold_results.append(
                    {
                        "win_rate": np.mean(fold_win_rates),
                        "avg_payout": np.mean(fold_payouts),
                    }
                )

        overall_win_rate = np.mean([f["win_rate"] for f in fold_results])
        overall_avg_payout = np.mean([f["avg_payout"] for f in fold_results])

        print(f"Overall Backtesting Results (across {k_folds} folds):")
        print(f"  Average Win Rate: {overall_win_rate:.2%}")
        print(f"  Average Payout: {overall_avg_payout:.2f}")
        return fold_results

    def run_stress_test_long_losing_streak(
        self, num_sessions=10, initial_bankroll=1000, stop_loss_threshold=0.5
    ):
        print(f"\n--- Running Stress Test: Long Losing Streak ---")
        strategy = self.strategy_class()
        bankroll = initial_bankroll
        max_drawdown = 0
        peak_bankroll = initial_bankroll
        time_to_ruin = -1
        ruined = False

        session_results = []

        for i in range(num_sessions):
            sim = self.simulator_class()
            sim.bankroll = bankroll  # Carry over bankroll
            sim.initialize_game()

            # Force a loss for this session
            sim.mines = set(
                random.sample(
                    [
                        (r, c)
                        for r in range(sim.board_size)
                        for c in range(sim.board_size)
                    ],
                    sim.mine_count,
                )
            )
            # Ensure the first click is a mine for a forced loss scenario (simplified)
            first_click_cell = random.choice(list(sim.mines))
            sim.click_cell(first_click_cell[0], first_click_cell[1])
            sim.win = False  # Explicitly set to loss
            sim.game_over = True
            sim.current_payout = 0.0

            bankroll -= sim.initial_bet  # Deduct initial bet for a loss
            session_results.append(
                {
                    "win": sim.win,
                    "final_payout": sim.current_payout,
                    "bankroll_after": bankroll,
                }
            )

            peak_bankroll = max(peak_bankroll, bankroll)
            current_drawdown = (peak_bankroll - bankroll) / peak_bankroll
            max_drawdown = max(max_drawdown, current_drawdown)

            if bankroll <= initial_bankroll * (1 - stop_loss_threshold):
                ruined = True
                time_to_ruin = i + 1
                break
            print(f"  Session {i+1}: Forced Loss, Bankroll: {bankroll:.2f}")

        print(f"Stress Test Results (Long Losing Streak):")
        print(f"  Final Bankroll: {bankroll:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2%}")
        if ruined:
            print(f"  Bankroll Ruined! Time to Ruin: {time_to_ruin} sessions")
        else:
            print(f"  Bankroll survived {num_sessions} sessions.")
        return {
            "final_bankroll": bankroll,
            "max_drawdown": max_drawdown,
            "time_to_ruin": time_to_ruin,
        }

    def run_stress_test_variable_house_edge(
        self, num_sessions=100, initial_bankroll=1000
    ):
        print(f"\n--- Running Stress Test: Variable House Edge ---")
        strategy = self.strategy_class()
        bankroll = initial_bankroll
        session_results = []

        for i in range(num_sessions):
            sim = self.simulator_class()
            sim.bankroll = bankroll

            # Introduce variable house edge by changing mine count randomly
            if (
                random.random() < 0.2
            ):  # 20% chance of higher mine count (worse house edge)
                sim.mine_count = min(
                    sim.board_size * sim.board_size - 1,
                    sim.mine_count + random.randint(1, 3),
                )
            else:  # 80% chance of normal or slightly better house edge
                sim.mine_count = max(1, sim.mine_count - random.randint(0, 1))

            sim.initialize_game()

            while not sim.game_over:
                strategy.apply(sim)

            if sim.win:
                bankroll += sim.initial_bet * sim.current_payout
            else:
                bankroll -= sim.initial_bet

            session_results.append(
                {
                    "win": sim.win,
                    "final_payout": sim.current_payout,
                    "bankroll_after": bankroll,
                }
            )
            # print(f"  Session {i+1}: Win={sim.win}, Payout={sim.current_payout:.2f}, Bankroll: {bankroll:.2f}")

        print(f"Stress Test Results (Variable House Edge):")
        print(f"  Final Bankroll: {bankroll:.2f}")
        final_win_rate = np.mean([s["win"] for s in session_results])
        print(f"  Final Win Rate: {final_win_rate:.2%}")
        return {"final_bankroll": bankroll, "final_win_rate": final_win_rate}


def run_robustness_tests():
    # Generate some dummy historical data for backtesting
    # In a real scenario, this would come from the DataLogger
    historical_sessions = []
    for i in range(100):
        win = random.random() > 0.5
        payout = random.uniform(1.1, 3.0) if win else 0.0
        historical_sessions.append({"win": win, "final_payout": payout})

    tester = RobustnessTester()
    tester.run_k_fold_backtesting(historical_sessions, k_folds=5)
    tester.run_stress_test_long_losing_streak(num_sessions=20, stop_loss_threshold=0.3)
    tester.run_stress_test_variable_house_edge(num_sessions=100)


if __name__ == "__main__":
    run_robustness_tests()
