import random
import numpy as np
from src.python.game_simulator import GameSimulator
from src.python.strategies import BasicStrategy
from src.python.bankroll_manager import BankrollSimulator, MoneyManager
from src.python.bayesian import BetaEstimator

"""Module for running deterministic simulations to ensure reproducibility and test stability."""

# Fixed seed for reproducibility
FIXED_SEED = 42
random.seed(FIXED_SEED)
np.random.seed(FIXED_SEED)


def _simulate_game_round(sim, basic_strategy, money_manager):
    """Helper function to simulate a single game round."""
    _bet_amount, p_win, _ci_width = basic_strategy.apply(sim, money_manager)

    if random.random() < p_win:
        sim.win = True
        sim.current_payout = 1.1  # Simplified payout for win
    else:
        sim.win = False
        sim.current_payout = 0.0  # Simplified payout for loss
    return sim.win, sim.current_payout


def run_deterministic_simulation(num_games=100):
    """Runs a deterministic simulation of the game using a fixed seed.

    This function is crucial for CI/CD pipelines to ensure that changes to the code
    do not inadvertently alter the simulation outcomes, thus maintaining reproducibility.

    Args:
        num_games (int): The number of game rounds to simulate.
    """
    print(f"Running deterministic simulation with seed: {FIXED_SEED}")

    initial_bankroll = 1000
    initial_bet_size = 10

    bankroll_sim = BankrollSimulator(initial_bankroll, initial_bet_size)
    money_manager = MoneyManager(bankroll_sim)

    # Initialize strategy with a Bayesian estimator
    basic_strategy = BasicStrategy(BetaEstimator())

    total_payout = 0
    total_wins = 0

    for i in range(num_games):
        if bankroll_sim.is_ruined() or money_manager.check_stop_loss():
            print(
                f"Simulation stopped at game {i+1}. "
                f"Bankroll: {bankroll_sim.get_current_bankroll():.2f}"
            )
            break

        sim = GameSimulator(
            seed=FIXED_SEED + i
        )  # Use a derived seed for each game for internal game randomness
        sim.initialize_game()

        win, payout = _simulate_game_round(sim, basic_strategy, money_manager)

        if win:
            total_wins += 1
            total_payout += payout

    final_bankroll = bankroll_sim.get_current_bankroll()
    win_rate = total_wins / num_games if num_games > 0 else 0
    avg_payout = total_payout / total_wins if total_wins > 0 else 0

    print("\n--- Deterministic Simulation Results ---")
    print(f"Final Bankroll: {final_bankroll:.2f}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Payout (on wins): {avg_payout:.2f}")
    print(f"Max Drawdown: {bankroll_sim.get_max_drawdown():.2%}")

    # Assertions for CI/CD pipeline
    # These values should be stable given the fixed seed
    assert final_bankroll > 900, "Bankroll dropped too low!"
    assert win_rate > 0.4, "Win rate too low!"
    assert bankroll_sim.get_max_drawdown() < 0.2, "Max drawdown too high!"

    print(
        "Deterministic simulation test passed! All metrics within expected tolerance."
    )


if __name__ == "__main__":
    run_deterministic_simulation()


