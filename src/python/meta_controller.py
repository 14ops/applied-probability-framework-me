import random
import numpy as np
from collections import deque
from bayesian import BetaEstimator, fractional_kelly
from strategies import BasicStrategy  # Assuming BasicStrategy is now updated

# Import other strategies as they are updated with Bayesian/Kelly logic
# Import DRL strategy with fallback
try:
    from strategies.drl_strategy import DRLStrategy
    DRL_AVAILABLE = True
except ImportError:
    DRL_AVAILABLE = False
    print("Warning: DRL strategy not available (torch may be missing)")


# Placeholder for individual strategy classes (e.g., BasicStrategy, TakeshiStrategy, etc.)
# In a real scenario, these would be imported from src/strategies.py, src/advanced_strategies.py, etc.
class BaseStrategy:
    def __init__(self, name="BaseStrategy"):
        self.name = name
        self.estimator = (
            BetaEstimator()
        )  # Each strategy gets its own Bayesian estimator

    def apply(self, simulator, bankroll_manager):
        # Placeholder for strategy logic
        pass


# The BasicStrategy class from strategies.py is now expected to have the apply method
# that uses the estimator and bankroll_manager. We will use that directly.


class StrategyManager:
    def __init__(self, strategies, window_size=100):
        self.strategies = {s.name: s for s in strategies}
        self.performance_history = {
            name: deque(maxlen=window_size) for name in self.strategies.keys()
        }
        # Thompson parameters are now managed within each strategy's BetaEstimator
        self.window_size = window_size

    def update_performance(self, strategy_name, win, payout):
        # Update rolling performance metrics
        self.performance_history[strategy_name].append({"win": win, "payout": payout})

        # The Bayesian estimator update is now handled within the strategy's apply method
        # This method is primarily for logging and meta-level performance tracking

    def select_strategy_thompson_sampling(self):
        best_strategy = None
        max_sampled_value = -np.inf

        for name, strategy_instance in self.strategies.items():
            # Sample from Beta distribution (win probability) for each strategy
            alpha = strategy_instance.estimator.a
            beta_param = strategy_instance.estimator.b
            sampled_win_prob = np.random.beta(alpha, beta_param)

            # For simplicity, let's assume an average expected payout of 1.5 for a win
            # In a real scenario, this would be dynamically estimated or passed.
            expected_payout_if_win = 1.5
            sampled_value = sampled_win_prob * expected_payout_if_win

            if sampled_value > max_sampled_value:
                max_sampled_value = sampled_value
                best_strategy = strategy_instance

        return best_strategy

    def get_rolling_metrics(self, strategy_name):
        history = self.performance_history[strategy_name]
        if not history:
            return {
                "win_rate": 0,
                "avg_payout": 0,
                "ev": 0,
                "sharpe_ratio": 0,
                "drawdown": 0,
            }

        wins = sum(1 for h in history if h["win"])
        win_rate = wins / len(history)
        payouts = [h["payout"] for h in history if h["win"]]
        avg_payout = np.mean(payouts) if payouts else 0

        # Simplified EV calculation (assuming initial bet of 1 unit)
        ev_per_game = (
            (wins * avg_payout - (len(history) - wins) * 1) / len(history)
            if len(history) > 0
            else 0
        )

        # Placeholder for Sharpe Ratio and Drawdown - these require more complex bankroll tracking
        sharpe_ratio = 0
        drawdown = 0

        return {
            "win_rate": win_rate,
            "avg_payout": avg_payout,
            "ev": ev_per_game,
            "sharpe_ratio": sharpe_ratio,
            "drawdown": drawdown,
        }


# Example Usage (requires a GameSimulator instance and BankrollManager)
if __name__ == "__main__":
    # from .game_simulator import GameSimulator # Assuming GameSimulator is in src
    from .bankroll_manager import (
        BankrollSimulator,
        MoneyManager,
    )  # Assuming BankrollManager is in src

    # Initialize strategies and manager
    strategies = [
        BasicStrategy(BetaEstimator()),  # Pass a new estimator to each strategy
        AggressiveStrategy(),
        ConservativeStrategy(),
    ]
    manager = StrategyManager(strategies)

    print("\n--- Meta-Controller Simulation (Thompson Sampling) ---")
    num_games = 100
    initial_bankroll = 1000
    initial_bet_size = 10

    bankroll_sim = BankrollSimulator(initial_bankroll, initial_bet_size)
    money_manager = MoneyManager(bankroll_sim)

    for i in range(num_games):
        if bankroll_sim.is_ruined() or money_manager.check_stop_loss():
            print(
                f"Simulation stopped at game {i+1}. Bankroll: {bankroll_sim.get_current_bankroll():.2f}"
            )
            break

        print(
            f"\nGame {i+1}/{num_games} (Bankroll: {bankroll_sim.get_current_bankroll():.2f})"
        )
        selected_strategy = manager.select_strategy_thompson_sampling()
        print(f"  Selected Strategy: {selected_strategy.name}")

        sim = GameSimulator(
            seed=random.randint(0, 100000)
        )  # Use a random seed for each game
        sim.initialize_game()

        # Apply the strategy, which now handles its own Bayesian update and betting
        bet_amount, p_win, ci_width = selected_strategy.apply(sim, money_manager)

        win = sim.win
        payout = sim.current_payout if sim.win else 0.0
        manager.update_performance(selected_strategy.name, win, payout)
        print(
            f"  Game Result: {'Win' if win else 'Loss'}, Payout: {payout:.2f}, Bet: {bet_amount:.2f}"
        )
        print(
            f"  Strategy {selected_strategy.name} Estimator: Mean={selected_strategy.estimator.mean():.2%}, CI Width={ci_width:.2%}"
        )

        # Print rolling metrics after every few games
        if (i + 1) % 20 == 0:
            print("\n--- Current Rolling Metrics ---")
            for name in manager.strategies.keys():
                metrics = manager.get_rolling_metrics(name)
                print(
                    f"  {name}: Win Rate={metrics['win_rate']:.2%}, Avg Payout={metrics['avg_payout']:.2f}, EV={metrics['ev']:.2f}"
                )

    print("\n--- Final Thompson Sampling Parameters (Estimated Win Probs) ---")
    for name, strategy_instance in manager.strategies.items():
        alpha = strategy_instance.estimator.a
        beta_param = strategy_instance.estimator.b
        print(
            f"  {name}: Alpha={alpha}, Beta={beta_param}, Estimated Win Prob={strategy_instance.estimator.mean():.2%}"
        )

    print(f"\nFinal Bankroll: {bankroll_sim.get_current_bankroll():.2f}")
    print(f"Max Drawdown: {bankroll_sim.get_max_drawdown():.2%}")
    print(
        "\nMeta-controller simulation complete. The StrategyManager dynamically selected strategies based on their estimated performance, demonstrating a basic form of adaptive play, much like a seasoned anime protagonist adapting their fighting style mid-battle!"
    )
