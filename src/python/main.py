"""
Main entry point for the Applied Probability and Automation Framework.
Author: Shihab Belal
"""

import time
import json
import logger
import bayesian
import sim_compare
import meta_controller
import bankroll_manager

# Core modules
import game_simulator
import mathematical_core
from game_simulator import GameSimulator
from bankroll_manager import BankrollSimulator, MoneyManager
import strategies
import utils

# Advanced strategy modules
import advanced_strategies
import rintaro_okabe_strategy
import monte_carlo_tree_search
import markov_decision_process
import strategy_auto_evolution
import confidence_weighted_ensembles

# AI and Machine Learning components
import drl_environment
import drl_agent
import bayesian_mines
import human_data_collector
import adversarial_agent
import adversarial_detector
import adversarial_trainer

# Multi-Agent System components
import multi_agent_core
import agent_comms
import multi_agent_simulator

# Behavioral Economics integration
import behavioral_value
import behavioral_probability

# Visualization and Analytics
import visualizations.visualization as visualization
import visualizations.advanced_visualizations as advanced_visualizations
import visualizations.comprehensive_visualizations as comprehensive_visualizations
import visualizations.specialized_visualizations as specialized_visualizations
import visualizations.realtime_heatmaps as realtime_heatmaps

# Testing modules
# from .. import drl_evaluation
# from .. import ab_testing_framework


def main():
    """
    Main function to run the framework.
    """
    print("Applied Probability and Automation Framework")
    print("All modules imported successfully.")


if __name__ == "__main__":
    main()

    print("\n--- Demonstrating High-Impact Features ---")

    # 1. Demonstrate Event Logger
    print("\nDemonstrating Event Logger...")
    dummy_session = {
        "ts": time.time(),
        "grid_size": 5,
        "mine_count": 3,
        "seed": 123,
        "strategy": "BasicStrategy",
        "win": True,
        "payout": 2.5,
        "cells_clicked": 3,
    }
    logger.log_session(dummy_session)
    print("Dummy session logged to logs/sessions.jsonl")

    # 2. Demonstrate Bayesian Click Estimator + Fractional Kelly
    print("\nDemonstrating Bayesian Estimator and Fractional Kelly...")
    estimator = bayesian.BetaEstimator(
        a=10, b=10
    )  # Start with some prior # Start with some prior
    print(
        f"Initial Bayesian Estimator: Mean={estimator.mean():.4f}, CI={estimator.ci()}"
    )
    # Simulate some wins and losses
    estimator.update(wins=5, losses=2)
    print(f"After 5 wins, 2 losses: Mean={estimator.mean():.4f}, CI={estimator.ci()}")
    # Assuming net odds of 1.0 (e.g., 2x payout)
    kelly_fraction = bayesian.fractional_kelly(estimator.mean(), 1.0, 0.5)  # 50% Kelly
    print(f"Fractional Kelly bet fraction (50%): {kelly_fraction:.4f}")

    # 3. Demonstrate Simulator-to-Live Comparator
    print("\nDemonstrating Simulator-to-Live Comparator...")
    # Ensure there are some logged sessions for comparison
    if not logger.OUT.exists():
        # Create a dummy log file if it doesn't exist
        if not logger.OUT.parent.exists():
            logger.OUT.parent.mkdir(parents=True, exist_ok=True)
        with logger.OUT.open("w") as f:
            f.write(json.dumps(dummy_session) + "\n")
    sim_compare.compare_sim_to_live()

    # 4. Refined Strategy Implementations (demonstrated implicitly by running strategies)
    # This would require running a full simulation with a strategy that uses the new Kelly logic.
    # For a quick demo, we can just show the Kelly calculation with adjusted conservatism.
    print("\nDemonstrating Refined Strategy Logic (Conservative Kelly)...")
    p_win_strategy = 0.55
    ci_width_strategy = 0.2  # High uncertainty
    f_conservative = 0.1  # Default conservative fraction
    if ci_width_strategy > 0.15:
        f_conservative = 0.05  # Even more conservative
    kelly_bet_conservative = bayesian.fractional_kelly(
        p_win_strategy, 1.0, f_conservative
    )
    print(
        f"Strategy P_win={p_win_strategy}, CI_width={ci_width_strategy} -> Conservative Kelly fraction={f_conservative:.2f}, Bet={kelly_bet_conservative:.4f}"
    )

    # 5. Implement Meta-Controller using Thompson Sampling
    print("\nDemonstrating Meta-Controller (Thompson Sampling)...")
    # This would typically run over many game rounds
    # For a quick demo, we'll instantiate and show a selection
    strategies_for_meta = [
        strategies.BasicStrategy(
            bayesian.BetaEstimator()
        ),  # Example with BasicStrategy
        # Add other strategies here once they are updated to accept an estimator
        # meta_controller.AggressiveStrategy(), # if these are defined outside main
        # meta_controller.ConservativeStrategy(), # if these are defined outside main
    ]
    strategy_manager = meta_controller.StrategyManager(strategies_for_meta)

    selected_strategy = strategy_manager.select_strategy_thompson_sampling()
    selected_strategy_name = selected_strategy.name
    print(f"Meta-Controller selected strategy: {selected_strategy_name}")

    # Define dummy strategies for Meta-Controller demonstration
    class DummyAggressiveStrategy:
        def __init__(self):
            self.name = "Aggressive"
            self.estimator = bayesian.BetaEstimator(a=5, b=5)

        def apply(self, simulator, bankroll_manager):
            # Simplified apply for demo
            if random.random() < 0.6:  # Simulate a win
                simulator.win = True
                simulator.current_payout = 2.0
                self.estimator.update(wins=1, losses=0)
            else:
                simulator.win = False
                simulator.current_payout = 0.0
                self.estimator.update(wins=0, losses=1)
            bankroll_manager.bankroll_simulator.process_outcome(
                simulator.win, simulator.current_payout
            )
            return (
                10,
                self.estimator.mean(),
                self.estimator.ci()[1] - self.estimator.ci()[0],
            )

    class DummyConservativeStrategy:
        def __init__(self):
            self.name = "Conservative"
            self.estimator = bayesian.BetaEstimator(a=15, b=5)

        def apply(self, simulator, bankroll_manager):
            # Simplified apply for demo
            if random.random() < 0.4:  # Simulate a win
                simulator.win = True
                simulator.current_payout = 1.5
                self.estimator.update(wins=1, losses=0)
            else:
                simulator.win = False
                simulator.current_payout = 0.0
                self.estimator.update(wins=0, losses=1)
            bankroll_manager.bankroll_simulator.process_outcome(
                simulator.win, simulator.current_payout
            )
            return (
                5,
                self.estimator.mean(),
                self.estimator.ci()[1] - self.estimator.ci()[0],
            )

    strategies_for_meta = [
        strategies.BasicStrategy(
            bayesian.BetaEstimator()
        ),  # Example with BasicStrategy
        DummyAggressiveStrategy(),
        DummyConservativeStrategy(),
    ]
    strategy_manager = meta_controller.StrategyManager(strategies_for_meta)

    # Add imports for random, numpy, and GameSimulator for the Meta-Controller demo
    # These are already imported at the top, but ensuring context for the demo block
    import random  # Ensure random is available in this scope
    import numpy as np  # Ensure numpy is available in this scope

    print("\n--- Meta-Controller Simulation (Thompson Sampling) ---")
    num_games = 10
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
        selected_strategy = strategy_manager.select_strategy_thompson_sampling()
        print(f"  Selected Strategy: {selected_strategy.name}")

        sim = GameSimulator(
            seed=random.randint(0, 100000)
        )  # Use a random seed for each game
        sim.initialize_game()

        # Apply the strategy, which now handles its own Bayesian update and betting
        bet_amount, p_win, ci_width = selected_strategy.apply(sim, money_manager)

        win = sim.win
        payout = sim.current_payout if sim.win else 0.0
        strategy_manager.update_performance(selected_strategy.name, win, payout)
        print(
            f"  Game Result: {'Win' if win else 'Loss'}, Payout: {payout:.2f}, Bet: {bet_amount:.2f}"
        )
        print(
            f"  Strategy {selected_strategy.name} Estimator: Mean={selected_strategy.estimator.mean():.2%}, CI Width={ci_width:.2%}"
        )

        # Print rolling metrics after every few games
        if (i + 1) % 5 == 0:
            print("\n--- Current Rolling Metrics ---")
            for name in strategy_manager.strategies.keys():
                metrics = strategy_manager.get_rolling_metrics(name)
                print(
                    f"  {name}: Win Rate={metrics['win_rate']:.2%}, Avg Payout={metrics['avg_payout']:.2f}, EV={metrics['ev']:.2f}"
                )

    print("\n--- Final Thompson Sampling Parameters (Estimated Win Probs) ---")
    for name, strategy_instance in strategy_manager.strategies.items():
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

    # 6. CI, Tests, and Reproducible Simulations (demonstrated by pytest and run_deterministic_sim.py)
    print(
        "\nCI, Tests, and Reproducible Simulations are handled via GitHub Actions and `run_deterministic_sim.py`."
    )
    print(
        "Please refer to .github/workflows/ci.yml and run_deterministic_sim.py for details."
    )

    print("\n--- High-Impact Feature Demonstrations Complete ---")

    # --- Demonstrating Mathematical Core Enhancements ---
    print("\n--- Demonstrating Mathematical Core Enhancements ---")

    # Demo 1: Confidence Intervals
    print("\n1. Confidence Intervals for Simulation Results")
    dummy_payouts = [
        random.gauss(1.5, 0.5) for _ in range(1000)
    ]  # Simulate 1000 game payouts
    mean, ci_lower, ci_upper = mathematical_core.calculate_confidence_interval(
        dummy_payouts
    )
    print(f"  Simulated Payout Mean: {mean:.4f}")
    print(f"  95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")

    # Demo 2: Monte Carlo Convergence Tracking
    print("\n2. Monte Carlo Convergence Tracking")
    tracker = mathematical_core.MonteCarloConvergenceTracker(
        window_size=100, tolerance=0.01
    )
    simulated_means = []
    for i in range(1, 1000):
        current_mean = np.mean(
            [random.random() for _ in range(i * 10)]
        )  # Growing number of samples
        simulated_means.append(current_mean)
        tracker.add_result(current_mean)
        converged, message = tracker.check_convergence()
        if converged:
            print(f"  Convergence detected at {i*10} samples: {message}")
            break
    if not converged:
        print("  Convergence not detected within 1000 iterations.")

    # Demo 3: Antithetic Variates (Conceptual)
    print("\n3. Variance Reduction with Antithetic Variates (Conceptual)")

    def simple_mines_payout_sim(seed):
        random.seed(seed)
        # Simplified simulation: 50% chance of winning 2x, 50% chance of losing 1x
        return 2.0 if random.random() < 0.5 else 0.0

    num_sims_no_antithetic = 2000
    results_no_antithetic = [
        simple_mines_payout_sim(random.randint(0, 2**32 - 1))
        for _ in range(num_sims_no_antithetic)
    ]
    mean_no_antithetic, ci_lower_no_antithetic, ci_upper_no_antithetic = (
        mathematical_core.calculate_confidence_interval(results_no_antithetic)
    )
    print(
        f"  Without Antithetic Variates (2000 simulations): Mean={mean_no_antithetic:.4f}, CI Width={ci_upper_no_antithetic - ci_lower_no_antithetic:.4f}"
    )

    num_pairs = 1000
    results_antithetic_concept = mathematical_core.simulate_with_antithetic_variates(
        simple_mines_payout_sim, num_pairs
    )
    mean_antithetic, ci_lower_antithetic, ci_upper_antithetic = (
        mathematical_core.calculate_confidence_interval(results_antithetic_concept)
    )
    print(
        f"  With Antithetic Variates Concept (2000 simulations): Mean={mean_antithetic:.4f}, CI Width={ci_upper_antithetic - ci_lower_antithetic:.4f}"
    )
    print(
        "  (Note: True antithetic variates would show a more significant reduction in CI width if properly implemented for the specific simulation.)"
    )

    # Demo 4: Markov Decision Process (Conceptual)
    print("\n4. Markov Decision Process (Conceptual Outline)")
    print("  Implementing a full MDP for Mines is complex. It would involve:")
    print(
        "  - Defining a finite set of states (e.g., (revealed_cells_mask, current_payout, clicks_made))."
    )
    print("  - Defining actions (e.g., click_cell(r,c), cash_out).")
    print(
        "  - Calculating transition probabilities P(s' | s, a) based on mine distribution."
    )
    print("  - Defining rewards R(s, a, s') (e.g., payout on cash-out, -1 on mine).")
    print(
        "  - Using algorithms like Value Iteration or Policy Iteration to find the optimal policy."
    )
    print(
        "  The `MarkovDecisionProcess` class above provides a skeletal structure for such an implementation."
    )

    print("\n--- Mathematical Core Enhancements Complete ---")
