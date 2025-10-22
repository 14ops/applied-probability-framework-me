import numpy as np
from scipy import stats
import random


# 1. Add confidence intervals and error bounds for simulation results
def calculate_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n < 2:
        return np.nan, np.nan, np.nan  # Not enough data for meaningful CI
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


# 2. Move toward Markov decision modeling (conceptual outline)
# This would involve defining states, actions, transition probabilities, and rewards.
# For a Mines game, states could be (board_state, clicks_made), actions are (click_row, click_col).
# Solving for optimal policy would require dynamic programming (Value Iteration, Policy Iteration)
# or reinforcement learning (Q-learning, SARSA).


class MarkovDecisionProcess:
    def __init__(self, states, actions, transitions, rewards):
        self.states = states
        self.actions = actions
        self.transitions = transitions  # P(s' | s, a)
        self.rewards = rewards  # R(s, a, s')

    def value_iteration(self, gamma=0.9, epsilon=1e-6):
        V = {s: 0 for s in self.states}
        while True:
            V_prime = V.copy()
            delta = 0
            for s in self.states:
                q_values = []
                for a in self.actions:
                    q = 0
                    for s_prime in self.states:
                        prob = self.transitions.get((s, a, s_prime), 0)
                        reward = self.rewards.get((s, a, s_prime), 0)
                        q += prob * (reward + gamma * V[s_prime])
                    q_values.append(q)
                V_prime[s] = max(q_values) if q_values else 0
                delta = max(delta, abs(V_prime[s] - V[s]))
            V = V_prime
            if delta < epsilon:
                break
        return V


# 3. Add Monte Carlo convergence tracking
class MonteCarloConvergenceTracker:
    def __init__(self, metric_name="mean", window_size=50, tolerance=1e-3):
        self.metric_name = metric_name
        self.results = []
        self.window_size = window_size
        self.tolerance = tolerance

    def add_result(self, result):
        self.results.append(result)

    def check_convergence(self):
        if len(self.results) < self.window_size * 2:
            return False, "Not enough samples for convergence check."

        # Compare the mean of the last two windows
        last_window_mean = np.mean(self.results[-self.window_size :])
        prev_window_mean = np.mean(
            self.results[-2 * self.window_size : -self.window_size]
        )

        if abs(last_window_mean - prev_window_mean) < self.tolerance:
            return (
                True,
                f"Converged: {self.metric_name} changed by less than {self.tolerance}.",
            )
        return (
            False,
            f"Not converged: {self.metric_name} difference = {abs(last_window_mean - prev_window_mean):.4f}.",
        )


# 4. Implement variance reduction techniques (e.g., Antithetic Variates)
# For a simple simulation where we're estimating the mean of some outcome.


def simulate_with_antithetic_variates(simulation_function, num_pairs):
    results = []
    for _ in range(num_pairs):
        # Run first simulation with a random seed
        seed1 = random.randint(0, 2**32 - 1)
        result1 = simulation_function(seed1)
        results.append(result1)

        # Run second simulation with an 'antithetic' seed or inverted randomness
        # This is a simplified example; true antithetic variates require careful design
        # For a Mines game, this might mean inverting mine placements or click order.
        # Here, we'll just use a different seed for simplicity of demonstration.
        seed2 = random.randint(
            0, 2**32 - 1
        )  # Not truly antithetic, but demonstrates the concept
        result2 = simulation_function(seed2)
        results.append(result2)
    return results


if __name__ == "__main__":
    print("--- Mathematical Core Enhancements Demonstration ---")

    # Demo 1: Confidence Intervals
    print("\n1. Confidence Intervals for Simulation Results")
    dummy_payouts = [
        random.gauss(1.5, 0.5) for _ in range(1000)
    ]  # Simulate 1000 game payouts
    mean, ci_lower, ci_upper = calculate_confidence_interval(dummy_payouts)
    print(f"  Simulated Payout Mean: {mean:.4f}")
    print(f"  95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")

    # Demo 2: Monte Carlo Convergence Tracking
    print("\n2. Monte Carlo Convergence Tracking")
    tracker = MonteCarloConvergenceTracker(window_size=100, tolerance=0.01)
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
        calculate_confidence_interval(results_no_antithetic)
    )
    print(
        f"  Without Antithetic Variates (2000 simulations): Mean={mean_no_antithetic:.4f}, CI Width={ci_upper_no_antithetic - ci_lower_no_antithetic:.4f}"
    )

    # For true antithetic variates, the second simulation would be designed to be negatively correlated.
    # Here, we'll just run more simulations to show the principle of reducing variance with more data.
    # A proper implementation for Mines would involve mirroring mine placements or click sequences.
    num_pairs = 1000
    results_antithetic_concept = simulate_with_antithetic_variates(
        simple_mines_payout_sim, num_pairs
    )
    mean_antithetic, ci_lower_antithetic, ci_upper_antithetic = (
        calculate_confidence_interval(results_antithetic_concept)
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
