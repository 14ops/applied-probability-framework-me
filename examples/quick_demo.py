"""
Applied Probability Framework - Quick Demo

This script demonstrates the core capabilities of the framework.
Run with: python examples/quick_demo.py
"""

import sys
sys.path.append('src/python')

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from core.base_estimator import BetaEstimator

print("="*60)
print("Applied Probability Framework - Quick Demo")
print("="*60)

# Demo 1: Bayesian Inference
print("\n--- Demo 1: Bayesian Inference ---")
estimator = BetaEstimator(alpha=1.0, beta=1.0)
print(f"Prior estimate: {estimator.estimate():.4f}")

estimator.update(successes=7, failures=3)
print(f"After 10 observations (7 wins, 3 losses):")
print(f"  Posterior estimate: {estimator.estimate():.4f}")
print(f"  95% CI: {estimator.confidence_interval()}")

# Demo 2: Sequential Learning
print("\n--- Demo 2: Sequential Learning Convergence ---")
true_p = 0.65
estimator2 = BetaEstimator(alpha=1.0, beta=1.0)

np.random.seed(42)
for i in range(100):
    win = np.random.random() < true_p
    estimator2.update(successes=int(win), failures=int(not win))

print(f"True probability: {true_p}")
print(f"Estimated after 100 trials: {estimator2.estimate():.4f}")
print(f"Error: {abs(estimator2.estimate() - true_p):.4f}")
ci = estimator2.confidence_interval()
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}] (width: {ci[1]-ci[0]:.4f})")

# Demo 3: Thompson Sampling
print("\n--- Demo 3: Thompson Sampling (Multi-Armed Bandit) ---")
true_probs = [0.45, 0.55, 0.60]
n_arms = len(true_probs)
estimators = [BetaEstimator(alpha=1.0, beta=1.0) for _ in range(n_arms)]
arm_selections = [0] * n_arms
cumulative_reward = 0

np.random.seed(42)
for _ in range(300):
    # Thompson Sampling: sample from each posterior
    samples = [est.sample(1)[0] for est in estimators]
    selected_arm = np.argmax(samples)
    
    # Play selected arm
    win = np.random.random() < true_probs[selected_arm]
    estimators[selected_arm].update(successes=int(win), failures=int(not win))
    arm_selections[selected_arm] += 1
    cumulative_reward += int(win)

print("Results after 300 rounds:")
for i in range(n_arms):
    print(f"  Arm {i} (true p={true_probs[i]:.2f}):")
    print(f"    Estimated: {estimators[i].estimate():.4f}")
    print(f"    Selected: {arm_selections[i]} times")

print(f"\nTotal reward: {cumulative_reward}")
print(f"Optimal reward: {int(300 * max(true_probs))}")
print(f"Performance: {cumulative_reward/(300*max(true_probs))*100:.1f}% of optimal")

# Demo 4: Kelly Criterion
print("\n--- Demo 4: Kelly Criterion Bankroll Growth ---")

def kelly_criterion(p, b=1.0):
    return max(0, (p * (b + 1) - 1) / b)

true_p = 0.55
initial_bankroll = 1000
n_bets = 500

strategies = {
    'Full Kelly': kelly_criterion(true_p),
    'Half Kelly': kelly_criterion(true_p) * 0.5,
    'Quarter Kelly': kelly_criterion(true_p) * 0.25,
}

results = {}
np.random.seed(42)

for name, fraction in strategies.items():
    bankroll = initial_bankroll
    for _ in range(n_bets):
        win = np.random.random() < true_p
        bet_amount = bankroll * fraction
        bankroll += bet_amount if win else -bet_amount
        bankroll = max(0, bankroll)
    results[name] = bankroll

print(f"Starting bankroll: ${initial_bankroll}")
print(f"After {n_bets} bets at {true_p:.0%} win rate:")
for name, final_bankroll in results.items():
    growth = (final_bankroll / initial_bankroll - 1) * 100
    print(f"  {name:15s}: ${final_bankroll:8.0f} ({growth:+.1f}% growth)")

print("\n" + "="*60)
print("✅ Demo Complete!")
print("\nNext steps:")
print("  - Check docs/API_REFERENCE.md for full documentation")
print("  - Read docs/TUTORIALS.md for more examples")
print("  - Explore docs/THEORETICAL_BACKGROUND.md for the math")
print("="*60)

