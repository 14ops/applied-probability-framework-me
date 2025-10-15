
import random
import numpy as np
from scipy.stats import chi2_contingency

# Placeholder for GameSimulator and a generic Strategy
class GameSimulator:
    def __init__(self, board_size=5, mine_count=5):
        self.board_size = board_size
        self.mine_count = mine_count
        self.game_over = False
        self.win = False

    def initialize_game(self):
        self.game_over = False
        self.win = False
        # Simplified game logic for A/B testing: just return a win/loss

    def play_game(self, strategy):
        self.initialize_game()
        # Simulate strategy playing a game and returning win/loss
        # For demonstration, let's assume Strategy A has a slightly higher win rate
        if strategy.name == "Strategy A":
            self.win = random.random() < 0.52 # 52% win rate
        elif strategy.name == "Strategy B":
            self.win = random.random() < 0.48 # 48% win rate
        else:
            self.win = random.random() < 0.50 # Default 50% win rate
        self.game_over = True
        return self.win

class GenericStrategy:
    def __init__(self, name):
        self.name = name


def run_ab_test(strategy_a, strategy_b, num_games_per_strategy):
    wins_a = 0
    losses_a = 0
    wins_b = 0
    losses_b = 0

    simulator = GameSimulator()

    print(f"\n--- Running A/B Test: {strategy_a.name} vs {strategy_b.name} ---")
    print(f"Simulating {num_games_per_strategy} games for each strategy.\n")

    for _ in range(num_games_per_strategy):
        # Test Strategy A
        if simulator.play_game(strategy_a):
            wins_a += 1
        else:
            losses_a += 1

        # Test Strategy B
        if simulator.play_game(strategy_b):
            wins_b += 1
        else:
            losses_b += 1

    total_a = wins_a + losses_a
    total_b = wins_b + losses_b

    win_rate_a = wins_a / total_a if total_a > 0 else 0
    win_rate_b = wins_b / total_b if total_b > 0 else 0

    print(f"Results for {strategy_a.name} (Total Games: {total_a}):")
    print(f"  Wins: {wins_a}, Losses: {losses_a}, Win Rate: {win_rate_a:.2%}")
    print(f"Results for {strategy_b.name} (Total Games: {total_b}):")
    print(f"  Wins: {wins_b}, Losses: {losses_b}, Win Rate: {win_rate_b:.2%}\n")

    # Perform Chi-squared test for statistical significance
    # Contingency table: [[wins_a, losses_a], [wins_b, losses_b]]
    contingency_table = np.array([[wins_a, losses_a], [wins_b, losses_b]])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)

    # Calculate Effect Size (e.g., Cohen's h for proportions)
    # For simplicity, we'll use difference in proportions here
    effect_size = win_rate_a - win_rate_b

    print(f"Statistical Analysis:")
    print(f"  Chi-squared statistic: {chi2:.2f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Effect Size (Difference in Win Rates): {effect_size:.2%}")

    alpha = 0.05 # Significance level
    if p_value < alpha:
        print(f"\nConclusion: With a p-value of {p_value:.4f} (which is less than {alpha}), the difference in win rates between {strategy_a.name} and {strategy_b.name} is statistically significant. This means {strategy_a.name if effect_size > 0 else strategy_b.name} is likely a genuinely better strategy, not just due to random chance. It's like discovering a new, more powerful jutsu!")
    else:
        print(f"\nConclusion: With a p-value of {p_value:.4f} (which is greater than or equal to {alpha}), the observed difference in win rates is not statistically significant. It's like two ninjas fighting, and the difference in their skill is too small to definitively say who is stronger based on these few battles.")

    return {
        "strategy_a_win_rate": win_rate_a,
        "strategy_b_win_rate": win_rate_b,
        "p_value": p_value,
        "effect_size": effect_size
    }


if __name__ == "__main__":
    strategy_a = GenericStrategy("Strategy A")
    strategy_b = GenericStrategy("Strategy B")

    # Run a test with a moderate number of games
    results_1 = run_ab_test(strategy_a, strategy_b, num_games_per_strategy=1000)

    print("\n" + "="*50 + "\n")

    # Run another test with more games to see if significance changes
    results_2 = run_ab_test(strategy_a, strategy_b, num_games_per_strategy=5000)

    print("\nThis A/B testing framework allows you to rigorously compare strategies and determine if observed differences are statistically meaningful, much like a seasoned scientist validating their hypotheses!")


