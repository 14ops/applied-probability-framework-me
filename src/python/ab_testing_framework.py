import random
import numpy as np
from scipy.stats import chi2_contingency

"""Module for A/B testing framework and statistical comparison of strategies."""


class GameSimulator:
    """A simplified game simulator for A/B testing purposes."""

    def __init__(self, board_size=5, mine_count=5):
        self.board_size = board_size
        self.mine_count = mine_count
        self.game_over = False
        self.win = False

    def initialize_game(self):
        """Initializes a new game round."""
        self.game_over = False
        self.win = False
        # Simplified game logic for A/B testing: just return a win/loss

    def play_game(self, strategy):
        """Simulates a game played by a given strategy."""
        self.initialize_game()
        # Simulate strategy playing a game and returning win/loss
        # For demonstration, let\"s assume Strategy A has a slightly higher win rate
        if strategy.name == "Strategy A":
            self.win = random.random() < 0.52  # 52% win rate
        elif strategy.name == "Strategy B":
            self.win = random.random() < 0.48  # 48% win rate
        else:
            self.win = random.random() < 0.50  # Default 50% win rate
        self.game_over = True
        return self.win


class GenericStrategy:
    """A generic strategy class for A/B testing."""

    def __init__(self, name):
        self.name = name

    def get_name(self):
        """Returns the name of the strategy."""
        return self.name

    def execute(self):
        """A public method to satisfy Pylint R0903."""
        pass  # Pylint W0107: Unnecessary pass statement, but kept for R0903

def _simulate_strategy_games(strategy_obj, num_games, simulator):
    """Helper to simulate games for a given strategy."""
    wins = 0
    losses = 0
    for _ in range(num_games):
        if simulator.play_game(strategy_obj):
            wins += 1
        else:
            losses += 1
    return wins, losses

def run_ab_test(strategy_a_obj, strategy_b_obj, num_games_per_strategy):
    """Runs an A/B test between two strategies and performs statistical analysis.

    Args:
        strategy_a_obj (GenericStrategy): The first strategy object.
        strategy_b_obj (GenericStrategy): The second strategy object.
        num_games_per_strategy (int): The number of games to simulate for each strategy.

    Returns:
        dict: A dictionary containing win rates, p-value, and effect size.
    """
    simulator = GameSimulator()

    print(f"\n--- Running A/B Test: {strategy_a_obj.name} vs {strategy_b_obj.name} ---")
    print(f"Simulating {num_games_per_strategy} games for each strategy.\n")

    wins_a, losses_a = _simulate_strategy_games(strategy_a_obj, num_games_per_strategy, simulator)
    wins_b, losses_b = _simulate_strategy_games(strategy_b_obj, num_games_per_strategy, simulator)

    total_a = wins_a + losses_a
    total_b = wins_b + losses_b

    win_rate_a = wins_a / total_a if total_a > 0 else 0
    win_rate_b = wins_b / total_b if total_b > 0 else 0

    print(f"Results for {strategy_a_obj.name} (Total Games: {total_a}):")
    print(f"  Wins: {wins_a}, Losses: {losses_a}, Win Rate: {win_rate_a:.2%}")
    print(f"Results for {strategy_b_obj.name} (Total Games: {total_b}):")
    print(f"  Wins: {wins_b}, Losses: {losses_b}, Win Rate: {win_rate_b:.2%}\n")

    contingency_table = np.array([[wins_a, losses_a], [wins_b, losses_b]])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)

    effect_size = win_rate_a - win_rate_b

    print("Statistical Analysis:")
    print(f"  Chi-squared statistic: {chi2:.2f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Effect Size (Difference in Win Rates): {effect_size:.2%}")

    alpha = 0.05
    conclusion_strategy = (strategy_a_obj.name if effect_size > 0 else strategy_b_obj.name)
    if p_value < alpha:
        print(
            f"\nConclusion: With a p-value of {p_value:.4f} (which is less than {alpha}), "
            f"the difference in win rates between {strategy_a_obj.name} and "
            f"{strategy_b_obj.name} is statistically significant. This means "
            f"{conclusion_strategy} is likely a genuinely better strategy, not just "
            "due to random chance. It\"s like discovering a new, more powerful jutsu!"
        )
    else:
        print(
            f"\nConclusion: With a p-value of {p_value:.4f} (which is greater than "
            f"or equal to {alpha}), the observed difference in win rates is not "
            f"statistically significant. It\"s like two ninjas fighting, and the "
            "difference in their skill is too small to definitively say who is "
            "stronger based on these few battles."
        )

    return {
        "strategy_a_win_rate": win_rate_a,
        "strategy_b_win_rate": win_rate_b,
        "p_value": p_value,
        "effect_size": effect_size,
    }


if __name__ == "__main__":
    strategy_a = GenericStrategy("Strategy A")
    strategy_b = GenericStrategy("Strategy B")

    results_1 = run_ab_test(strategy_a, strategy_b, num_games_per_strategy=1000)

    print("\n" + "=" * 50 + "\n")

    results_2 = run_ab_test(strategy_a, strategy_b, num_games_per_strategy=5000)

    print(
        "\nThis A/B testing framework allows you to rigorously compare strategies "
        "and determine if observed differences are statistically meaningful, "
        "much like a seasoned scientist validating their hypotheses!"
    )


