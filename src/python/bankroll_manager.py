import random
import numpy as np


class BankrollSimulator:
    def __init__(self, initial_bankroll, initial_bet_size=1.0):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.initial_bet_size = initial_bet_size
        self.bet_size = initial_bet_size
        self.bankroll_history = [initial_bankroll]
        self.peak_bankroll = initial_bankroll
        self.in_game_bet = 0.0  # Bet placed for the current game

    def place_bet(self, amount):
        if amount > self.current_bankroll:
            # print(f"Warning: Attempted to bet {amount:.2f} but only {self.current_bankroll:.2f} available. Betting all available funds.")
            self.in_game_bet = self.current_bankroll
            self.current_bankroll = 0  # Bankroll becomes 0 if all is bet
        else:
            self.in_game_bet = amount
            self.current_bankroll -= amount
        return self.in_game_bet

    def process_outcome(self, win, payout_multiplier=0.0):
        if win:
            winnings = self.in_game_bet * payout_multiplier
            self.current_bankroll += winnings
        # If loss, the bet amount was already deducted

        self.bankroll_history.append(self.current_bankroll)
        self.peak_bankroll = max(self.peak_bankroll, self.current_bankroll)
        self.in_game_bet = 0.0  # Reset in-game bet

    def get_current_bankroll(self):
        return self.current_bankroll

    def get_bankroll_history(self):
        return self.bankroll_history

    def get_max_drawdown(self):
        if not self.bankroll_history:
            return 0.0
        max_drawdown = 0.0
        peak = self.bankroll_history[0]
        for balance in self.bankroll_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        return max_drawdown

    def is_ruined(self):
        return self.current_bankroll <= 0


def fractional_kelly(p_win, net_odds, f=0.5):
    # p_win = probability of winning
    # net_odds = (payout_multiplier - 1) for a 1 unit bet
    # f = fraction of Kelly criterion to use (e.g., 0.5 for half-Kelly)
    if net_odds <= 0:
        return 0.0  # No positive edge

    kelly_fraction = (p_win * (net_odds + 1) - 1) / net_odds
    return max(0.0, f * kelly_fraction)


class MoneyManager:
    def __init__(
        self,
        bankroll_simulator,
        stop_loss_percent=0.5,
        drawdown_reduction_threshold=0.2,
        drawdown_reduction_factor=0.5,
    ):
        self.bankroll_simulator = bankroll_simulator
        self.stop_loss_percent = stop_loss_percent  # e.g., 0.5 means stop if bankroll drops to 50% of initial
        self.drawdown_reduction_threshold = drawdown_reduction_threshold  # e.g., 0.2 means reduce stake if drawdown > 20%
        self.drawdown_reduction_factor = (
            drawdown_reduction_factor  # e.g., 0.5 means halve the stake
        )
        self.original_bet_size = bankroll_simulator.initial_bet_size
        self.current_bet_size = bankroll_simulator.initial_bet_size

    def check_stop_loss(self):
        if (
            self.bankroll_simulator.get_current_bankroll()
            <= self.bankroll_simulator.initial_bankroll * (1 - self.stop_loss_percent)
        ):
            print(
                f"!!! STOP LOSS TRIGGERED! Bankroll at {self.bankroll_simulator.get_current_bankroll():.2f}"
            )
            return True
        return False

    def adjust_bet_size(self):
        current_drawdown = self.bankroll_simulator.get_max_drawdown()
        if current_drawdown >= self.drawdown_reduction_threshold:
            # Reduce bet size, but not below a minimum (e.g., 1 unit or original_bet_size * 0.1)
            new_bet_size = max(
                self.original_bet_size * 0.1,
                self.current_bet_size * (1 - self.drawdown_reduction_factor),
            )
            if new_bet_size < self.current_bet_size:
                # print(f"  Stake reduced due to drawdown ({current_drawdown:.2%}). New bet size: {new_bet_size:.2f}")
                self.current_bet_size = new_bet_size
        else:
            # Optionally, increase bet size if performance is good and drawdown is low
            # For now, we'll just reset to original if recovered significantly
            if (
                current_drawdown < self.drawdown_reduction_threshold / 2
                and self.current_bet_size < self.original_bet_size
            ):
                # print(f"  Stake increased due to recovery. New bet size: {self.original_bet_size:.2f}")
                self.current_bet_size = self.original_bet_size

    def get_recommended_bet_size(self, p_win=None, net_odds=None, kelly_fraction=0.5):
        bet_from_kelly = 0.0
        if p_win is not None and net_odds is not None:
            kelly_bet_fraction = fractional_kelly(p_win, net_odds, kelly_fraction)
            bet_from_kelly = (
                kelly_bet_fraction * self.bankroll_simulator.get_current_bankroll()
            )

        # Combine Kelly with dynamic stake sizing, ensuring a minimum bet
        # Use max(1.0, ...) to ensure a minimum bet of 1 unit
        return max(
            1.0,
            min(
                bet_from_kelly if bet_from_kelly > 0 else self.current_bet_size,
                self.current_bet_size,
            ),
        )


# Example Usage
if __name__ == "__main__":
    initial_bankroll = 1000
    initial_bet = 10
    num_games = 200

    sim = BankrollSimulator(initial_bankroll, initial_bet)
    mm = MoneyManager(
        sim,
        stop_loss_percent=0.2,
        drawdown_reduction_threshold=0.15,
        drawdown_reduction_factor=0.5,
    )

    print("\n--- Bankroll Management Simulation ---")
    print(f"Initial Bankroll: {sim.get_current_bankroll():.2f}")

    for i in range(num_games):
        if sim.is_ruined() or mm.check_stop_loss():
            print(
                f"Simulation stopped at game {i+1}. Bankroll: {sim.get_current_bankroll():.2f}"
            )
            break

        # Simulate a strategy's estimated win probability and net odds
        # In a real scenario, these would come from the Meta-Controller or Bayesian Estimator
        # Adjusted to be slightly more favorable for demonstration
        estimated_p_win = random.uniform(
            0.51, 0.55
        )  # Example: fluctuating win prob, slightly positive edge
        estimated_net_odds = random.uniform(
            0.9, 1.2
        )  # Example: fluctuating net odds (payout - 1)

        # Adjust bet size based on dynamic rules and Kelly criterion
        mm.adjust_bet_size()
        bet_amount = mm.get_recommended_bet_size(
            estimated_p_win, estimated_net_odds, kelly_fraction=0.3
        )
        bet_amount = min(
            bet_amount, sim.get_current_bankroll()
        )  # Ensure bet doesn't exceed current bankroll

        if bet_amount <= 0:
            print(
                f"No funds or bet size too small to continue at game {i+1}. Bankroll: {sim.get_current_bankroll():.2f}"
            )
            break

        sim.place_bet(bet_amount)

        # Simulate game outcome
        win_game = random.random() < estimated_p_win
        payout_mult = estimated_net_odds + 1 if win_game else 0.0
        sim.process_outcome(win_game, payout_mult)

        if (i + 1) % 20 == 0:
            print(
                f"Game {i+1}: Bankroll={sim.get_current_bankroll():.2f}, Bet Size={bet_amount:.2f}, Max Drawdown={sim.get_max_drawdown():.2%}"
            )

    print(f"\nFinal Bankroll: {sim.get_current_bankroll():.2f}")
    print(f"Max Drawdown: {sim.get_max_drawdown():.2%}")
    print(
        f"Probability of Ruin (conceptual): This would be derived from many such simulations."
    )
    print(
        "Bankroll management simulation complete. Your financial fortress is now equipped with automated defenses, much like a seasoned strategist managing their war chest!"
    )
