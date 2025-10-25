import random
try:
    from .bayesian import BetaEstimator, fractional_kelly # Updated import
except ImportError:
    from bayesian import BetaEstimator, fractional_kelly


class BasicStrategy:
    """A basic betting strategy that uses Bayesian estimation for win probability
    and fractional Kelly criterion for bet sizing. It aims for conservative play
    by adjusting the Kelly fraction based on the confidence interval width of the estimate.
    """

    def __init__(self, estimator):
        """Initializes the BasicStrategy with a Bayesian estimator.

        Args:
            estimator (BetaEstimator): An instance of BetaEstimator to provide
                                       win probability and confidence intervals.
        """
        self.name = "BasicStrategy"
        self.estimator = estimator

    def apply(self, simulator, bankroll_manager):
        """Applies the basic strategy to a given game simulator and bankroll manager.

        The strategy calculates a bet size based on estimated win probability and
        fractional Kelly, places the bet, performs a click in the simulator,
        updates its Bayesian estimator based on the outcome, and processes the
        game outcome with the bankroll manager.

        Args:
            simulator (GameSimulator): The game simulator instance to interact with.
            bankroll_manager (MoneyManager): The money manager instance to handle
                                             betting and bankroll updates.

        Returns:
            tuple: A tuple containing the bet amount, estimated win probability,
                   and confidence interval width.
        """
        # Get probability estimate from the Bayesian estimator
        p_win = self.estimator.mean()
        ci_lower, ci_upper = self.estimator.ci()
        ci_width = ci_upper - ci_lower

        # Determine net odds (payout - 1)
        # This is a simplification; a real implementation would get this from the game state
        net_odds = 1.0  # Assuming a 2x payout for simplicity

        # Adjust Kelly fraction based on confidence interval width
        f = 0.25  # Default fraction
        if ci_width > 0.15:
            f = 0.1  # Be more conservative if uncertainty is high

        # Calculate bet size using fractional Kelly
        bet_fraction = fractional_kelly(p_win, net_odds, f)
        bet_amount = (
            bet_fraction * bankroll_manager.bankroll_simulator.get_current_bankroll()
        )
        bet_amount = max(1.0, bet_amount)  # Ensure a minimum bet

        # Place the bet
        bankroll_manager.bankroll_simulator.place_bet(bet_amount)

        # Simulate a click (in a real scenario, this would be a more complex decision)
        unrevealed_cells = []
        for r in range(simulator.board_size):
            for c in range(simulator.board_size):
                if not simulator.revealed[r][c]:
                    unrevealed_cells.append((r, c))

        if unrevealed_cells:
            r, c = simulator.rng.choice(unrevealed_cells)
            result, message = simulator.click_cell(r, c)

            # Update the Bayesian estimator based on the outcome
            if message == "Safe":
                self.estimator.update(is_safe_click=True)
            elif message == "Mine":
                self.estimator.update(is_safe_click=False)

            # Process the game outcome for the bankroll manager
            bankroll_manager.bankroll_simulator.process_outcome(
                simulator.win, simulator.current_payout
            )
        else:
            # No more cells to click, cash out
            simulator.cash_out()
            bankroll_manager.bankroll_simulator.process_outcome(
                simulator.win, simulator.current_payout
            )

        return bet_amount, p_win, ci_width


