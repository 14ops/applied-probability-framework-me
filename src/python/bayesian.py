from math import isfinite, sqrt

try:
    from scipy.stats import beta
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class BetaEstimator:
    def __init__(self, a=1.0, b=1.0):
        self.a = a
        self.b = b

    def update(self, wins=0, losses=0, win=None):
        """Update the estimator with wins/losses or a single win/loss."""
        if win is not None:
            if win:
                self.a += 1
            else:
                self.b += 1
        else:
            self.a += wins
            self.b += losses

    def mean(self):
        return self.a / (self.a + self.b)

    def ci(self, alpha=0.05):
        """Compute confidence interval. Falls back to normal approximation if scipy not available."""
        if HAS_SCIPY:
            return beta.ppf(alpha / 2, self.a, self.b), beta.ppf(
                1 - alpha / 2, self.a, self.b
            )
        else:
            # Normal approximation for large samples
            mean = self.mean()
            n = self.a + self.b
            # Standard error of beta distribution
            std_err = sqrt(self.a * self.b / (n * n * (n + 1)))
            z = 1.96  # For 95% CI (alpha=0.05)
            return max(0, mean - z * std_err), min(1, mean + z * std_err)


def fractional_kelly(p, b, f=0.25):
    # p: win prob, b: net odds (payout - 1)
    if b <= 0:
        return 0.0
    k = (p * (b + 1) - 1) / b
    return max(0.0, f * k)
