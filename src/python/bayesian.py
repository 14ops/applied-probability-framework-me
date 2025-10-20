from math import isfinite
from scipy.stats import beta


class BetaEstimator:
    def __init__(self, a=1.0, b=1.0):
        self.a = a
        self.b = b

    def update(self, wins, losses):
        self.a += wins
        self.b += losses

    def mean(self):
        return self.a / (self.a + self.b)

    def ci(self, alpha=0.05):
        return beta.ppf(alpha / 2, self.a, self.b), beta.ppf(
            1 - alpha / 2, self.a, self.b
        )


def fractional_kelly(p, b, f=0.25):
    # p: win prob, b: net odds (payout - 1)
    if b <= 0:
        return 0.0
    k = (p * (b + 1) - 1) / b
    return max(0.0, f * k)
