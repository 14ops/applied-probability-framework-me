from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FixedFraction:
    fraction: float = 0.1  # risk fraction of available cash to allocate
    min_shares: int = 1

    def allocate(self, cash: float, price: float) -> Tuple[int, float]:
        alloc = max(0.0, self.fraction) * cash
        shares = int(alloc // price)
        return max(self.min_shares if shares > 0 else 0, shares), alloc


@dataclass
class CappedKelly:
    edge: float = 0.05    # probability edge or expected edge per trade
    odds: float = 1.0     # payoff odds (profit per unit risk)
    cap_fraction: float = 0.25
    min_shares: int = 1

    def kelly_fraction(self) -> float:
        # Standard Kelly: f* = edge / odds for binary bets
        if self.odds <= 0:
            return 0.0
        k = self.edge / self.odds
        return max(0.0, min(self.cap_fraction, k))

    def allocate(self, cash: float, price: float) -> Tuple[int, float]:
        frac = self.kelly_fraction()
        alloc = frac * cash
        shares = int(alloc // price)
        return max(self.min_shares if shares > 0 else 0, shares), alloc


