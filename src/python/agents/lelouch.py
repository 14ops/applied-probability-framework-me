from dataclasses import dataclass, field

from .base import Agent


@dataclass
class Lelouch(Agent):
    name: str = "lelouch"
    weights: dict = field(
        default_factory=lambda: {"senku": 1.0, "takeshi": 1.0, "aoi": 1.0, "yuzu": 1.0}
    )

    def decide(self, *a, **k):
        return ("hold", 0.0, False)

