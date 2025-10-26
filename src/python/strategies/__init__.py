"""
Strategies Package

Collection of character-based strategies for the Mines game.
Each strategy implements unique betting and decision-making logic.

Available Strategies:
- Takeshi: Aggressive doubling-up with tranquility mode
- Yuzu: Controlled chaos with max 7 clicks
- Aoi: Cooperative sync mechanism
- Kazuya: Conservative with periodic dagger strikes  
- Lelouch: Streak-based strategic betting
"""

from .base import StrategyBase, SimpleRandomStrategy
from .takeshi import TakeshiStrategy
from .yuzu import YuzuStrategy
from .aoi import AoiStrategy
from .kazuya import KazuyaStrategy
from .lelouch import LelouchStrategy

__all__ = [
    'StrategyBase',
    'SimpleRandomStrategy',
    'TakeshiStrategy',
    'YuzuStrategy',
    'AoiStrategy',
    'KazuyaStrategy',
    'LelouchStrategy',
]

