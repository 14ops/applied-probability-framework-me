"""
Character Strategies Package

Comprehensive character-based strategies for High-RTP games.
Each strategy embodies a unique philosophy and approach.
"""

from .takeshi_kovacs import TakeshiKovacsStrategy
from .lelouch_vi_britannia import LelouchViBritanniaStrategy
from .kazuya_kinoshita import KazuyaKinoshitaStrategy
from .senku_ishigami import SenkuIshigamiStrategy
from .rintaro_okabe import RintaroOkabeStrategy
from .hybrid_strategy import HybridStrategy

__all__ = [
    'TakeshiKovacsStrategy',
    'LelouchViBritanniaStrategy',
    'KazuyaKinoshitaStrategy',
    'SenkuIshigamiStrategy',
    'RintaroOkabeStrategy',
    'HybridStrategy',
]

