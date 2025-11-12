from .senku import Senku
from .takeshi import Takeshi
from .aoi import Aoi
from .yuzu import Yuzu
from .lelouch import Lelouch


def build_default_agents():
    return {
        "senku": Senku(),
        "takeshi": Takeshi(),
        "aoi": Aoi(),
        "yuzu": Yuzu(),
        "lelouch": Lelouch(),
    }

