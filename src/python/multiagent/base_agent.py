from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    @abstractmethod
    def propose(self, state) -> Dict[str, Any]:
        """Return {'action': int, 'confidence': float, 'reason': str}"""
        ...
