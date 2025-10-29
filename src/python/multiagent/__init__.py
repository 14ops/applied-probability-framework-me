"""Multi-agent coordination package for collaborative mines game strategies."""

from .base_agent import BaseAgent
from .blackboard import Blackboard
from .consensus import majority_vote
from .simulator import TeamSimulator

__all__ = ["BaseAgent", "Blackboard", "majority_vote", "TeamSimulator"]
