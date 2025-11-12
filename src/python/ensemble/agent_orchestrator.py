"""
Agent orchestration utilities that gather signals for ensemble governance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from src.python.agents.base import PositionState
from src.python.agents.registry import build_default_agents


@dataclass
class AgentSignal:
    agent: str
    action: str
    confidence: float
    is_veto: bool = False


def collect_signals(
    history: pd.DataFrame,
    pos_state: PositionState,
    equity: float,
    context: Dict,
    agents: Dict[str, object] | None = None,
    include_lelouch: bool = False,
) -> List[AgentSignal]:
    """
    Evaluate all registered agents and return their raw signals.
    """
    agent_map = agents or build_default_agents()
    results: List[AgentSignal] = []

    for name, agent in agent_map.items():
        action, conf, is_veto = agent.decide(history, pos_state, equity, context)

        if name == "lelouch" and not include_lelouch:
            continue

        results.append(
            AgentSignal(
                agent=name,
                action=action,
                confidence=float(conf),
                is_veto=bool(is_veto),
            )
        )

    return results


def get_signals(
    history: pd.DataFrame,
    pos_state: PositionState,
    equity: float,
    context: Dict,
) -> List[Tuple[str, float]]:
    """
    Backwards-compatible helper returning action/confidence tuples.
    """
    signals = collect_signals(history, pos_state, equity, context)
    veto = any(sig.agent == "yuzu" and sig.is_veto for sig in signals)

    filtered = [(sig.action, sig.confidence) for sig in signals if sig.agent != "lelouch"]

    if veto:
        return [("hold", 1.0)] * len(filtered)

    return filtered


def get_named_signals(
    history: pd.DataFrame,
    pos_state: PositionState,
    equity: float,
    context: Dict,
) -> List[Tuple[str, str, float]]:
    """
    Convenience wrapper returning (agent, action, confidence) tuples.
    """
    signals = collect_signals(history, pos_state, equity, context)
    return [(sig.agent, sig.action, sig.confidence) for sig in signals]
