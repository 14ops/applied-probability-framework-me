
'''
Main entry point for the Applied Probability and Automation Framework.
Author: Shihab Belal
'''

# Core modules
from . import game_simulator
from . import strategies
from . import utils

# Advanced strategy modules
from . import advanced_strategies
from . import rintaro_okabe_strategy
from . import monte_carlo_tree_search
from . import markov_decision_process
from . import strategy_auto_evolution
from . import confidence_weighted_ensembles

# AI and Machine Learning components (optional at runtime)
try:
    from . import drl_environment  # Reinforcement learning environment
except ModuleNotFoundError:  # Optional dependency may be missing during setup
    drl_environment = None

try:
    from . import drl_agent  # Q-learning / RL agents
except ModuleNotFoundError:
    drl_agent = None

try:
    from . import bayesian_mines  # Bayesian inference components
except ModuleNotFoundError:
    bayesian_mines = None

try:
    from . import human_data_collector  # Data pipelines
except ModuleNotFoundError:
    human_data_collector = None

try:
    from . import adversarial_agent
    from . import adversarial_detector
    from . import adversarial_trainer
except ModuleNotFoundError:
    adversarial_agent = None
    adversarial_detector = None
    adversarial_trainer = None

# Multi-Agent System components
from . import multi_agent_core
from . import agent_comms
from . import multi_agent_simulator

# Behavioral Economics integration
from . import behavioral_value
from . import behavioral_probability

# Visualization and Analytics (optional at runtime)
try:
    from visualizations import (
        visualization,
        advanced_visualizations,
        comprehensive_visualizations,
        specialized_visualizations,
        realtime_heatmaps,
    )
except ModuleNotFoundError:
    visualization = None
    advanced_visualizations = None
    comprehensive_visualizations = None
    specialized_visualizations = None
    realtime_heatmaps = None

# Testing modules
#from .. import drl_evaluation
#from .. import ab_testing_framework

def main():
    """
    Main function to run the framework.
    """
    print("Applied Probability and Automation Framework")
    print("Core modules loaded.")
    if drl_agent is None:
        print("[warn] Optional ML components not available (install requirements).")
    if visualization is None:
        print("[warn] Visualization modules not available (install plotting libs).")


# Minimal FastAPI app exposing agents and progress
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import glob
    import json
    import os
    from typing import Dict, List, Any
except Exception:
    FastAPI = None  # type: ignore


def _load_session_logs(log_dir: str) -> list:
    if not os.path.isdir(log_dir):
        return []
    sessions: list = []
    for path in sorted(glob.glob(os.path.join(log_dir, "session_log_*.json"))):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    sessions.extend(data)
        except Exception:
            continue
    return sessions


def _aggregate_progress_from_logs(sessions: list) -> dict:
    by_session = []
    wins = 0
    total = 0
    avg_payouts = []
    for s in sessions:
        total += 1
        if s.get("win"):
            wins += 1
        if "final_payout" in s:
            try:
                avg_payouts.append(float(s["final_payout"]))
            except Exception:
                pass
        by_session.append({
            "session_id": s.get("session_id"),
            "win": s.get("win"),
            "final_payout": s.get("final_payout", 0.0),
        })

    win_rate = (wins / total) if total else 0.0
    avg_payout = (sum(avg_payouts) / len(avg_payouts)) if avg_payouts else 0.0
    return {
        "summary": {"total_sessions": total, "win_rate": win_rate, "avg_payout": avg_payout},
        "sessions": by_session,
    }


def _load_ai_improvements(log_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(log_dir, "ai_improvements.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def _aggregate_improvements(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_strategy: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for r in records:
        name = r.get("strategy") or r.get("name") or "Unknown"
        if name not in by_strategy:
            by_strategy[name] = {"series": [], "wins": 0, "losses": 0, "avg_payouts": []}
            order.append(name)
        entry = by_strategy[name]
        entry["series"].append({
            "t": r.get("timestamp"),
            "win": r.get("win"),
            "payout": r.get("payout", 0),
            "win_rate": r.get("win_rate"),
            "avg_payout": r.get("avg_payout"),
            "ev": r.get("ev"),
        })
        if r.get("win"):
            entry["wins"] += 1
            if "payout" in r:
                entry["avg_payouts"].append(float(r["payout"]))
        else:
            entry["losses"] += 1
    summary: Dict[str, Any] = {}
    for name in order:
        e = by_strategy[name]
        total = e["wins"] + e["losses"]
        win_rate = (e["wins"] / total) if total else 0.0
        avg_payout = (sum(e["avg_payouts"]) / len(e["avg_payouts"])) if e["avg_payouts"] else 0.0
        summary[name] = {"win_rate": win_rate, "avg_payout": avg_payout, "games": total}
    return {"strategies": list(order), "by_strategy": by_strategy, "summary": summary}


app = None
if FastAPI is not None:
    app = FastAPI(title="Agents Progress API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/agents")
    def list_agents():
        try:
            # Prefer real core if available
            agents = []
            try:
                core = multi_agent_core.MultiAgentCore(num_agents=10)
                agents = core.get_agents()
            except Exception:
                agents = [f"Agent_{i}" for i in range(10)]
            return {"agents": agents}
        except Exception as e:
            return {"agents": [], "error": str(e)}

    @app.get("/api/progress")
    def get_progress():
        try:
            sessions = _load_session_logs(os.path.join(os.path.dirname(__file__), "logs"))
            agg = _aggregate_progress_from_logs(sessions)
            return agg
        except Exception as e:
            return {"summary": {}, "sessions": [], "error": str(e)}

    @app.get("/api/improvements")
    def get_improvements():
        try:
            records = _load_ai_improvements(os.path.join(os.path.dirname(__file__), "logs"))
            return _aggregate_improvements(records)
        except Exception as e:
            return {"strategies": [], "by_strategy": {}, "summary": {}, "error": str(e)}


if __name__ == "__main__":
    main()
