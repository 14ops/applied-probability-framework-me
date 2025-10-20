
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

if __name__ == "__main__":
    main()
