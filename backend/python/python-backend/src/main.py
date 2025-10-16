
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

# AI and Machine Learning components
from . import drl_environment
from . import drl_agent
from . import bayesian_mines
from . import human_data_collector
from . import adversarial_agent
from . import adversarial_detector
from . import adversarial_trainer

# Multi-Agent System components
from . import multi_agent_core
from . import agent_comms
from . import multi_agent_simulator

# Behavioral Economics integration
from . import behavioral_value
from . import behavioral_probability

# Visualization and Analytics
from ..visualizations import visualization
from ..visualizations import advanced_visualizations
from ..visualizations import comprehensive_visualizations
from ..visualizations import specialized_visualizations
from ..visualizations import realtime_heatmaps

# Testing modules
#from .. import drl_evaluation
#from .. import ab_testing_framework

def main():
    """
    Main function to run the framework.
    """
    print("Applied Probability and Automation Framework")
    print("All modules imported successfully.")

if __name__ == "__main__":
    main()
