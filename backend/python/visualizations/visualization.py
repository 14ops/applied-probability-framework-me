'''
Basic visualization functions for the Applied Probability and Automation Framework.
'''

import matplotlib.pyplot as plt

def plot_simulation_results(data):
    '''
    Plots the results of a simulation.
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title("Simulation Results")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.grid(True)
    plt.savefig("simulation_results.png")
    plt.show()
