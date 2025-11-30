"""Class containing equations."""

from deap import tools
import numpy as np
import cupy as cp

def initialize_stats():
    """Calculate the travel time between two points including wait time at the end node."""
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    # stats.register("avg", np.mean)
    # stats.register("std", np.std)
    # stats.register("min MSE", np.min)
    stats.register("max R^2", np.max)
    return stats