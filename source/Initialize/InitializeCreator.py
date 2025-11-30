"""Class containing equations."""

import geppy as gep
from deap import creator, base

def initialize_creator():
    """Calculate the travel time between two points including wait time at the end node."""
    creator.create("FitnessMin", base.Fitness, weights=(-1,))  # to minimize the objective (fitness)
    creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)
    return creator