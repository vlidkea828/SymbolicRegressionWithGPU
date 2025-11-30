"""Class containing equations."""

import math
import operator 
import geppy as gep

from ..Utilities.ExtraEquations import protected_div

def make_pset(primitive_set_name, input_names):
    """Calculate the travel time between two points including wait time at the end node."""
    pset = gep.PrimitiveSet(name=primitive_set_name, input_names=input_names)
    pset.add_function(operator.add, 2)
    pset.add_function(operator.sub, 2)
    pset.add_function(operator.mul, 2)
    pset.add_function(protected_div, 2)
    # pset.add_function(math.sin, 1)        # I tested adding my own functions
    # pset.add_function(math.cos, 1)
    # pset.add_function(math.tan, 1)
    pset.add_rnc_terminal()

    return pset
