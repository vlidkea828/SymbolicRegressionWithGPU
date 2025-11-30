from source.CSVData.SetupTrainingData import (
    test_training_data
)
from source.CSVData.MakeDataSheet import (
    make_local_ratios
)
from source.CSVData.GetCSVData import get_csv_training_and_holdout
from source.Initialize.PSetGenerator import make_pset
import importlib

import cupy as cp
import numpy as np
from numba import cuda

def init():
    """Init al globals"""
    global pset, inputs, results
    pset = make_pset(
        primitive_set_name='Main',
        input_names=['X', 'Y'])
    train, holdout = get_csv_training_and_holdout(
        csv_name='ratios.csv')
    inputs_test, results_test = test_training_data(train)

    inputs = []

    for input_test_array in inputs_test:
        test = cp.array(input_test_array)
        inputs.append(test)
    results = cp.array(results_test)

    # for input_test_array in inputs_test:
    #     inputs.append(np.array(input_test_array))
    # results = np.array(results_test)

    # inputs = inputs_test
    # results = results_test
    test = 1

def init_local():
    """Init all globals locally"""
    global pset, inputs, results
    pset = make_pset(
        primitive_set_name='Main',
        input_names=['X', 'Y'])
    inputs, results = make_local_ratios()
