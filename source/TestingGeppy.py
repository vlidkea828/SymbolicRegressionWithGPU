
import operator 
import multiprocess as mp
import cupy
from cupy import cuda

from .Initialize.PSetGenerator import make_pset
from .Utilities.Lists import initial_test_inputs
from .CSVData.GetCSVData import get_csv_training_and_holdout
from .CSVData.SetupTrainingData import (
    test_training_data,
    attacker_value_training_data,
    attacker_defender_value_training_data,
    damage_training_data
)
from .Initialize.InitializeToolbox import test_toolbox
from .Utilities.ExtraEquations import (
    add_all,
    add_all_no_health
)
from .Initialize.InitializeStats import initialize_stats
from .Evaluation.Evaluate import (
    test_evaluate_and_print,
    initialize_creator
)
from .Evaluation.EvaluateResults import test_evaluate_results
from .Evaluation.EvaluationMethods import (
    EvaluationModelTest,
    EvaluationModelAttackerValue,
    EvaluationModelAttackerDefenderValue,
    EvaluationModelAllValues,
    EvaluationModelAllValuesNoHealth,
    EvaluationModelAllValuesWithHealth
)
from .Utilities import globals

@staticmethod
def test_geppy():
    test_inputs = initial_test_inputs()
    train, holdout = get_csv_training_and_holdout(
        csv_name=test_inputs['csv_name'])

    inputs, results = test_training_data(train)


    toolbox = test_toolbox(
        pset=globals.pset,
        creator=initialize_creator(),
        inputs=inputs,
        results=results,
        linker_method=add_all,
        evaluation_model=EvaluationModelTest(),
        mapping_function=test_inputs['mapping_function'],
        head_length=test_inputs['head_length'],
        number_of_genes_in_chromosome=test_inputs['number_of_genes_in_chromosome'],
        rnc_array_length=test_inputs['rnc_array_length'],
        enable_linear_scaling=test_inputs['enable_linear_scaling']
    )

    symplified_best = test_evaluate_and_print(
        toolbox=toolbox,
        stats=initialize_stats(),
        enable_linear_scaling=test_inputs['enable_linear_scaling'],
        population_size=test_inputs['population_size'],
        number_of_generations=test_inputs['number_of_generations'],
        number_of_retained_champions_accross_generations=test_inputs[
            'number_of_retained_champions_accross_generations']
    )

@staticmethod
def test_geppy_attacker_value():
    train, holdout = get_csv_training_and_holdout(
        csv_name='attacker_value.csv')

    inputs, results = attacker_value_training_data(train)

    pset = make_pset(
        primitive_set_name='Main',
        input_names=['A', 'AL'])

    enable_linear_scaling = True

    toolbox = test_toolbox(
        pset=pset,
        creator=initialize_creator(),
        inputs=inputs,
        results=results,
        linker_method=operator.add,
        evaluation_model=EvaluationModelAttackerValue(),
        mapping_function=lambda func, inputs : map(
            func,
            inputs[0],
            inputs[1]),
        head_length=8,
        number_of_genes_in_chromosome=2,
        rnc_array_length=10,
        enable_linear_scaling=enable_linear_scaling
    )

    symplified_best = test_evaluate_and_print(
        toolbox=toolbox,
        stats=initialize_stats(),
        enable_linear_scaling=enable_linear_scaling,
        population_size=120,
        number_of_generations=50,
        number_of_retained_champions_accross_generations=3
    )

@staticmethod
def test_geppy_attacker_defender_value():
    train, holdout = get_csv_training_and_holdout(
        csv_name='attacker_defender_value.csv')

    inputs, results = attacker_defender_value_training_data(train)

    pset = make_pset(
        primitive_set_name='Main',
        input_names=['D', 'DL'])

    enable_linear_scaling = True

    toolbox = test_toolbox(
        pset=pset,
        creator=initialize_creator(),
        inputs=inputs,
        results=results,
        linker_method=operator.add,
        evaluation_model=EvaluationModelAttackerDefenderValue(),
        mapping_function=lambda func, inputs : map(
            func,
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3]),
        head_length=16,
        number_of_genes_in_chromosome=2,
        rnc_array_length=20,
        enable_linear_scaling=enable_linear_scaling
    )

    symplified_best = test_evaluate_and_print(
        toolbox=toolbox,
        stats=initialize_stats(),
        enable_linear_scaling=enable_linear_scaling,
        population_size=200,
        number_of_generations=1000,
        number_of_retained_champions_accross_generations=10
    )

def test_geppy_special():
    train, holdout = get_csv_training_and_holdout(
        csv_name='ratios.csv')

    inputs, results = test_training_data(train)

    pset = make_pset(
        primitive_set_name='Main',
        input_names=['X', 'Y'])

    enable_linear_scaling = True

    toolbox = test_toolbox(
        pset=pset,
        # creator=initialize_creator(),
        inputs=inputs,
        results=results,
        # linker_method=add_all,
        evaluation_model=EvaluationModelAllValues(),
        mapping_function=lambda func, inputs : map(
            func,
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5]),
        head_length=30,
        # number_of_genes_in_chromosome=6,
        rnc_array_length=25,
        enable_linear_scaling=enable_linear_scaling,
        special_run=True
    )

    symplified_best = test_evaluate_and_print(
        toolbox=toolbox,
        stats=initialize_stats(),
        enable_linear_scaling=enable_linear_scaling,
        population_size=500,
        number_of_generations=10000,
        number_of_retained_champions_accross_generations=10
    )

def use_gpu(gpu_id):
    with cuda.Device(gpu_id):
        cupy.asarray(12345)
        print("Using GPU {gpu_id}")

def test_geppy_special_two():
    this_pset = globals.pset
    this_inputs = globals.inputs
    this_results = globals.results
    enable_linear_scaling = True
    procs = 8
    pool = mp.Pool(processes=procs)
    tournoment = 2
    number_of_migrants = 3
    toolbox = test_toolbox(
        pset=this_pset,
        inputs=this_inputs,
        results=this_results,
        evaluation_model=EvaluationModelAllValuesWithHealth(),
        mapping_function=lambda func, inputs : map(
            func,
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5]),
        pool=pool,
        head_length=6,
        rnc_array_length=10,
        enable_linear_scaling=enable_linear_scaling,
        tournament=tournoment,
        k_migrants=number_of_migrants,
        special_run=True
    )

    symplified_best = test_evaluate_and_print(
        toolbox=toolbox,
        stats=initialize_stats(),
        pool=pool,
        linker_method=add_all_no_health,
        enable_linear_scaling=enable_linear_scaling,
        population_size=10,
        number_of_generations=100,
        number_of_retained_champions_accross_generations=3,
        number_of_islands=3,
        number_of_elites=2,
        number_of_migrants=number_of_migrants,
        FREQ=40,
        number_of_genes_in_chromosome=5
    )
