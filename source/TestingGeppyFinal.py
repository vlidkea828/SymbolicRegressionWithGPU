import datetime
import multiprocess as mp
import cupy as cp
# from cupy import cuda
import math
import operator 
import numpy as np
import re as regex
import time
import importlib

import torch
from GPUtil import showUtilization as gpu_usage
import numba

from sympy import *
from sympy.abc import X, Y

import random
import geppy as gep
from deap import tools, creator

from .Evaluation.EvaluationMethods import (
    EvaluationMethod
)

from .Utilities.ExtraEquations import (
    new_attacker_value,
    add_all,
    old_health_value
)

from .Utilities.CupyCalcRecursion import (
    get_results,
    get_final_results,
    get_final_results_2,
    get_equation_string
)

from .Utilities.ExtraEquations import protected_div

from .Utilities.ExtraEquations import (
    add_all_no_health,
    add_all_with_health,
    add_all_with_health_and_level,
    add_all_with_health_and_level_ind_mult,
    add_all_with_health_and_level_ind_mult_add
)
from .Initialize.InitializeStats import initialize_stats
from .Evaluation.EvaluationMethods import (
    EvaluationModelAllValuesWithHealth
)
from .Utilities import globals
from memory_profiler import profile
from sys import getsizeof
from source.Utilities.globals import init

@cp.fuse()
def fadd(X, Y):
    return cp.add(X, Y)

@cp.fuse()
def fsub(X, Y):
    return cp.subtract(X, Y)

@cp.fuse()
def fmul(X, Y):
    return cp.multiply(X, Y)

# @cp.fuse()
# def protected_div(X, Y):
#     """Calculate the travel time between two points including wait time at the end node."""
#     # if cp.less(cp.abs(Y), 0.0000001):
#     #     return cp.divide(X, 0.0000001)
#     return cp.divide(X, Y)

@cp.fuse()
def fsin(X):
    """ """
    return cp.sin(X)

@cp.fuse()
def fcos(X):
    """ """
    return cp.cos(X)

@cp.fuse()
def ftan(X):
    """ """
    return cp.tan(X)

# add some helper functions (sourced from gep.simple to use explicitly in our islands processing
def gep_apply_modification(population, operator, pb):
    """
    Apply the modification given by *operator* to each individual in *population* with probability *pb* in place.
    """
    for i in range(len(population)):
        if random.random() < pb:
            population[i], = operator(population[i])
            del population[i].fitness.values
    return population


def gep_apply_crossover(population, operator, pb):
    """
    Mate the *population* in place using *operator* with probability *pb*.
    """
    for i in range(1, len(population), 2):
        if random.random() < pb:
            population[i - 1], population[i] = operator(population[i - 1], population[i])
            del population[i - 1].fitness.values
            del population[i].fitness.values
    return population

class test_map:
    def __init__(self, pool):
        self.pool = pool

    def map(self, evaluation_method, total):
        # start = "start"
        # manager = mp.Manager()
        # return_dict = manager.dict()
        # jobs = []
        # for index, individual in enumerate(total):
        #     p = mp.Process(
        #         target=evaluation_method, args=(individual, return_dict, index))
        #     # jobs.append(p)
        #     p.start()
        #     p.join()
        #     individual.fitness.values = (return_dict,)

        # for proc in jobs:
        #     proc.join()
        # for index, results in enumerate(return_dict.values()):
        #     total[index].fitness.values = (results,)
        # tyest = mp.Pool(1)
        # tyest
        for individual in total:
            individual.fitness.values = (evaluation_method(individual),)

            # time.sleep(0.01)
        # end = "end"
            
def my_map(evaluation_method, total):
        # start = "start"
        # manager = mp.Manager()
        # return_dict = manager.dict()
        # jobs = []
        # for index, individual in enumerate(total):
        #     p = mp.Process(
        #         target=evaluation_method, args=(individual, return_dict, index))
        #     # jobs.append(p)
        #     p.start()
        #     p.join()
        #     individual.fitness.values = (return_dict,)

        # for proc in jobs:
        #     proc.join()
        # for index, results in enumerate(return_dict.values()):
        #     total[index].fitness.values = (results,)
        # tyest = mp.Pool(1)
        # tyest
        for individual in total:
            individual.fitness.values = (evaluation_method(individual),)

            # time.sleep(0.01)
        # end = "end"

def mapping_function(func, inputs):
    return map(
        func,
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5])

def _compile_gene(g, pset):
    """
    Compile one gene *g* with the primitive set *pset*.
    :return: a function or an evaluated result
    """
    code = str(g)
    if len(pset.input_names) > 0:   # form a Lambda function
        args = 'T {}'.format(', T '.join(pset.input_names))
        newCode = cp.ElementwiseKernel(
            args,
            'T z',
            'z = {}'.format(code),
            'new_func')
        code =  'lambda {}: {}'.format(args, code)
    # evaluate the code
    try:
        return newCode
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("The expression tree generated by GEP is too deep. Python cannot evaluate a tree higher "
                          "than 90. You should try to adopt a smaller head length for the genes, for example, by using"
                          "more genes in a chromosome.").with_traceback(traceback)
    
def _compile_cupy_gene(g, replacement1, replacement2):
    """
    Compile one gene *g* formated as a cupy equation.
    :return: a string representing a cupy equation
    """
    code = str(g)

    code = code.replace('X', replacement1)
    code = code.replace('Y', replacement2)

    # code = code.replace('sin', 'fsin')
    # code = code.replace('cos', 'fcos')
    # code = code.replace('tan', 'ftan')

    # code = code.replace('add', 'fadd')
    # code = code.replace('sub', 'fsub')
    # code = code.replace('mul', 'fmul')
    return code

def _compile_numba_gene(g, replacement1, replacement2):
    """
    Compile one gene *g* formated as a cupy equation.
    :return: a string representing a cupy equation
    """
    code = str(g)
    if 'X' in code or 'Y' in code:
        code = code.replace('X', replacement1)
        code = code.replace('Y', replacement2)
    else:
        return '', True

    # code = code.replace('sin', 'fsin')
    # code = code.replace('cos', 'fcos')
    # code = code.replace('tan', 'ftan')

    # code = code.replace('add', 'fadd')
    # code = code.replace('sub', 'fsub')
    # code = code.replace('mul', 'fmul')
    return get_equation_string(code), False

def _compile_sympy_gene(g):
    """
    Compile one gene *g* formated as a cupy equation.
    :return: a string representing a cupy equation
    """
    code = str(g)
    if 'X' in code or 'Y' in code:
        return get_equation_string(code), False
    else:
        return '', True

def _sub_divide_and_replace_equation(function_name, values):
    m = regex.search(r"\[([A-Za-z0-9_]+)\]", values)

squared_difference = cp.ElementwiseKernel(
    'T x, T y',
    'T z',
    'z = (x - y) * (x - y)',
    'squared_difference')

coefficient_of_determination_equation = cp.ElementwiseKernel(
    'T x, T y',
    'T z',
    'z = 1 - (x / y)',
    'coefficient_of_determination_equation')

add = cp.ElementwiseKernel(
    'T x, T y',
    'T z',
    'z = x + y',
    'add')

protected_div = cp.ElementwiseKernel(
    'T x, T y',
    'T z',
    '''
        if (y < 0.0000001){
            z = x / 0.0000006;
        }
        else{
            z = x / y;
        }
    ''',
    'protected_div')

class newest_evaluate_test_1:
    def __init__(self, pset, inputs, results):
        self.pset = pset
        self.inputs = inputs
        self.results = results
        self.total_sum_of_squares = cp.sum(
            squared_difference(results, cp.mean(results)))

    def newest_evaluate(self, individual):
        """
        First apply linear scaling (ls) to the individual 
        and then evaluate its fitness: MSE (mean squared error)
        """
        attack_string = _compile_cupy_gene(individual[0], 'A', 'AL')
        attacker_level_string = _compile_cupy_gene(individual[1], 'AL', 'AL')
        power_string = _compile_cupy_gene(individual[2], 'P', 'P')
        defense_string = _compile_cupy_gene(individual[3], 'D', 'DL')
        defender_level_string = _compile_cupy_gene(individual[4], 'DL', 'DL')
        health_string = '(floor((2 * H * DL) / 100) + DL + 10)'
        equation_string = 'z = ({} + {} + {} - {} - {}) / {}'.format(
            attack_string,
            attacker_level_string,
            power_string,
            defense_string,
            defender_level_string,
            health_string
        )

        new_evaluate_func = cp.ElementwiseKernel(
            'float64 A, float64 AL, float64 P, float64 D, float64 DL, float64 H',
            'float64 z',
            """
            auto protected_div = [] (double X,  double Y)
            {{
                if (abs(Y) < 0.0000006)
                {{
                    return X / 0.0000006;
                }} else {{
                    return X / Y;
                }}
            }};
            auto add = [] (double X,  double Y)
            {{
                return X + Y;
            }};
            auto sub = [] (double X,  double Y)
            {{
                return X - Y;
            }};
            auto mul = [] (double X,  double Y)
            {{
                return X * Y;
            }};
            auto fsin = [] (double X)
            {{
                return sin(X);
            }};
            auto fcos = [] (double X)
            {{
                return cos(X);
            }};
            auto ftan = [] (double X)
            {{
                return tan(X);
            }};
            {equation}
            """.format(equation=equation_string),
            'current_evaluation',
            options=("-std=c++11",),
            loop_prep="double delta { 2.0 / ( _ind.size() - 1 ) }; \
               double start { -1.0 + delta };",)

        Rp = new_evaluate_func(
            self.inputs[0],
            self.inputs[1],
            self.inputs[2],
            self.inputs[3],
            self.inputs[4],
            self.inputs[5]
        )

        # TODO: DO IT HERE
        sum_squared_regression = cp.sum(squared_difference(self.results, Rp))
        coefficient_of_determination = coefficient_of_determination_equation(
            sum_squared_regression,
            self.total_sum_of_squares
        )
        
        return coefficient_of_determination
    
class newest_evaluate_test_2:
    def __init__(self, pset, inputs, results):
        self.pset = pset
        self.inputs = inputs
        self.results = results
        self.total_sum_of_squares = cp.sum(
            squared_difference(results, cp.mean(results)))

    def newest_evaluate(self, individual):
        """
        First apply linear scaling (ls) to the individual 
        and then evaluate its fitness: MSE (mean squared error)
        """

        attack_string = _compile_cupy_gene(individual[0], 'A', 'AL')
        power_string = _compile_cupy_gene(individual[1], 'P', 'AL')
        defense_string = _compile_cupy_gene(individual[2], 'D', 'DL')
        health_string = _compile_cupy_gene(individual[3], 'H', 'DL')
        equation_string = 'z = fmin(fmin({} + {} - {}, 1) / {}, 1)'.format(
            attack_string,
            power_string,
            defense_string,
            health_string
        )

        new_evaluate_func = cp.ElementwiseKernel(
            'float64 A, float64 AL, float64 P, float64 D, float64 DL, float64 H',
            'float64 z',
            """
            auto protected_div = [] (double X,  double Y)
            {{
                if (abs(Y) < 0.0000006)
                {{
                    return X / 0.0000006;
                }} else {{
                    return X / Y;
                }}
            }};
            auto add = [] (double X,  double Y)
            {{
                return X + Y;
            }};
            auto sub = [] (double X,  double Y)
            {{
                return X - Y;
            }};
            auto mul = [] (double X,  double Y)
            {{
                return X * Y;
            }};
            auto fsin = [] (double X)
            {{
                return sin(X);
            }};
            auto fcos = [] (double X)
            {{
                return cos(X);
            }};
            auto ftan = [] (double X)
            {{
                return tan(X);
            }};
            auto fmin = [] (double X, double Y)
            {{
                if (X > Y)
                {{
                    return Y;
                }} else {{
                    return X;
                }}
            }};
            {equation}
            """.format(equation=equation_string),
            'current_evaluation',
            options=("-std=c++11",),
            loop_prep="double delta { 2.0 / ( _ind.size() - 1 ) }; \
               double start { -1.0 + delta };",)

        Rp = new_evaluate_func(
            self.inputs[0],
            self.inputs[1],
            self.inputs[2],
            self.inputs[3],
            self.inputs[4],
            self.inputs[5]
        )

        # TODO: DO IT HERE
        sum_squared_regression = cp.sum(squared_difference(self.results, Rp))
        coefficient_of_determination = coefficient_of_determination_equation(
            sum_squared_regression,
            self.total_sum_of_squares
        )
        return cp.asnumpy(coefficient_of_determination)
    
class newest_evaluate_test_3:
    def __init__(self, pset, inputs, results):
        self.pset = pset
        self.inputs = inputs
        self.results = results
        self.total_sum_of_squares = cp.sum(
            squared_difference(results, cp.mean(results)))

    def newest_evaluate(self, individual):
        """
        First apply linear scaling (ls) to the individual 
        and then evaluate its fitness: MSE (mean squared error)
        """

        attack_string = _compile_cupy_gene(individual[0], 'A', 'AL')
        power_string = _compile_cupy_gene(individual[1], 'P', 'AL')
        defense_string = _compile_cupy_gene(individual[2], 'D', 'DL')
        health_string = _compile_cupy_gene(individual[3], 'H', 'DL')
        equation_string = 'z = fmin(protected_div(fmax(sub(add(fabs({}),fabs({})),fabs({})),1),fabs({})),1)'.format(
            attack_string,
            power_string,
            defense_string,
            health_string
        )

        Rp = get_results(equation_string)

        # TODO: DO IT HERE
        sum_squared_regression = cp.sum(squared_difference(self.results, Rp))
        coefficient_of_determination = coefficient_of_determination_equation(
            sum_squared_regression,
            self.total_sum_of_squares
        )
        return cp.asnumpy(coefficient_of_determination)
    
class newest_evaluate_test_4:
    def __init__(self, pset, inputs, results):
        self.pset = pset
        self.inputs = inputs
        self.results = results
        self.total_sum_of_squares = cp.sum(
            squared_difference(results, cp.mean(results)))

    def newest_evaluate(self, individual):
        """
        First apply linear scaling (ls) to the individual 
        and then evaluate its fitness: MSE (mean squared error)
        """

        attack_results = get_results(str(individual[0]), self.inputs[0], self.inputs[1])
        if cp.any(cp.less(attack_results, 0)):
            return 0
        power_results = get_results(str(individual[1]), self.inputs[2], self.inputs[1])
        if cp.any(cp.less(power_results, 0)):
            return 0
        defense_results = get_results(str(individual[2]), self.inputs[3], self.inputs[4])
        if cp.any(cp.less(defense_results, 0)):
            return 0
        health_results = get_results(str(individual[3]), self.inputs[5], self.inputs[4])
        if cp.any(cp.less(health_results, 0)):
            return 0
        if cp.any(cp.greater(health_results, 255)):
            return 0

        Rp = get_final_results(
            attack_results,
            power_results,
            defense_results,
            health_results)

        # TODO: DO IT HERE
        sum_squared_regression = cp.sum(squared_difference(self.results, Rp))
        coefficient_of_determination = coefficient_of_determination_equation(
            sum_squared_regression,
            self.total_sum_of_squares
        )
        return cp.asnumpy(coefficient_of_determination)

class newest_evaluate_test_5:
    def __init__(self, pset, inputs, results):
        self.pset = pset
        self.inputs = inputs
        self.results = results
        self.total_sum_of_squares = cp.sum(
            squared_difference(results, cp.mean(results)))
        self.mempool = cp.get_default_memory_pool()
        self.pinned_mempool = cp.get_default_pinned_memory_pool()

    def newest_evaluate(self, individual):
        """
        First apply linear scaling (ls) to the individual 
        and then evaluate its fitness: MSE (mean squared error)
        """
        attack_results, _ = get_results(str(individual[0]), self.inputs[0], self.inputs[1])
        if cp.any(cp.less(attack_results, 0)):
            return 0
        attacker_level_results, _ = get_results(str(individual[1]), self.inputs[1], self.inputs[1])
        if cp.any(cp.less(attacker_level_results, 0)):
            return 0
        power_results, _ = get_results(str(individual[2]), self.inputs[2], self.inputs[1])
        if cp.any(cp.less(power_results, 0)):
            return 0
        defense_results, _ = get_results(str(individual[3]), self.inputs[3], self.inputs[4])
        if cp.any(cp.less(defense_results, 0)):
            return 0
        defender_level_results, _ = get_results(str(individual[4]), self.inputs[4], self.inputs[4])
        if cp.any(cp.less(defender_level_results, 0)):
            return 0
        health_results, _ = get_results(str(individual[5]), self.inputs[5], self.inputs[4])
        if cp.any(cp.less(health_results, 0)):
            return 0

        Rp = get_final_results(
            attack_results,
            attacker_level_results,
            power_results,
            defense_results,
            defender_level_results,
            health_results)

        # TODO: DO IT HERE
        sum_squared_regression = cp.sum(squared_difference(self.results, Rp))
        coefficient_of_determination = coefficient_of_determination_equation(
            sum_squared_regression,
            self.total_sum_of_squares
        )
        return cp.asnumpy(coefficient_of_determination)

@numba.jit(nopython=True)
def add_all_with_health_numba(A, P, D, H):
    return (A + P - D) / H

@numba.jit(nopython=True)
def squared_difference_numba(x, y):
    return (x - y) * (x - y)

@numba.jit(nopython=True)
def coefficient_of_determination_equation_numba(x, y):
    return 1 - (x / y)

# lambdastrtest = """def calculate_ratios(A, AL, P, D, DL, H, OR): 
#                 AT = {}
#                 if AT.any() < 0:
#                     return -1 * A
#                 PL = {}
#                 if PL.any() < 0:
#                     return -1 * P
#                 DT = {}
#                 if DT.any() < 0:
#                     return -1 * D
#                 HL = {}
#                 if HL.any() < 0:
#                     return -1 * H
#                 R = (AT + PL - DT) / HL
#                 Det = (OR - R) * (OR - R)
#                 return Det
#             """.format(
#             attack_string,
#             power_string,
#             defense_string,
#             health_string
#         )

class newest_evaluate_test_6:
    def __init__(self, pset, inputs, results):
        self.pset = pset
        self.inputs = inputs
        self.attack_input = numba.cuda.to_device(inputs[0])
        self.attacker_level_input = numba.cuda.to_device(inputs[1])
        self.power_input = numba.cuda.to_device(inputs[2])
        self.defense_input = numba.cuda.to_device(inputs[3])
        self.defender_level_input = numba.cuda.to_device(inputs[4])
        self.health_input = numba.cuda.to_device(inputs[5])
        self.results = numba.cuda.to_device(results)
        mean = np.mean(results)
        self.total_sum_of_squares = np.sum(
            np.multiply(
                np.subtract(results, mean),
                np.subtract(results, mean))
        ).item()

    def newest_evaluate(self, individual):
        """
        First apply linear scaling (ls) to the individual 
        and then evaluate its fitness: MSE (mean squared error)
        """
        attack_string, bad = _compile_numba_gene(individual[0], 'A', 'AL')
        if bad:
            return 0
        power_string, bad = _compile_numba_gene(individual[1], 'P', 'AL')
        if bad:
            return 0
        defense_string, bad = _compile_numba_gene(individual[2], 'D', 'DL')
        if bad:
            return 0
        health_string, bad = _compile_numba_gene(individual[3], 'H', 'DL')
        if bad:
            return 0
        lambdastr = """lambda A, AL, P, D, DL, H, OR: (OR - (({attack_string} + {power_string} - {defense_string}) / {health_string})) * (OR - (({attack_string} + {power_string} - {defense_string}) / {health_string}))""".format(
            attack_string=attack_string,
            power_string=power_string,
            defense_string=defense_string,
            health_string=health_string
        )
        lambdafunc = eval(lambdastr)
        numbafunc = numba.jit(nopython=True,parallel=True)(lambdafunc)
        numba_output = numbafunc(
            self.attack_input,
            self.attacker_level_input,
            self.power_input,
            self.defense_input,
            self.defender_level_input,
            self.health_input,
            self.results
        )
        if np.any(np.less(numba_output, 0)):
            return 0
        rp_sum = np.sum(numba_output).item()
        diff = rp_sum / self.total_sum_of_squares
        return 1 - diff
        
        attack_string = str(gep.simplify(individual[0]))
        attacklambdastr = f'lambda X, Y: {attack_string}'
        attacklambdafunc = eval(attacklambdastr)
        attacknumbafunc = numba.jit(nopython=True, parallel=True)(attacklambdafunc)
        attack_results = attacknumbafunc(self.inputs[0], self.inputs[1])
        if np.any(np.less(attack_results, 0)):
            return 0
        
        power_string = str(gep.simplify(individual[1]))
        powerlambdastr = f'lambda X, Y: {power_string}'
        powerlambdafunc = eval(powerlambdastr)
        powernumbafunc = numba.jit(nopython=True, parallel=True)(powerlambdafunc)
        power_results = powernumbafunc(self.inputs[2], self.inputs[1])
        if np.any(np.less(power_results, 0)):
            return 0
        
        defense_string = str(gep.simplify(individual[2]))
        defenselambdastr = f'lambda X, Y: {defense_string}'
        defenselambdafunc = eval(defenselambdastr)
        defensenumbafunc = numba.jit(nopython=True, parallel=True)(defenselambdafunc)
        defense_results = defensenumbafunc(self.inputs[3], self.inputs[4])
        if np.any(np.less(defense_results, 0)):
            return 0

        health_string = str(gep.simplify(individual[3]))
        healthlambdastr = f'lambda X, Y: {health_string}'
        healthlambdafunc = eval(healthlambdastr)
        healthnumbafunc = numba.jit(nopython=True, parallel=True)(healthlambdafunc)
        health_results = healthnumbafunc(self.inputs[5], self.inputs[4])
        if np.any(np.less(health_results, 0)):
            return 0

        Rp = add_all_with_health_numba(
            attack_results,
            power_results,
            defense_results,
            health_results)

        # TODO: DO IT HERE
        sum_squared_regression = np.sum(squared_difference_numba(self.results, Rp))
        coefficient_of_determination = coefficient_of_determination_equation_numba(
            sum_squared_regression,
            self.total_sum_of_squares
        )
        return coefficient_of_determination

class newest_evaluate_test_7:
    def __init__(self, pset, inputs, results):
        self.pset = pset
        self.inputs = inputs
        self.results = results
        self.total_sum_of_squares = cp.sum(
            squared_difference(results, cp.mean(results)))
        self.mempool = cp.get_default_memory_pool()
        self.pinned_mempool = cp.get_default_pinned_memory_pool()

    def newest_evaluate(self, individual):
        """
        First apply linear scaling (ls) to the individual 
        and then evaluate its fitness: MSE (mean squared error)
        """
        attack_string, bad = _compile_sympy_gene(individual[0])
        if bad:
            return 0
        attack_function = lambdify((X, Y), '(' + attack_string + ')', "cupy")
        attack_results = attack_function(self.inputs[0], self.inputs[1])
        if cp.any(cp.less(attack_results, 0)):
            return 0
        power_string, bad = _compile_sympy_gene(individual[1])
        if bad:
            return 0
        power_function = lambdify((X, Y), '(' + power_string + ')', "cupy")
        power_results = power_function(self.inputs[2], self.inputs[1])
        if cp.any(cp.less(power_results, 0)):
            return 0
        defense_string, bad = _compile_sympy_gene(individual[2])
        if bad:
            return 0
        defense_function = lambdify((X, Y), '(' + defense_string + ')', "cupy")
        defense_results = defense_function(self.inputs[3], self.inputs[4])
        if cp.any(cp.less(defense_results, 0)):
            return 0
        health_string, bad = _compile_sympy_gene(individual[3])
        if bad:
            return 0
        health_function = lambdify((X, Y), '(' + health_string + ')', "cupy")
        health_results = health_function(self.inputs[5], self.inputs[4])
        if cp.any(cp.less(health_results, 0)):
            return 0

        Rp = get_final_results(
            attack_results,
            power_results,
            defense_results,
            health_results)

        # TODO: DO IT HERE
        sum_squared_regression = cp.sum(squared_difference(self.results, Rp))
        coefficient_of_determination = coefficient_of_determination_equation(
            sum_squared_regression,
            self.total_sum_of_squares
        )
        # self.mempool.free_all_blocks()
        return cp.asnumpy(coefficient_of_determination)

class newest_evaluate_test_8:
    def __init__(self, pset, inputs, results):
        self.pset = pset
        self.inputs = inputs
        self.results = results
        self.total_sum_of_squares = cp.sum(
            squared_difference(results, cp.mean(results)))
        self.mempool = cp.get_default_memory_pool()
        self.pinned_mempool = cp.get_default_pinned_memory_pool()

    def newest_evaluate(self, individual):
        """
        First apply linear scaling (ls) to the individual 
        and then evaluate its fitness: MSE (mean squared error)
        """
        attack_results, _ = get_results(str(individual[0]), self.inputs[0], self.inputs[1])
        if cp.any(cp.less(attack_results, 0)):
            return 0
        attacker_level_add_results, _ = get_results(str(individual[1]), self.inputs[1], self.inputs[1])
        if cp.any(cp.less(attacker_level_add_results, 0)):
            return 0
        attacker_level_mul_results, _ = get_results(str(individual[2]), self.inputs[1], self.inputs[1])
        if cp.any(cp.less(attacker_level_mul_results, 0)):
            return 0
        power_results, _ = get_results(str(individual[3]), self.inputs[2], self.inputs[1])
        if cp.any(cp.less(power_results, 0)):
            return 0
        defense_results, _ = get_results(str(individual[4]), self.inputs[3], self.inputs[4])
        if cp.any(cp.less(defense_results, 0)):
            return 0
        defender_level_add_results, _ = get_results(str(individual[5]), self.inputs[4], self.inputs[4])
        if cp.any(cp.less(defender_level_add_results, 0)):
            return 0
        defender_level_mul_results, _ = get_results(str(individual[6]), self.inputs[4], self.inputs[4])
        if cp.any(cp.less(defender_level_mul_results, 0)):
            return 0
        health_results, _ = get_results(str(individual[7]), self.inputs[5], self.inputs[4])
        if cp.any(cp.less(health_results, 0)):
            return 0

        Rp = get_final_results_2(
            attack_results,
            attacker_level_add_results,
            attacker_level_mul_results,
            power_results,
            defense_results,
            defender_level_add_results,
            defender_level_mul_results,
            health_results)

        # TODO: DO IT HERE
        sum_squared_regression = cp.sum(squared_difference(self.results, Rp))
        coefficient_of_determination = coefficient_of_determination_equation(
            sum_squared_regression,
            self.total_sum_of_squares
        )
        return cp.asnumpy(coefficient_of_determination)

class newest_new_evaluate_test:
    def __init__(self, pset, inputs, results):
        self.pset = pset
        self.inputs = inputs
        self.results = results
        self.total_sum_of_squares = cp.sum(
            squared_difference(results, cp.mean(results)))

    def newest_evaluate(self, individual):
        """
        First apply linear scaling (ls) to the individual 
        and then evaluate its fitness: MSE (mean squared error)
        """
        attack_string = _compile_cupy_gene(individual[0], 'A', 'AL')
        attacker_level_string = _compile_cupy_gene(individual[1], 'AL', 'AL')
        power_string = _compile_cupy_gene(individual[2], 'P', 'P')
        defense_string = _compile_cupy_gene(individual[3], 'D', 'DL')
        defender_level_string = _compile_cupy_gene(individual[4], 'DL', 'DL')
        health_string = 'fadd(fadd(protected_div(fmul(fmul(2, H), DL), 100), DL), 10)'
        equation_string = 'protected_div(fsub(fsub(fadd(fadd({}, {}), {}), {}), {}), {})'.format(
            attack_string,
            attacker_level_string,
            power_string,
            defense_string,
            defender_level_string,
            health_string
        )

        # attack_result = eval(
        #     attack_string,
        #     {'__builtins__': None},
        #     {
        #         'A' : self.inputs[0],
        #         'AL': self.inputs[1],
        #         'P' : self.inputs[2],
        #         'D' : self.inputs[3],
        #         'DL' : self.inputs[4],
        #         'H' : self.inputs[5],
        #         'fadd' : fadd,
        #         'fsub' : fsub,
        #         'fmul' : fmul,
        #         'protected_div' : protected_div,
        #         'fsin' : fsin,
        #         'fcos' : fcos,
        #         'ftan' : ftan
        #     }
        # )

        # attacker_level_result = eval(
        #     attacker_level_string,
        #     {'__builtins__': None},
        #     {
        #         'A' : self.inputs[0],
        #         'AL': self.inputs[1],
        #         'P' : self.inputs[2],
        #         'D' : self.inputs[3],
        #         'DL' : self.inputs[4],
        #         'H' : self.inputs[5],
        #         'fadd' : fadd,
        #         'fsub' : fsub,
        #         'fmul' : fmul,
        #         'protected_div' : protected_div,
        #         'fsin' : fsin,
        #         'fcos' : fcos,
        #         'ftan' : ftan
        #     }
        # )

        # power_result = eval(
        #     power_string,
        #     {'__builtins__': None},
        #     {
        #         'A' : self.inputs[0],
        #         'AL': self.inputs[1],
        #         'P' : self.inputs[2],
        #         'D' : self.inputs[3],
        #         'DL' : self.inputs[4],
        #         'H' : self.inputs[5],
        #         'fadd' : fadd,
        #         'fsub' : fsub,
        #         'fmul' : fmul,
        #         'protected_div' : protected_div,
        #         'fsin' : fsin,
        #         'fcos' : fcos,
        #         'ftan' : ftan
        #     }
        # )

        # defense_result = eval(
        #     defense_string,
        #     {'__builtins__': None},
        #     {
        #         'A' : self.inputs[0],
        #         'AL': self.inputs[1],
        #         'P' : self.inputs[2],
        #         'D' : self.inputs[3],
        #         'DL' : self.inputs[4],
        #         'H' : self.inputs[5],
        #         'fadd' : fadd,
        #         'fsub' : fsub,
        #         'fmul' : fmul,
        #         'protected_div' : protected_div,
        #         'fsin' : fsin,
        #         'fcos' : fcos,
        #         'ftan' : ftan
        #     }
        # )

        # test = eval(
        #     'protected_div(DL, -19)',
        #     {'__builtins__': None},
        #     {
        #         'A' : self.inputs[0],
        #         'AL': self.inputs[1],
        #         'P' : self.inputs[2],
        #         'D' : self.inputs[3],
        #         'DL' : self.inputs[4],
        #         'H' : self.inputs[5],
        #         'fadd' : fadd,
        #         'fsub' : fsub,
        #         'fmul' : fmul,
        #         'protected_div' : protected_div,
        #         'fsin' : fsin,
        #         'fcos' : fcos,
        #         'ftan' : ftan
        #     }
        # )


        # defender_level_result = eval(
        #     defender_level_string,
        #     {'__builtins__': None},
        #     {
        #         'A' : self.inputs[0],
        #         'AL': self.inputs[1],
        #         'P' : self.inputs[2],
        #         'D' : self.inputs[3],
        #         'DL' : self.inputs[4],
        #         'H' : self.inputs[5],
        #         'fadd' : fadd,
        #         'fsub' : fsub,
        #         'fmul' : fmul,
        #         'protected_div' : protected_div,
        #         'fsin' : fsin,
        #         'fcos' : fcos,
        #         'ftan' : ftan
        #     }
        # )

        # health_result = eval(
        #     health_string,
        #     {'__builtins__': None},
        #     {
        #         'A' : self.inputs[0],
        #         'AL': self.inputs[1],
        #         'P' : self.inputs[2],
        #         'D' : self.inputs[3],
        #         'DL' : self.inputs[4],
        #         'H' : self.inputs[5],
        #         'fadd' : fadd,
        #         'fsub' : fsub,
        #         'fmul' : fmul,
        #         'protected_div' : protected_div,
        #         'fsin' : fsin,
        #         'fcos' : fcos,
        #         'ftan' : ftan
        #     }
        # )

        Rp = eval(
            equation_string,
            {'__builtins__': None},
            {
                'A' : self.inputs[0],
                'AL': self.inputs[1],
                'P' : self.inputs[2],
                'D' : self.inputs[3],
                'DL' : self.inputs[4],
                'H' : self.inputs[5],
                'fadd' : fadd,
                'fsub' : fsub,
                'fmul' : fmul,
                'protected_div' : protected_div,
                'fsin' : fsin,
                'fcos' : fcos,
                'ftan' : ftan
            }
        )

        # TODO: DO IT HERE
        sum_squared_regression = cp.sum(squared_difference(self.results, Rp))
        coefficient_of_determination = coefficient_of_determination_equation(
            sum_squared_regression,
            self.total_sum_of_squares
        )

        return coefficient_of_determination

def test_geppy_final_special_two():
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    print(mempool.used_bytes())              # 0
    print(mempool.total_bytes())             # 0
    print(pinned_mempool.n_free_blocks())    # 0    
    # pset = globals.pset
    # inputs = globals.inputs
    # results = globals.results
    # evaluation_model=EvaluationModelAllValuesWithHealth()
    # mapping_function=lambda func, inputs : map(
    #     func,
    #     inputs[0],
    #     inputs[1],
    #     inputs[2],
    #     inputs[3],
    #     inputs[4],
    #     inputs[5])
    head_length=12
    rnc_array_length=25
    number_of_genes_in_chromosome=8
    tournament=20
    number_of_migrants = 7
    k_migrants=number_of_migrants
    # special_run=True

    # enable_linear_scaling = True
    
    toolbox = gep.Toolbox()
    toolbox.register('rnc_gen', random.randint, a=-1000, b=1000)   # each RNC is random integer within [-5, 5]
    toolbox.register('gene_gen', gep.GeneDc, pset=globals.pset, head_length=head_length, rnc_gen=toolbox.rnc_gen, rnc_array_length=rnc_array_length)
    
    toolbox.register('select', tools.selTournament, tournsize=tournament)
    toolbox.register("migrate", tools.migRing, k=k_migrants, selection=tools.selBest, replacement=tools.selWorst)    # testing an alternative idea to the   replacement=random.sample  code I've seen
    # compile utility: which translates an individual into an executable function (Lambda)
    toolbox.register('compile', gep.compile_, pset=globals.pset)

    test_evaluator = newest_evaluate_test_8(
        pset=globals.pset,
        inputs=globals.inputs,
        results=globals.results)
    toolbox.register('evaluate', test_evaluator.newest_evaluate)
    # evaluationMethod = EvaluationMethod(
    #     toolbox=toolbox,
    #     inputs=inputs,
    #     results=results,
    #     mapping_function=mapping_function,
    #     evaluation_model=evaluation_model,
    #     pset=pset
    # )
    # test_evaluator = newest_evaluate_test_2(
    #     pset=globals.pset,
    #     inputs=globals.inputs,
    #     results=globals.results)
    # if special_run:
    #     toolbox.register('evaluate', test_evaluator.newest_evaluate)
    # else:
    #     if enable_linear_scaling:
    #         toolbox.register('evaluate', evaluationMethod.evaluate_ls)
    #     else:
    #         toolbox.register('evaluate', evaluationMethod.evaluate)

    toolbox.register('select', tools.selTournament, tournsize=3)
    # 1. general operators
    toolbox.register('mut_uniform', gep.mutate_uniform, pset=globals.pset, ind_pb=0.05, pb=1)
    toolbox.register('mut_invert', gep.invert, pb=0.1)
    toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
    toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
    toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
    toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
    toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
    toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)
    # 2. Dc-specific operators
    toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
    toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.1)
    toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)
    # for some uniform mutations, we can also assign the ind_pb a string to indicate our expected number of point mutations in an individual
    toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='0.5p')
    toolbox.pbs['mut_rnc_array_dc'] = 1  # we can also give the probability via the pbs property

    stats=initialize_stats()
    linker_method=add_all_with_health_and_level_ind_mult_add
    population_size=100
    number_of_generations=1000
    number_of_retained_champions_accross_generations=8
    number_of_islands=6
    number_of_elites=6
    FREQ=100
    # pop = toolbox.population(n=population_size) 
    # only record the best three individuals ever found in all generations
    # hof = tools.HallOfFame(number_of_retained_champions_accross_generations)
    startDT = datetime.datetime.now()
    print (str(startDT))
    # start evolution
    # pop, log = gep.gep_simple(pop, toolbox, n_generations=number_of_generations, n_elites=1,
    #                         stats=stats, hall_of_fame=hof, verbose=True)

    champs=number_of_retained_champions_accross_generations
    number_islands=number_of_islands
    n_gen=number_of_generations
    num_elites=number_of_elites

    ## here we replace gep.simple with a multidemic version where there are population islands with periodic migration

    ##################### Evolve Solution for a single population:
    hof = tools.HallOfFame(champs)

    # if I turn all this into a function with defaults to no islands...
    # here are the driving parameters
    # gep_islands(population_size, toolbox, n_gen, n_elites, hall_of_fame, stats, number_islands, k_migrants, FREQ)
    
    toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=number_of_genes_in_chromosome, linker=linker_method)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    mapper = test_map(
        pool=mp.Pool(1))
    toolbox.register("map", mapper.map)
    log = GenerationsMethod(
        mempool,
        pinned_mempool, 
        k_migrants, 
        toolbox, 
        stats, 
        population_size, 
        FREQ, 
        number_islands, 
        n_gen, 
        num_elites, 
        hof)

    print(log[-1]["max R^2"])
    if np.isnan(log[-1]["max R^2"]):
        print("why______________")


    # do house keeping to close our pool, and record ending timestamp
    mapper.pool.close()

    print (
        "Evolution times were:\n\nStarted:\t", startDT,
        "\nEnded:   \t", str(datetime.datetime.now()))

    print(hof[0])

    # print the best symbolic regression we found:
    best_ind = hof[0]
    symplified_best = gep.simplify(best_ind)

    # if enable_linear_scaling:
    #     symplified_best = best_ind.a * symplified_best + best_ind.b

    key= '''
    Given training examples of

        AT = Atmospheric Temperature (C)
        V = Exhaust Vacuum Speed
        AP = Atmospheric Pressure
        RH = Relative Humidity

    we trained a computer using Genetic Algorithms to predict the 

        PE = Power Output

    Our symbolic regression process found the following equation offers our best prediction:

    '''

    print('\n', key,'\t', str(symplified_best), '\n\nwhich formally is presented as:\n\n')


    init_printing()
    # symplified_best

    # use   str(symplified_best)   to get the string of the symplified model

    # output the top 3 champs
    number_of_retained_champions_accross_generations = 3
    for i in range(number_of_retained_champions_accross_generations):
        ind = hof[i]
        symplified_model = gep.simplify(ind)

        print('\nSymplified best individual {}: '.format(i))
        print(symplified_model)
        print("raw indivudal:")
        print(hof[i])

def GenerationsMethod(mempool, pinned_mempool, k_migrants, toolbox, stats, population_size, FREQ, number_islands, n_gen, num_elites, hof):
    if number_islands == 0:
        # POP_SIZE = population_size
        pop = toolbox.population(n=population_size)
        #pop = tools.selBest(pop, int(0.1 * len(pop))) + tools.selTournament(pop, len(pop) - int(0.1 * len(pop)), tournsize=10)

        # run simple single population evolution 
        # pop, log= gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=num_elites
        #                         ,stats=stats, hall_of_fame=hof, verbose=True)

    ##################### Evolve "multidemic" Solution having subpopulations, with "ring migrations" occuring on each FREQ defined generation
    elif number_islands > 0:
        # ISLAND SETUP AND MIGRATIONS
        # based on an example herre: https://github.com/DEAP/deap/blob/master/examples/ga/onemax_multidemic.py 
        # note:our user parameters called number_islands = number_demes
        # define island ring migration strategy: we only migrate unique individuals: 
        toolbox.register("migrate", tools.migRing, k=k_migrants, selection=tools.selBest, replacement=tools.selWorst)    # testing an alternative idea to the   replacement=random.sample  code I've seen

        # define demes, which are sub-populations, or "islands" of size population_size. Note global population = number_islands*population_size
        demes = [toolbox.population(n=population_size) for _ in range(number_islands)]

        # add extra logging for demes
        log = tools.Logbook()
        log.header = "gen", "deme", "evals", "max R^2"

        # configure demes, and fitness to run in parallel with multi-processing, run generation 0
        for idx, deme in enumerate(demes):
            demewide_ind = [ind for ind in deme]
            toolbox.map(toolbox.evaluate, demewide_ind)
            # OLD ONE
            # fitnesses = toolbox.map(toolbox.evaluate, demewide_ind)
            # between = 'test'
            # for ind, fit in zip(demewide_ind, fitnesses):
            #     ind.fitness.values = (fit.item(),)

            log.record(gen=0, deme=idx, evals=len(deme), **stats.compile(deme))
            hof.update(deme)
            # og at deme level 
            print(log.stream)

        # run Deme based evolution, with migrations each FREQ generations
        gen = 1
        while gen <= n_gen and log[-1]["max R^2"] < 0.99 and not np.isnan(log[-1]["max R^2"]):     # halt if MSE "min" is zero, we've solved the problem
            # for each deme, define mutation and crossover. Consider varOR ... and include reproduction?
            for idx, deme in enumerate(demes):
                OneGeneration(
                    mempool,
                    pinned_mempool, 
                    toolbox, 
                    stats, 
                    num_elites, 
                    hof, 
                    log, 
                    idx, 
                    deme, 
                    gen)
                # log.pop()


            # ISLAND MIGRATION
            # On a pulse of FREQ, force ring migration of individuals across our demes/islands

            if gen > 30 and gen % FREQ == 0 or gen > (n_gen - 10):
                # Standard migRing based Migration Strategy is called here.
                toolbox.migrate(demes)
                print("------------------------migration across islands---------------")

            
            
            # this generation is done, increment and restart next 
            gen += 1
    return log

def OneGeneration(mempool, pinned_mempool, toolbox, stats, num_elites, hof, log, idx, deme, gen):
    
    deme[:] = toolbox.select(deme, len(deme))
                # here I need to dip into geppy and construct my own sequence to evolve the deme
                #---------------------- replace varAnd with the following         
                # find elites to exclude them from mutation/crossover
    elites = tools.selBest(deme, k=num_elites)
                # select the rest as offspring, then mutate and cross them together
    offspring = toolbox.select(deme, len(deme) - num_elites)
    offspring = [toolbox.clone(ind) for ind in offspring]
                #mutation
    for op in toolbox.pbs:
        if op.startswith('mut'):
            offspring = gep_apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])
                # crossover
    for op in toolbox.pbs:
        if op.startswith('cx'):
            offspring = gep_apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])
    deme[:] = elites + offspring
                #---------------------- end of replace varAnd

                # each edit of an individual/population via genetic operators, invalidates fitness, so we need to reset/recalc
    invalid_ind = [ind for ind in deme if not ind.fitness.valid]

                # the below evaluates fitness in parallel within the deme. good!
    toolbox.map(toolbox.evaluate, invalid_ind)
                # OLD ONE
                # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                # between = 'test'
                # for ind, fit in zip(invalid_ind, fitnesses):
                #     ind.fitness.values = (fit.item(),)

                # log progress after fitness evaluations
    log.record(gen=gen, deme=idx, evals=len(deme), **stats.compile(deme))
    hof.update(deme)

                # output progress
    print(log.stream)
    mempool.free_all_blocks()
