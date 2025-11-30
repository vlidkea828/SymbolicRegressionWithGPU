import datetime
import multiprocess as mp
import cupy as cp
import numpy as np
import time
import os

from sympy import *

import random
import geppy as gep
from deap import tools, creator

from .Utilities.ExtraEquations import (
    add_all_with_health
)
from .Initialize.InitializeStats import initialize_stats
from .Utilities import globals
from memory_profiler import profile
from sys import getsizeof

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

def my_map(evaluation_method, total):
        # start = "start"
        for individual in total:
            individual.fitness.values = (evaluation_method(individual),)

            # time.sleep(0.01)
        # end = "end"
   
def _compile_cupy_gene(g, replacement1, replacement2):
    """
    Compile one gene *g* formated as a cupy equation.
    :return: a string representing a cupy equation
    """
    code = str(g)
    code = code.replace('X', replacement1)
    code = code.replace('Y', replacement2)
    code = code.replace('sin', 'fsin')
    code = code.replace('cos', 'fcos')
    code = code.replace('tan', 'ftan')

    # code = code.replace('add', 'fadd')
    # code = code.replace('sub', 'fsub')
    # code = code.replace('mul', 'fmul')
    return code

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

        equation_string = self.make_equation_string(individual)

        new_evaluate_func = self.make_new_evaluate_func(equation_string)

        Rp = self.get_results(new_evaluate_func)

        # TODO: DO IT HERE
        coefficient_of_determination = self.get_stats_results(Rp)
        
        return cp.asnumpy(coefficient_of_determination)

    def get_stats_results(self, Rp):
        sum_squared_regression = cp.sum(squared_difference(self.results, Rp))
        coefficient_of_determination = coefficient_of_determination_equation(
            sum_squared_regression,
            self.total_sum_of_squares
        )
        
        return coefficient_of_determination

    def get_results(self, new_evaluate_func):
        Rp = new_evaluate_func(
            self.inputs[0],
            self.inputs[1],
            self.inputs[2],
            self.inputs[3],
            self.inputs[4],
            self.inputs[5]
        )
        # print(listdir('/home/vlidkea/.cupy/kernel_cache'))
        
        return Rp

    def make_new_evaluate_func(self, equation_string):
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
            # 'current_evaluation')
            options=("-std=c++11",)
            ,loop_prep="double delta { 2.0 / ( _ind.size() - 1 ) }; \
               double start { -1.0 + delta };",)
               
        return new_evaluate_func

    def make_equation_string(self, individual):
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
        
        return equation_string

def test_geppy_final_special_two():
    
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    pset = globals.pset
    head_length=4
    rnc_array_length=10
    number_of_genes_in_chromosome=4
    tournament=14
    number_of_migrants = 3
    k_migrants=number_of_migrants
    
    toolbox = gep.Toolbox()
    toolbox.register('rnc_gen', random.randint, a=-1000, b=1000)   # each RNC is random integer within [-5, 5]
    toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=head_length, rnc_gen=toolbox.rnc_gen, rnc_array_length=rnc_array_length)
    
    toolbox.register('select', tools.selTournament, tournsize=tournament)
    toolbox.register("migrate", tools.migRing, k=k_migrants, selection=tools.selBest, replacement=tools.selWorst)    # testing an alternative idea to the   replacement=random.sample  code I've seen
    # compile utility: which translates an individual into an executable function (Lambda)
    toolbox.register('compile', gep.compile_, pset=pset)
    test_evaluator = newest_evaluate_test_2(
        pset=globals.pset,
        inputs=globals.inputs,
        results=globals.results)
    toolbox.register('evaluate', test_evaluator.newest_evaluate)
    toolbox.register('select', tools.selTournament, tournsize=3)
    # 1. general operators
    toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
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
    linker_method=add_all_with_health
    population_size=30
    number_of_generations=10
    number_of_retained_champions_accross_generations=5
    number_of_islands=4
    number_of_elites=4
    FREQ=20

    startDT = datetime.datetime.now()
    print (str(startDT))

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
    toolbox.register("map", my_map)
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
        hof,
        test_evaluator)

    print(log[-1]["max R^2"])
    if np.isnan(log[-1]["max R^2"]):
        print("why______________")

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

def GenerationsMethod(mempool, pinned_mempool, k_migrants, toolbox, stats, population_size, FREQ, number_islands, n_gen, num_elites, hof, test_evaluator):
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
        GetFirstGeneration(demes, toolbox, log, hof, stats)

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
                    gen,
                    test_evaluator)
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

def GetFirstGeneration(demes, toolbox, log, hof, stats):
    """"""
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

def OneGeneration(mempool, pinned_mempool, toolbox, stats, num_elites, hof, log, idx, deme, gen, test_evaluator):
    invalid_ind = get_invalid_ind(toolbox, num_elites, deme)

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

def get_invalid_ind(toolbox, num_elites, deme):
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
    return invalid_ind
    # elites  = None
    # offspring = None
    
