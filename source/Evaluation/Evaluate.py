"""Class containing equations."""

import geppy as gep
from deap import tools, creator, base

import datetime
import random

from sympy import *

def initialize_creator():
    """Calculate the travel time between two points including wait time at the end node."""
    creator.create("FitnessMin", base.Fitness, weights=(-1,))  # to minimize the objective (fitness)
    creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)
    return creator

def test_evaluate_and_print(
    toolbox,
    stats,
    pool,
    linker_method,
    enable_linear_scaling : bool,
    population_size : int,
    number_of_generations : int,
    number_of_retained_champions_accross_generations : int,
    number_of_islands : int,
    number_of_elites : int,
    number_of_migrants : int,
    FREQ : int,
    number_of_genes_in_chromosome : int
):
    """Calculate the travel time between two points including wait time at the end node."""

    # pop = toolbox.population(n=population_size) 
    # only record the best three individuals ever found in all generations
    # hof = tools.HallOfFame(number_of_retained_champions_accross_generations)
    startDT = datetime.datetime.now()
    print (str(startDT))
    # start evolution
    # pop, log = gep.gep_simple(pop, toolbox, n_generations=number_of_generations, n_elites=1,
    #                         stats=stats, hall_of_fame=hof, verbose=True)
    hof = testing_extra_stuff(
        toolbox=toolbox,
        stats=stats,
        pool=pool,
        linker_method=linker_method,
        champs=number_of_retained_champions_accross_generations,
        number_islands=number_of_islands,
        population_size=population_size,
        n_gen=number_of_generations,
        num_elites=number_of_elites,
        k_migrants=number_of_migrants,
        FREQ=FREQ,
        number_of_genes_in_chromosome=number_of_genes_in_chromosome
    )

    print (
        "Evolution times were:\n\nStarted:\t", startDT,
        "\nEnded:   \t", str(datetime.datetime.now()))

    print(hof[0])

    # print the best symbolic regression we found:
    best_ind = hof[0]
    # symplified_best = gep.simplify(best_ind)

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

    # print('\n', key,'\t', str(symplified_best), '\n\nwhich formally is presented as:\n\n')


    init_printing()
    # symplified_best

    # use   str(symplified_best)   to get the string of the symplified model

    # output the top 3 champs
    number_of_retained_champions_accross_generations = 3
    for i in range(number_of_retained_champions_accross_generations):
        ind = hof[i]
        # symplified_model = gep.simplify(ind)

        print('\nSymplified best individual {}: '.format(i))
        # print(symplified_model)
        print("raw indivudal:")
        print(hof[i])

    # return symplified_best

def testing_extra_stuff(
        toolbox,
        stats,
        pool,
        linker_method,
        champs : int,
        number_islands : int,
        population_size : int,
        n_gen : int,
        num_elites : int,
        k_migrants : int,
        FREQ : int,
        number_of_genes_in_chromosome : int,
):

    ## here we replace gep.simple with a multidemic version where there are population islands with periodic migration

    ##################### Evolve Solution for a single population:
    hof = tools.HallOfFame(champs)

    # if I turn all this into a function with defaults to no islands...
    # here are the driving parameters
    # gep_islands(population_size, toolbox, n_gen, n_elites, hall_of_fame, stats, number_islands, k_migrants, FREQ)
    
    toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=number_of_genes_in_chromosome, linker=linker_method)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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
        log.header = "gen", "deme", "evals", "min MSE"

        # configure demes, and fitness to run in parallel with multi-processing, run generation 0
        for idx, deme in enumerate(demes):
            demewide_ind = [ind for ind in deme]
            fitnesses = toolbox.map(toolbox.evaluate, demewide_ind)
            for ind, fit in zip(demewide_ind, fitnesses):
                ind.fitness.values = fit

            log.record(gen=0, deme=idx, evals=len(deme), **stats.compile(deme))
            hof.update(deme)
            # og at deme level 
            print(log.stream)

        # run Deme based evolution, with migrations each FREQ generations
        gen = 1
        while gen <= n_gen and log[-1]["min MSE"] > 0:     # halt if MSE "min" is zero, we've solved the problem

            # for each deme, define mutation and crossover. Consider varOR ... and include reproduction?
            for idx, deme in enumerate(demes):
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
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # log progress after fitness evaluations
                log.record(gen=gen, deme=idx, evals=len(deme), **stats.compile(deme))
                hof.update(deme)

                # output progress
                print(log.stream)

            # ISLAND MIGRATION
            # On a pulse of FREQ, force ring migration of individuals across our demes/islands

            if gen > 30 and gen % FREQ == 0 or gen > (n_gen - 10):
                # Standard migRing based Migration Strategy is called here.
                toolbox.migrate(demes)
                print("------------------------migration across islands---------------")

            # this generation is done, increment and restart next 
            gen += 1


    # do house keeping to close our pool, and record ending timestamp
    pool.close()
    return hof

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
