"""Class containing equations."""
import random
import geppy as gep
import multiprocess as mp
from deap import tools

from ..Evaluation.EvaluationMethods import (
    EvaluationMethod,
    newest_evaluate
)


def test_toolbox(
    pset,
    inputs,
    results,
    mapping_function,
    evaluation_model,
    pool,
    head_length : int,
    rnc_array_length : int,
    enable_linear_scaling : bool,
    tournament : int,
    k_migrants : int,
    special_run : bool = False,
):
    """Class containing equations."""

    toolbox = gep.Toolbox()
    toolbox.register('rnc_gen', random.randint, a=-1000, b=1000)   # each RNC is random integer within [-5, 5]
    toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=head_length, rnc_gen=toolbox.rnc_gen, rnc_array_length=rnc_array_length)
    toolbox.register("map", pool.map)
    toolbox.register('select', tools.selTournament, tournsize=tournament)
    toolbox.register("migrate", tools.migRing, k=k_migrants, selection=tools.selBest, replacement=tools.selWorst)    # testing an alternative idea to the   replacement=random.sample  code I've seen
    # compile utility: which translates an individual into an executable function (Lambda)
    toolbox.register('compile', gep.compile_, pset=pset)

    evaluationMethod = EvaluationMethod(
        toolbox=toolbox,
        inputs=inputs,
        results=results,
        mapping_function=mapping_function,
        evaluation_model=evaluation_model,
        pset=pset
    )
    if special_run:
        toolbox.register('evaluate', newest_evaluate)
    else:
        if enable_linear_scaling:
            toolbox.register('evaluate', evaluationMethod.evaluate_ls)
        else:
            toolbox.register('evaluate', evaluationMethod.evaluate)

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
    return toolbox
