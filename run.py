"""Main module for code."""
from source.TestingGeppyFinal import (
    test_geppy_final_special_two
)
from deap import creator, base
import geppy as gep
import cupy as cp
import cProfile
import pstats
import multiprocess
from source.Utilities.globals import (
    init,
    init_local
)
from source.Utilities.CupyCalc import (
    get_results
)
# from source.Utilities.CupyCalcAllC import (
#     test_raw_kernal,
#     get_results
# )
from source.Utilities.CupyCalcNumba import (
    test_raw_kernal,
    get_results
)
from source.CSVData.MakeDataSheet import (
    make_ratio_csv
)
import datetime

creator.create("FitnessMax", base.Fitness, weights=(1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMax)

def main():
    """Main runner."""

    # test_geppy_final_special_two()
    cProfile.run('test_geppy_final_special_two()', filename='run_stats')
    p = pstats.Stats('run_stats')
    p.sort_stats('cumulative').print_stats(30)

    # cProfile.run('make_ratio_csv()', filename='run_stats')
    # p = pstats.Stats('run_stats')
    # p.sort_stats('cumulative').print_stats(30)
    # print(str(datetime.datetime.now()))
    # get_results('mul(add(sub(A,AL),cos(1)),add(D,5))')
    # print(str(datetime.datetime.now()))
    

if __name__ == '__main__':
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    print(mempool.used_bytes())              # 0
    print(mempool.total_bytes())             # 0
    print(pinned_mempool.n_free_blocks())    # 0
    init()
    multiprocess.set_start_method("forkserver")
    print(mempool.used_bytes())              # 0
    print(mempool.total_bytes())             # 0
    print(pinned_mempool.n_free_blocks())    # 0
    main()
    # test_raw_kernal()