"""Main module for code."""
from source.TestingGeppy import (
    test_geppy,
    test_geppy_attacker_value,
    test_geppy_attacker_defender_value,
    test_geppy_special,
    test_geppy_special_two
)
from source.CSVData.MakeDataSheet import (
    make_ratio_csv,
    make_attacker_value_csv,
    make_attacker_defender_value_csv,
    make_damage_csv
)
from source.ExtraTests import (
    attacker_value_test,
    attacker_defender_value_test,
    newest_value_test,
    newest_value_test_two,
    newest_ratio_test
)
from source.TestingGeppyFinal import (
    test_geppy_final_special_two
)
from deap import creator, base
import geppy as gep
import multiprocess
import cupy as cp
import cProfile
import pstats
from source.Utilities.globals import init

creator.create("FitnessMax", base.Fitness, weights=(1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMax)



def main():
    """Main runner."""
    
    # make_ratio_csv()
    # test_geppy()
    
    # make_attacker_value_csv()
    # test_geppy_attacker_value()
    # attacker_value_test()

    # make_attacker_defender_value_csv()
    # test_geppy_attacker_defender_value()
    # attacker_defender_value_test()

    # test_geppy_special()
    # newest_value_test()
    
    # make_damage_csv()
    # test_geppy_special_two()
    # newest_ratio_test()

    # test_geppy_final_special_two()
    cProfile.run('test_geppy_final_special_two()', filename='run_stats')
    p = pstats.Stats('run_stats')
    p.sort_stats('cumulative').print_stats(30)
    

if __name__ == '__main__':

    init()
    multiprocess.set_start_method("forkserver")
    main()