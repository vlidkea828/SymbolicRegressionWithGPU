import pandas as pd
import cupy as cp
from ..Utilities.OldEquations import OldEquationSolver
from ..Utilities.Lists import get_levels, get_powers, get_stats


def make_local_ratios():
    attacks = get_stats()
    attacker_levels = get_levels()
    powers = get_powers()
    defenses = get_stats()
    defender_levels = get_levels()
    healths = get_stats()
    attack_array = []
    attacker_Level_array = []
    power_array = []
    defense_array = []
    defender_Level_array = []
    health_array = []
    ratio_array = []

    for attack in attacks:
        for attacker_level in attacker_levels:
            for power in powers:
                attack_array.append(attack)
                attacker_Level_array.append(attacker_level)
                power_array.append(power)
                for defense in defenses:
                    for defender_level in defender_levels:
                        for health in healths:
                            defense_array.append(defense)
                            defender_Level_array.append(defender_level)
                            health_array.append(health)
                            equation_solver = OldEquationSolver(
                                attack=attack,
                                attacker_level=attacker_level,
                                power=power,
                                defense=defense,
                                defender_level=defender_level,
                                health=health
                            )
                            ratio_array.append(equation_solver.evaluate_ratio())
    inputs = []
    inputs.append(cp.array(attack_array))
    inputs.append(cp.array(attacker_Level_array))
    inputs.append(cp.array(power_array))
    inputs.append(cp.array(defense_array))
    inputs.append(cp.array(defender_Level_array))
    inputs.append(cp.array(health_array))
    results = cp.array(ratio_array)
    return inputs, results


def make_ratio_csv():
    attacks = get_stats()
    attacker_levels = get_levels()
    powers = get_powers()
    defenses = get_stats()
    defender_levels = get_levels()
    healths = get_stats()

    data = {
        'Attack' : [],
        'Attacker_Level' : [],
        'Power' : [],
        'Defense' : [],
        'Defender_Level' : [],
        'Health' : [],
        'Ratio' : [] }

    for attack in attacks:
        for attacker_level in attacker_levels:
            for power in powers:
                for defense in defenses:
                    for defender_level in defender_levels:
                        for health in healths:
                            data['Attack'].append(attack)
                            data['Attacker_Level'].append(attacker_level)
                            data['Power'].append(power)
                            data['Defense'].append(defense)
                            data['Defender_Level'].append(defender_level)
                            data['Health'].append(health)
                            equation_solver = OldEquationSolver(
                                attack=attack,
                                attacker_level=attacker_level,
                                power=power,
                                defense=defense,
                                defender_level=defender_level,
                                health=health
                            )
                            data['Ratio'].append(equation_solver.evaluate_ratio())
    df = pd.DataFrame(data)
    df.to_csv('source/CSVData/data/ratios.csv')

@staticmethod
def make_attacker_value_csv():
    attacks = get_stats()
    attacker_levels = get_levels()

    data = {
        'Attack' : [],
        'Attacker_Level' : [],
        'Attacker_Value' : [] }

    for attack in attacks:
        for attacker_level in attacker_levels:
            data['Attack'].append(attack)
            data['Attacker_Level'].append(attacker_level)
            equation_solver = OldEquationSolver(
                attack=attack,
                attacker_level=attacker_level,
                power=0,
                defense=0,
                defender_level=0,
                health=0
            )
            data['Attacker_Value'].append(equation_solver.evaluate_attacker_value())
    df = pd.DataFrame(data)
    df.to_csv('source/CSVData/data/attacker_value.csv')

@staticmethod
def make_attacker_defender_value_csv():
    attacks = get_stats()
    attacker_levels = get_levels()
    defenses = get_stats()
    defender_levels = get_levels()

    data = {
        'Attack' : [],
        'Attacker_Level' : [],
        'Defense' : [],
        'Defender_Level' : [],
        'Attacker_Defender_Value' : [] }

    for attack in attacks:
        for attacker_level in attacker_levels:
            for defense in defenses:
                for defender_level in defender_levels:
                    data['Attack'].append(attack)
                    data['Attacker_Level'].append(attacker_level)
                    data['Defense'].append(defense)
                    data['Defender_Level'].append(defender_level)
                    equation_solver = OldEquationSolver(
                        attack=attack,
                        attacker_level=attacker_level,
                        power=0,
                        defense=defense,
                        defender_level=defender_level,
                        health=0
                    )
                    data['Attacker_Defender_Value'].append(
                        equation_solver.evaluate_attacker_defender_value())
    df = pd.DataFrame(data)
    df.to_csv('source/CSVData/data/attacker_defender_value.csv')

@staticmethod
def make_damage_csv():
    attacks = get_stats()
    attacker_levels = get_levels()
    powers = get_powers()
    defenses = get_stats()
    defender_levels = get_levels()

    data = {
        'Attack' : [],
        'Attacker_Level' : [],
        'Power' : [],
        'Defense' : [],
        'Defender_Level' : [],
        'Damage' : [] }

    for attack in attacks:
        for attacker_level in attacker_levels:
            for power in powers:
                for defense in defenses:
                    for defender_level in defender_levels:
                        data['Attack'].append(attack)
                        data['Attacker_Level'].append(attacker_level)
                        data['Power'].append(power)
                        data['Defense'].append(defense)
                        data['Defender_Level'].append(defender_level)
                        equation_solver = OldEquationSolver(
                            attack=attack,
                            attacker_level=attacker_level,
                            power=power,
                            defense=defense,
                            defender_level=defender_level,
                            health=0
                        )
                        data['Damage'].append(equation_solver.evaluate_damage())
    df = pd.DataFrame(data)
    df.to_csv('source/CSVData/data/damages.csv')
