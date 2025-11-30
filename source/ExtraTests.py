import statistics
from .Utilities.OldEquations import OldEquationSolver
from .Utilities.Lists import (
    get_levels,
    get_stats,
    get_powers
)
from .Utilities.ExtraEquations import (
    new_attacker_value,
    new_defender_value,
    newest_ratio,
    newest_damage,
    current_ratio
)

@staticmethod
def attacker_value_test():
    attacks = get_stats()
    attacker_levels = get_levels()

    results = []

    for attack in attacks:
        for attacker_level in attacker_levels:
            equation_solver = OldEquationSolver(
                attack=attack,
                attacker_level=attacker_level,
                power=0,
                defense=0,
                defender_level=0,
                health=0
            )
            old_result = equation_solver.evaluate_attacker_value()
            new_result = new_attacker_value(attack, attacker_level)
            results.append(old_result / (old_result + abs(old_result - new_result)))
    print(statistics.mean(results))

@staticmethod
def attacker_defender_value_test():
    attacks = get_stats()
    attacker_levels = get_levels()
    defenses = get_stats()
    defender_levels = get_levels()

    results = []

    for attack in attacks:
        for attacker_level in attacker_levels:
            for defense in defenses:
                for defender_level in defender_levels:
                    equation_solver = OldEquationSolver(
                        attack=attack,
                        attacker_level=attacker_level,
                        power=0,
                        defense=defense,
                        defender_level=defender_level,
                        health=0
                    )
                    old_result = equation_solver.evaluate_attacker_defender_value()
                    new_result = new_attacker_value(A=attack, AL=attacker_level) \
                        - new_defender_value(D=defense, DL=defender_level)
                    results.append(old_result / (old_result + abs(old_result - new_result)))
    print(statistics.mean(results))

@staticmethod
def newest_value_test():
    attacks = get_stats()
    attacker_levels = get_levels()
    powers = get_powers()
    defenses = get_stats()
    defender_levels = get_levels()
    healths = get_stats()

    results = []

    for attack in attacks:
        for attacker_level in attacker_levels:
            for power in powers:
                for defense in defenses:
                    for defender_level in defender_levels:
                        for health in healths:
                            equation_solver = OldEquationSolver(
                                attack=attack,
                                attacker_level=attacker_level,
                                power=power,
                                defense=defense,
                                defender_level=defender_level,
                                health=health
                            )
                            old_result = equation_solver.evaluate_ratio()
                            new_result = newest_ratio(
                                A=attack,
                                AL=attacker_level,
                                P=power,
                                D=defense,
                                DL=defender_level,
                                H=health
                            )
                            results.append(old_result / (old_result + abs(old_result - new_result)))
    print(statistics.mean(results))

@staticmethod
def newest_value_test_two():
    attacks = get_stats()
    attacker_levels = get_levels()
    powers = get_powers()
    defenses = get_stats()
    defender_levels = get_levels()

    results = []

    for attack in attacks:
        for attacker_level in attacker_levels:
            for power in powers:
                for defense in defenses:
                    for defender_level in defender_levels:
                        equation_solver = OldEquationSolver(
                            attack=attack,
                            attacker_level=attacker_level,
                            power=power,
                            defense=defense,
                            defender_level=defender_level,
                            health=0
                        )
                        old_result = equation_solver.evaluate_damage()
                        new_result = newest_damage(
                            A=attack,
                            AL=attacker_level,
                            P=power,
                            D=defense,
                            DL=defender_level
                        )
                        results.append(old_result / (old_result + abs(old_result - new_result)))
    print(statistics.mean(results))

@staticmethod
def newest_ratio_test():
    attacks = get_stats()
    attacker_levels = get_levels()
    powers = get_powers()
    defenses = get_stats()
    defender_levels = get_levels()
    healths = get_stats()

    results = []

    for attack in attacks:
        for attacker_level in attacker_levels:
            for power in powers:
                for defense in defenses:
                    for defender_level in defender_levels:
                        for health in healths:
                            equation_solver = OldEquationSolver(
                                attack=attack,
                                attacker_level=attacker_level,
                                power=power,
                                defense=defense,
                                defender_level=defender_level,
                                health=health
                            )
                            old_result = equation_solver.evaluate_ratio()
                            new_result = current_ratio(
                                A=attack,
                                AL=attacker_level,
                                P=power,
                                D=defense,
                                DL=defender_level,
                                H=health
                            )
                            results.append(old_result / (old_result + abs(old_result - new_result)))
    print(statistics.mean(results))
