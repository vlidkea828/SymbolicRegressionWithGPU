import cupy as cp
import re

one_input_operators = [
    'sin',
    'cos',
    'tan',
    'result',
    'fabs'
]

variable_list = {
    'X' : 0,
    'Y' : 1
}

full_variable_list = [
    'A',
    'AL',
    'P',
    'D',
    'DL',
    'H',
    'X',
    'Y'
]

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

final_results_kernal = cp.ElementwiseKernel(
    'T A, T AL, T P, T D, T DL, T H',
    'T R',
    'R = min(max((P + AL) + (A + AL) - (D + DL), 1.0) / (H + DL), 1.0)',
    'final_results')

final_results_kernal_2 = cp.ElementwiseKernel(
    'T A, T AL1, T AL2, T P, T D, T DL1, T DL2, T H',
    'T R',
    'R = max(max(((DL2 * (D + H + DL1)) - (AL2 * (A + P + AL1))), 0.0) / (DL2 * (H + DL1)), 0.0)',
    'final_results_2')

operator_list = {
    'add' : cp.add,
    'sub' : cp.subtract,
    'mul' : cp.multiply,
    'protected_div' : protected_div,
    'sin' : cp.sin,
    'cos' : cp.cos,
    'tan' : cp.sin,
    'fmin' : cp.minimum,
    'fmax' : cp.maximum,
    'fabs' : cp.absolute
}

operator_symbols = {
    'add' : '+',
    'sub' : '-',
    'mul' : '*',
    'protected_div' : '/ 0.0000001 +'
}

def get_results(equation_string, variable_1, variable_2):
    """Turns equation string into answer"""
    parsed_equation_string = re.split(r'[(,)\s]+', equation_string)
    return calculate_step(parsed_equation_string, variable_1, variable_2)

def calculate_step(parsed_equation_string, variable_1, variable_2):
    """Calculates a step of the equation"""
    if one_input_operators.__contains__(parsed_equation_string[0]):
        input_1, remaining_parset_equation_string_1 = calculate_step(
            parsed_equation_string[1:],
            variable_1=variable_1,
            variable_2=variable_2
        )
        return (
            operator_list[parsed_equation_string[0]](input_1),
            remaining_parset_equation_string_1
        )
    if operator_list.__contains__(parsed_equation_string[0]):
        input_1, remaining_parset_equation_string_1 = calculate_step(
            parsed_equation_string[1:],
            variable_1=variable_1,
            variable_2=variable_2
        )
        input_2, remaining_parset_equation_string_2 = calculate_step(
            remaining_parset_equation_string_1,
            variable_1=variable_1,
            variable_2=variable_2
        )
        return (
            operator_list[parsed_equation_string[0]](input_1, input_2),
            remaining_parset_equation_string_2
        )
    if parsed_equation_string[0] == 'X':
        return variable_1, parsed_equation_string[1:]
    if parsed_equation_string[0] == 'Y':
        return variable_2, parsed_equation_string[1:]
    return cp.array(float(parsed_equation_string[0])), parsed_equation_string[1:]

def get_equation_string(equation_string):
    """Calculates a step of the equation"""
    parsed_equation_string = re.split(r'[(,)\s]+', equation_string)
    final_string, _ = get_equation_string_step(parsed_equation_string)
    return final_string

def get_equation_string_step(parsed_equation_string):
    """Calculates a step of the equation"""
    if operator_symbols.__contains__(parsed_equation_string[0]):
        input_1, remaining_parset_equation_string_1 = get_equation_string_step(
            parsed_equation_string[1:]
        )
        input_2, remaining_parset_equation_string_2 = get_equation_string_step(
            remaining_parset_equation_string_1
        )
        return (
            "({} {} {})".format(
                input_1,
                operator_symbols[parsed_equation_string[0]],
                input_2
            ),
            remaining_parset_equation_string_2
        )
    if parsed_equation_string[0] in full_variable_list:
        return parsed_equation_string[0], parsed_equation_string[1:]
    return cp.array(float(parsed_equation_string[0])), parsed_equation_string[1:]
 
def get_final_results(
        attack_results,
        attacker_level_results,
        power_results,
        defense_results,
        defender_level_results,
        health_results):
    """Calculate the final results"""
    return final_results_kernal(
        attack_results,
        attacker_level_results,
        power_results,
        defense_results,
        defender_level_results,
        health_results)

def get_final_results_2(
        attack_results,
        attacker_level_add_results,
        attacker_level_mul_results,
        power_results,
        defense_results,
        defender_level_add_results,
        defender_level_mul_results,
        health_results):
    """Calculate the final results"""
    return final_results_kernal_2(
        attack_results,
        attacker_level_add_results,
        attacker_level_mul_results,
        power_results,
        defense_results,
        defender_level_add_results,
        defender_level_mul_results,
        health_results)
