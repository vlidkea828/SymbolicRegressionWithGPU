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
    'T A, T P, T D, T H',
    'T R',
    'R = min(max(P + A - D, 1.0) / H, 1.0)',
    'final_results')

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

def get_results(equation_string, variable_1, variable_2):
    """Turns equation string into answer"""
    equation_steps = make_string_equation(equation_string)
    for index, step in enumerate(equation_steps):
        if step['operator'] == 'result':
            if step['first_variable_index'] == 0:
                return variable_1
            if step['first_variable_index'] == 1:
                return variable_2
            return step['first_variable']
        if step['which_variable'] == 1:
            result = get_step_results(step, [variable_1, variable_2])
            equation_steps[index + step['steps_to_send_answer']]['first_variable'] = result
        else:
            result = get_step_results(step, [variable_1, variable_2])
            equation_steps[index + step['steps_to_send_answer']]['second_variable'] = result
        
def make_string_equation(equation_string):
    """Used to turn a string to an equation"""
    equation_list = [
        {
            'operator': 'result',
            'first_variable': 0,
            'first_variable_index': 6
        }
    ]
    equation_array = re.split(r'[(,)\s]+', equation_string)
    for component in equation_array:
        if operator_list.__contains__(component):
            equation_list.insert(
                0,
                {
                    'operator': component,
                    'first_variable_index': 0,
                    'first_variable' : 0,
                    'second_variable_index' : 0,
                    'second_variable' : 0,
                    'completion' : 0,
                    'steps_to_send_answer' : 0,
                    'which_variable' : 0
                })
        else:
            insert_variable(
                previous_index=-1, 
                equation_list=equation_list, 
                component=component)
    if len(equation_list) == 1:
        if variable_list.__contains__(equation_array[0]):
            equation_list[0]['first_variable_index'] = variable_list[equation_array[0]]
        else:
            equation_list[0]['first_variable_index'] = 6
            equation_list[0]['first_variable'] = cp.array(float(equation_array[0]))
    return equation_list

def insert_variable(previous_index, equation_list, component):
    """Find where to add variables and previous answers"""
    for index in range(previous_index+1, len(equation_list)):
        if equation_list[index]['operator'] == 'result':
            return (index - previous_index, 1)
        if equation_list[index]['completion'] == 0:
            if variable_list.__contains__(component):
                equation_list[index]['first_variable_index'] = variable_list[component]
            else:
                equation_list[index]['first_variable_index'] = 6
                equation_list[index]['first_variable'] = cp.array(float(component))
            if one_input_operators.__contains__(equation_list[index]['operator']):
                equation_list[index]['completion'] = 2
                equation_list[index]['second_variable_index'] = 6
                steps_to_send_answer, which_variable = insert_variable(
                    previous_index=index,
                    equation_list=equation_list, 
                    component=6)
                equation_list[index]['steps_to_send_answer'] = steps_to_send_answer
                equation_list[index]['which_variable'] = which_variable
            else:
                equation_list[index]['completion'] = 1
            return (index - previous_index, 1)
        if equation_list[index]['completion'] == 1:
            if variable_list.__contains__(component):
                equation_list[index]['second_variable_index'] = variable_list[component]
            else:
                equation_list[index]['second_variable_index'] = 6
                values = cp.array(float(component))
                equation_list[index]['second_variable'] = values
            equation_list[index]['completion'] = 2
            steps_to_send_answer, which_variable = insert_variable(
                    previous_index=index,
                    equation_list=equation_list, 
                    component=6)
            equation_list[index]['steps_to_send_answer'] = steps_to_send_answer
            equation_list[index]['which_variable'] = which_variable
            return (index - previous_index, 2)

def get_step_results(step, variables):
    """Calculate the step using geppy"""
    first_variable = None
    if step['first_variable_index'] > 5:
        first_variable = step['first_variable']
    else:
        first_variable = variables[step['first_variable_index']]
    second_variable = None
    if step['second_variable_index'] > 5:
        second_variable = step['second_variable']
    else:
        second_variable = variables[step['second_variable_index']]
    if one_input_operators.__contains__(step['operator']):
        return operator_list[step['operator']](first_variable)
    else:
        return operator_list[step['operator']](first_variable, second_variable)
    
def get_final_results(attack_results, power_results, defense_results, health_results):
    """Calculate the final results"""
    return final_results_kernal(
        attack_results,
        power_results,
        defense_results,
        health_results)
