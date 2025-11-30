import cupy as cp
import numpy as np
import numba
import re
from source.Utilities import globals
from sympy import *
from sympy.abc import X, Y

one_input_operators = [
    4, # 'sin',
    5, # 'cos',
    6, # 'tan',
    11, # 'result',
    9 # 'fabs'
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

operator_list = {
    'add' : 0,
    'sub' : 1,
    'mul' : 2,
    'protected_div' : 3,
    'sin' : 4,
    'cos' : 5,
    'tan' : 6,
    'fmin' : 7,
    'fmax' : 8,
    'fabs' : 9,
    'result' : 11
}

equation_component_order = {
   'operator': 0,
    'first_variable_index': 1,
    'first_variable' : 2,
    'second_variable_index' : 3,
    'second_variable' : 4,
    'completion' : 5,
    'steps_to_send_answer' : 6,
    'which_variable' : 7
}

@cp.fuse(kernel_name='newest_test')
def newest_calc_test(x, y, q, z):
    return q

new_extra_stuff = """double func1(double a, double b){{
                if (a > b)
                    return 0.0;
                else
                    return a + b;
            }};"""

new_evaluate_func = cp.ElementwiseKernel(
            'float64 X, float64 Y, raw T Q, int32 W, int32 O',
            'float64 z',
            """
            auto get_result = [] (
                double variable_1,
                double variable_2,
                double operation
            )
            {{
                if (operation == 0.0)
                {{
                    return variable_1 + variable_2;
                }}
                if (operation == 1.0)
                {{
                    return variable_1 - variable_2;
                }}
                if (operation == 2.0)
                {{
                    return variable_1 * variable_2;
                }}
                if (operation == 3.0)
                {{
                    if (variable_2 < 0.0000001)
                    {{
                        return variable_1 / 0.0000001;
                    }}
                    else
                    {{
                        return variable_1 / variable_2;
                    }}
                }}
                if (operation == 4.0)
                {{
                    return sin(variable_1);
                }}
                if (operation == 5.0)
                {{
                    return cos(variable_1);
                }}
                if (operation == 6.0)
                {{
                    return tan(variable_1);
                }}
                return 0.0;
            }};
            auto get_variable = [] (
                double x1,
                double x2,
                double prepped,
                double list,
                double choice)
            {{
               if (choice == 0.0)
               {{
                    return x1;
                }}
                if (choice == 1.0)
                {{
                    return x2;
                }}
                if (choice == 6.0)
                {{
                    return prepped;
                }}
                if (choice == 7.0)
                {{
                    return list;
                }}
                return 0.0;
            }};
            int current_gup_index = i * O;
            if (W == 0.0) {{
                if (Q[current_gup_index+1] == 0.0)
                {{
                    z = X;
                }}
                if (Q[current_gup_index+1] == 1.0)
                {{
                    z = Y;
                }}
                if (Q[current_gup_index+1] == 6.0)
                {{
                    z = Q[current_gup_index+2];
                }}
            }}
            else
            {{
                double variable_1;
                double variable_2;
                double answer;
                int index_start;
                int variable_location;
                double first_variable_list[50] = { };
                double second_variable_list[50] = { };
                for(int index=0;index<W;index++)
                {{
                    index_start = index * 8;
                    variable_location = index + Q[current_gup_index+index_start+6];
                    variable_1 = get_variable(
                        X,
                        Y,
                        Q[current_gup_index+index_start+2],
                        first_variable_list[index],
                        Q[current_gup_index+index_start+1]
                    );
                    variable_2 = get_variable(
                        X,
                        Y,
                        Q[current_gup_index+index_start+4],
                        second_variable_list[index],
                        Q[current_gup_index+index_start+3]
                    );
                    answer = get_result(
                        variable_1,
                        variable_2,
                        Q[current_gup_index+index_start]);
                    if (Q[current_gup_index+index_start+7] == 1.0)
                    {{
                        first_variable_list[variable_location] = answer;
                    }}
                    else
                    {{
                        second_variable_list[variable_location] = answer;
                    }}
                }}
                z = first_variable_list[W]; 
            }}
            """,
            'current_evaluation',
            options=("-std=c++11",),)

extra_stuff_new = """
         double variable_1;
                double variable_2;
                double answer;
                int index_start;
                int variable_location;
                double *first_variable_list = new double[W + 1];
                double *second_variable_list = new double[W + 1];
                for(int index=0;index<W;index++)
                {{
                    index_start = index * 8;
                    variable_location = index + Q[index_start+6];
                    variable_1 = get_variable(
                        X,
                        Y,
                        Q[index_start+2],
                        first_variable_list[index],
                        Q[index_start+1]
                    );
                    variable_2 = get_variable(
                        X,
                        Y,
                        Q[index_start+4],
                        second_variable_list[index],
                        Q[index_start+3]
                    );
                    answer = get_result(
                        variable_1,
                        variable_2,
                        Q[index_start]);
                    if (Q[index_start+7] == 1.0)
                    {{
                        first_variable_list[variable_location] = answer;
                    }}
                    else
                    {{
                        second_variable_list[variable_location] = answer;
                    }}
                }}
                z = first_variable_list[W];   
"""

stat_results_kernel = cp.RawKernel(r'''
    extern "C" __host__
    float get_variable(
        const float x1,
        const float x2,
        const float prepped,
        const float list,
        const float choice) {
        if (choice == 0){
            return x1;
        }
        if (choice == 1){
            return x2;
        }
        if (choice == 6){
            return prepped;
        }
        if (choice == 7){
            return list;
        }
        return 0;
    }
    extern "C" __host__
    float get_reault(
        const float variable_1,
        const float variable_2,
        const float operation) {
        if (operation == 0){
            return variable_1 + variable_2;
        }
        if (operation == 1){
            return variable_1 - variable_2;
        }
        if (operation == 2){
            return variable_1 * variable_2;
        }
        if (operation == 3){
            if (variable_2 < 0.0000001){
                return variable_1 / 0.0000001;
            }
            else{
                return variable_1 / variable_2;
            }
        }
        if (operation == 4){
            return sin(variable_1);
        }
        if (operation == 5){
            return cos(variable_1);
        }
        if (operation == 6){
            return tan(variable_1);
        }
        return 0;
    }
    extern "C" __global__
    void stat_results(const float* x1, const float* x2, float* y, const float q[][8], const int z) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (z == 0) {
            if (q[0][1] == 0){
                y[tid] = x1[tid];
            }
            if (q[0][1] == 1){
                y[tid] = x2[tid];
            }
            if (q[0][1] == 6){
                y[tid] = q[0][2];
            }
        }
        else {
            float variable_1;
            float variable_2;
            int variable_location;
            float answer;
            float *first_variable_list = new float[z + 1];
            float *second_variable_list = new float[z + 1];
            for (int i = 0; i < z; i++) {
                variable_location = i + q[i][6];
                variable_1 = get_variable(
                    x1[tid],
                    x2[tid],
                    q[i][2],
                    first_variable_list[i],
                    q[i][1]);
                variable_2 = get_variable(
                    x1[tid],
                    x2[tid],
                    q[i][4],
                    second_variable_list[i],
                    q[i][3]);
                answer = get_reault(
                    variable_1,
                    variable_2,
                    q[i][0]);
                if (q[i][7] == 1){
                    first_variable_list[variable_location] = answer;
                }
                else{
                    second_variable_list[variable_location] = answer;
                }
            }
            y[tid] = first_variable_list[z];
        }
    }
    ''',
    'stat_results',
    jitify=True)

extra_stuff = r'''
int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (z == 0) {
            if (q[0][1] == 0){
                y[tid] = x1[tid];
            }
            if (q[0][1] == 1){
                y[tid] = x2[tid];
            }
            if (q[0][1] == 6){
                y[tid] = q[0][2];
            }
        }
        else {
            float variable_1;
            float variable_2;
            int variable_location;
            float answer;
            float *first_variable_list = new float[z + 1];
            float *second_variable_list = new float[z + 1];
            for (int i = 0; i < z; i++) {
                variable_location = i + q[i][6];
                variable_1 = get_variable(
                    x1[tid],
                    x2[tid],
                    q[i][2],
                    first_variable_list[i],
                    q[i][1]);
                variable_2 = get_variable(
                    x1[tid],
                    x2[tid],
                    q[i][4],
                    second_variable_list[i],
                    q[i][3]);
                answer = get_reault(
                    variable_1,
                    variable_2,
                    q[i][0]);
                if (q[i][7] == 1){
                    first_variable_list[variable_location] = answer;
                }
                else{
                    second_variable_list[variable_location] = answer;
                }
            }
            y[tid] = first_variable_list[z];
        }
'''

final_results_kernal = cp.ElementwiseKernel(
    'T A, T P, T D, T H',
    'T R',
    'R = min(max(P + A - D, 1.0) / H, 1.0)',
    'final_results')

def test_raw_kernal():
    """Testing raw kernals functionality"""
    size = globals.inputs[3].shape[0]
    x1 = globals.inputs[0]
    x2 = globals.inputs[1]
    lambdastr = 'lambda x, y: ((x + y) / 16.0)'
    lambdafunc = eval(lambdastr)
    f = lambdify((X, Y), '(X)', "cupy")
    thing = f(x1, x2)

    scope = {}
    lambdastr = """def new_func(A, AL, P): 
            AT = (A + AL) / 32
            if AT.any() < 0:
                return -1 * A
            PL = (P - AL)
            if PL.any() < 0:
                return -1 * P
            return PL + AT
        """
    exec(lambdastr, scope)
    numbafunc = numba.jit(nopython=True, parallel=True)(scope['new_func'])
    numba_output = numbafunc(x1, x2, x3)
    print(np.any(np.less(numba_output, 0)))
    # x1 = cp.arange(size, dtype=cp.float32).reshape(size)
    # x2 = cp.arange(size, dtype=cp.float32).reshape(size)
    y1 = cp.zeros((26941), dtype=cp.float32)
    y2 = cp.zeros((26941), dtype=cp.float32)
    y3 = cp.zeros((26941), dtype=cp.float32)
    
    # remember this is a memory overwrite error, q cannot store its own values they must
    # be saved to an internally made array
    x11 = x1[0:26941]
    x12 = x1[26941:53882]
    print(x12[0])
    x13 = x1[53882:80823]
    x21 = x2[0:26941]
    x22 = x2[26941:53882]
    x23 = x2[53882:80823]
    y1 = calculate(x1, x2)
    y2 = calculate(x12, x22)
    y3 = calculate(x13, x23)
    # stat_results_kernel((929,), (29,), (x11, x21, y1, q, z))  # grid, block and arguments
    # stat_results_kernel((929,), (29,), (x12, x22, y2, q, z))  # grid, block and arguments
    # stat_results_kernel((929,), (29,), (x13, x23, y3, q, z))  # grid, block and arguments
    # print(y1)
    # print(y2)
    # print(y3)
    # y = cp.concatenate(cp.concatenate(y1, y2), y3)
    # print(y)

def calculate(x1, x2):
    """Does a calc"""
    q = cp.zeros((1, 8), dtype=cp.float64)
    q[0][0] = 11
    q[0][1] = 7
    q[0][2] = 0
    q[0][3] = 0
    q[0][4] = 0
    q[0][5] = 2
    q[0][6] = 0
    q[0][7] = 0

    r = cp.zeros((1, 8), dtype=cp.float64)
    r[0][0] = 3
    r[0][1] = 0
    r[0][2] = 0
    r[0][3] = 6
    r[0][4] = 498
    r[0][5] = 2
    r[0][6] = 1
    r[0][7] = 1
    q = cp.vstack((r, q))

    r = cp.zeros((1, 8), dtype=cp.float64)
    r[0][0] = 0
    r[0][1] = 0
    r[0][2] = 0
    r[0][3] = 1
    r[0][4] = 0
    r[0][5] = 2
    r[0][6] = 1
    r[0][7] = 1
    q = cp.vstack((r, q))
    z = q.shape[0] - 1
    o = q.shape[0] * 8
    q = q.reshape(1, o)
    q = cp.repeat(q, x1.shape[0], axis=0)
    print(x1)
    print(x2)
    print(r)
    print(q)
    
    print(z)
    y = new_evaluate_func(x1, x2, q, z, o) 
    print(y)
    return y

def get_results(equation_string, variables_1, variables_2):
    """Turns equation string into answer"""
    equation_steps = make_string_equation(equation_string)
    equation_steps_final_index = equation_steps.shape[0] - 1
    equation_steps_array_length = equation_steps.shape[0] * 8
    equation_steps = equation_steps.reshape(1, equation_steps_array_length)
    equation_steps = cp.repeat(equation_steps, variables_1.shape[0], axis=0)
    results = new_evaluate_func(
        variables_1, 
        variables_2, 
        equation_steps, 
        equation_steps_final_index,
        equation_steps_array_length)
    return results

def make_string_equation(equation_string):
    """Used to turn a string to an equation"""
    equation_list = np.zeros((1, 8), dtype=cp.float32)
    equation_list[0][equation_component_order['operator']] = 11
    equation_list[0][equation_component_order['first_variable_index']] = 7
    equation_list[0][equation_component_order['first_variable']] = 0
    equation_list[0][equation_component_order['second_variable_index']] = 0
    equation_list[0][equation_component_order['second_variable']] = 0
    equation_list[0][equation_component_order['completion']] = 2
    equation_list[0][equation_component_order['steps_to_send_answer']] = 0
    equation_list[0][equation_component_order['which_variable']] = 0
    equation_array = re.split(r'[(,)\s]+', equation_string)
    for component in equation_array:
        if operator_list.__contains__(component):
            equation_list = np.vstack((np.zeros((1, 8), dtype=cp.float32), equation_list))
            equation_list[0][equation_component_order['operator']] = operator_list[component]
        else:
            insert_variable(
                previous_index=-1,
                equation_list=equation_list,
                component=component)
    if equation_list.shape[0] == 1:
        if variable_list.__contains__(equation_array[0]):
            equation_list[0][equation_component_order['first_variable_index']] = variable_list[equation_array[0]]
        else:
            equation_list[0][equation_component_order['first_variable_index']] = 6
            equation_list[0][equation_component_order['first_variable']] = np.array(float(equation_array[0]))
    return cp.asarray(equation_list)

def insert_variable(previous_index, equation_list, component):
    """Find where to add variables and previous answers"""
    for index in range(previous_index+1, len(equation_list)):
        if equation_list[index][equation_component_order['operator']] == operator_list['result']:
            return (index - previous_index, 1)
        if equation_list[index][equation_component_order['completion']] == 0:
            if variable_list.__contains__(component):
                equation_list[index][equation_component_order['first_variable_index']] = variable_list[component]
            else:
                equation_list[index][equation_component_order['first_variable_index']] = 6
                equation_list[index][equation_component_order['first_variable']] = np.array(float(component))
            if one_input_operators.__contains__(equation_list[index][equation_component_order['operator']]):
                equation_list[index][equation_component_order['completion']] = 2
                equation_list[index][equation_component_order['second_variable_index']] = 6
                steps_to_send_answer, which_variable = insert_variable(
                    previous_index=index,
                    equation_list=equation_list,
                    component=6)
                equation_list[index][equation_component_order['steps_to_send_answer']] = steps_to_send_answer
                equation_list[index][equation_component_order['which_variable']] = which_variable
            else:
                equation_list[index][equation_component_order['completion']] = 1
            return (index - previous_index, 1)
        if equation_list[index][equation_component_order['completion']] == 1:
            if variable_list.__contains__(component):
                equation_list[index][equation_component_order['second_variable_index']] = variable_list[component]
            else:
                equation_list[index][equation_component_order['second_variable_index']] = 6
                values = np.array(float(component))
                equation_list[index][equation_component_order['second_variable']] = values
            equation_list[index][equation_component_order['completion']] = 2
            steps_to_send_answer, which_variable = insert_variable(
                    previous_index=index,
                    equation_list=equation_list,
                    component=6)
            equation_list[index][equation_component_order['steps_to_send_answer']] = steps_to_send_answer
            equation_list[index][equation_component_order['which_variable']] = which_variable
            return (index - previous_index, 2)

def get_final_results(attack_results, power_results, defense_results, health_results):
    """Calculate the final results"""
    return final_results_kernal(
        attack_results,
        power_results,
        defense_results,
        health_results)
