"""Class containing equations."""

def get_stats():
    """Calculate the travel time between two points including wait time at the end node."""
    return [5.0
        # ,23.0
        # ,41.0
        ,59.0
        # ,77.0
        # ,95.0
        ,113.0
        # ,131.0
        ,149.0
        # ,167.0
        ,185.0
        # ,203.0
        # ,221.0
        ,239.0]

# @staticmethod
# def get_stats():
#     """Calculate the travel time between two points including wait time at the end node."""
#     return [5.0
#         ,23.0
#         ,41.0
#         ,59.0
#         ,77.0
#         ,95.0
#         ,113.0
#         ,131.0
#         ,149.0
#         ,167.0
#         ,185.0
#         ,203.0
#         ,221.0
#         ,239.0]

def get_levels():
    """Calculate the travel time between two points including wait time at the end node."""
    return [7.0
        # ,14.0
        # ,21.0
        ,28.0
        # ,35.0
        # ,42.0
        ,49.0
        # ,56.0
        ,63.0
        # ,70.0
        ,77.0
        # ,84.0
        # ,91.0
        ,98.0]

# @staticmethod
# def get_levels():
#     """Calculate the travel time between two points including wait time at the end node."""
#     return [7.0
#         ,14.0
#         ,21.0
#         ,28.0
#         ,35.0
#         ,42.0
#         ,49.0
#         ,56.0
#         ,63.0
#         ,70.0
#         ,77.0
#         ,84.0
#         ,91.0
#         ,98.0]

def get_powers():
    """Calculate the travel time between two points including wait time at the end node."""
    return [10.0
        # ,15.0
        # ,20.0
        # ,25.0
        ,30.0
        # ,35.0
        # ,40.0
        # ,45.0
        ,50.0
        # ,55.0
        # ,60.0
        # ,65.0
        ,70.0
        # ,75.0
        # ,80.0
        # ,85.0
        ,90.0
        # ,95.0
        # ,100.0
        # ,105.0
        ,110.0
        # ,115.0
        # ,120.0
        # ,125.0
        ,130.0
        # ,135.0
        # ,140.0
        # ,145.0
        ,150.0
        # ,155.0
        # ,160.0
        # ,165.0
        ,170.0
        # ,175.0
        # ,180.0
        # ,185.0
        ,190.0
        # ,195.0
        # ,200.0
        # ,205.0
        ,210.0
        # ,215.0
        # ,220.0
        # ,225.0
        ,230.0
        # ,235.0
        # ,240.0
        # ,245.0
        ,250.0]

# @staticmethod
# def get_powers():
#     """Calculate the travel time between two points including wait time at the end node."""
#     return [10.0
#         ,15.0
#         ,20.0
#         ,25.0
#         ,30.0
#         ,35.0
#         ,40.0
#         ,45.0
#         ,50.0
#         ,55.0
#         ,60.0
#         ,65.0
#         ,70.0
#         ,75.0
#         ,80.0
#         ,85.0
#         ,90.0
#         ,95.0
#         ,100.0
#         ,105.0
#         ,110.0
#         ,115.0
#         ,120.0
#         ,125.0
#         ,130.0
#         ,135.0
#         ,140.0
#         ,145.0
#         ,150.0
#         ,155.0
#         ,160.0
#         ,165.0
#         ,170.0
#         ,175.0
#         ,180.0
#         ,185.0
#         ,190.0
#         ,195.0
#         ,200.0
#         ,205.0
#         ,210.0
#         ,215.0
#         ,220.0
#         ,225.0
#         ,230.0
#         ,235.0
#         ,240.0
#         ,245.0
#         ,250.0]

@staticmethod
def initial_test_inputs():
    return {
        'csv_name' : "ratios.csv",
        'primitive_set_name' : 'Main',
        'input_names' : ['A','AL','P','D','DL','H'],
        'mapping_function' : lambda func, inputs : map(
            func,
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5]),
        'head_length' : 8,
        'number_of_genes_in_chromosome' : 6,
        'rnc_array_length' : 10,
        'enable_linear_scaling' : True,
        'population_size' : 120,
        'number_of_generations' : 2, # 50 100 3000
        'number_of_retained_champions_accross_generations' : 3}
