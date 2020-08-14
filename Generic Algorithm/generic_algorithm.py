# https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
import numpy as np 

# inputs of equation
equation_inputs = [4, -2, 3.5, 5, -11, -4.7]
# number of weight we want to optimize
num_weights = 6

sol_per_pop = 8
# define the popuation size
pop_size = (sol_per_pop, num_weights)

# create the initial population
new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)

