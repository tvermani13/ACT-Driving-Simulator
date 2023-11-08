from scipy.optimize import minimize
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

# # Define the equations for ti
# def t_equations(x):
#     t1 = 0.3 * x
#     t2 = 1 + 0.2 * x
#     t3 = 7 + 0.1 * x
#     t4 = 15 + 0.3 * x
#     t5 = 5 + 0.2 * x
#     t6 = 8 + 0.1 * x
#     return t1, t2, t3, t4, t5, t6

# # Define the objective function
# def objective_function1(x):
#     ti_values = t_equations(x)
#     integral_values = []
#     for ti in ti_values:
#         integral_values.append(quad(ti, 0, x)[0])
#     # integral_values = [quad(ti, 0, xi)[0] for ti, xi in zip(ti_values, x)]
#     return sum(integral_values)

# # Initial guess for x_i
# x_initial_guess = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])

# # Minimize the function
# result = minimize(objective_function1, x_initial_guess, method='SLSQP', constraints=[])  

# # Print the result
# print("Minimized value of the function:", result.fun)
# print("Optimized values of x_i:", result.x)


###### 

    # f1_4 = x[0]  # 15 decision variables, 9 for path flows and 6 for link flows
    # f1_5 = x[1]
    # f1_6 = x[2]
    # f2_4 = x[3]
    # f2_5 = x[4]
    # f2_6 = x[5]
    # f3_4 = x[6]
    # f3_5 = x[7]
    # f3_6 = x[8]
    # x1 = x[9]
    # x2 = x[10]
    # x3 = x[11]
    # x4 = x[12]
    # x5 = x[13]
    # x6 = x[14] 
    
################

########## OBJECTIVE FUNCTIONS TO MINIMIZE ##############

def objective_function_1(x):
    ## plugged in the link performance function into given objective function, integrated with respect to x_i
    x1 = x[9]
    x2 = x[10]
    x3 = x[11]
    x4 = x[12]
    x5 = x[13]
    x6 = x[14]
    return ((0.15*(x1**2)) + (x2 + 0.1*(x2**2)) + (7*x3 + 0.05*(x3**2)) + (15*x4 + 0.15*(x4**2)) +( 5*x5 + 0.1*(x5**2)) + (8*x6 + 0.05*(x6**2)))

def objective_function_2(x):
    x1 = x[9]
    x2 = x[10]
    x3 = x[11]
    x4 = x[12]
    x5 = x[13]
    x6 = x[14]
    return (0.3 * (x1**2)) + (x2 + 0.2*(x2**2)) + (7*x3 + 0.1 * (x3**2)) + (15*x4 + 0.3 * (x4**2)) + (5*x5 + 0.2*(x5**2)) + (8*x6 + 0.1*(x6**2))

####### CONSTRAINTS ##########

def constraint_1(x):
    return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] - 200

def constraint_2(x):
    return x[9] - (x[0] + x[1] + x[2])

def constraint_3(x):
    return x[10] - (x[3] + x[4] + x[5])

def constraint_4(x):
    return x[11] - (x[6] + x[7] + x[8])

def constraint_5(x):
    return x[12] - (x[0] + x[3] + x[6])

def constraint_6(x):
    return x[13] - (x[1] + x[4] + x[7])

def constraint_7(x):
    return x[14] - (x[2] + x[5] + x[8])


con_1 = {"type": "eq", "fun": constraint_1}
con_2 = {"type": "eq", "fun": constraint_2}
con_3 = {"type": "eq", "fun": constraint_3}
con_4 = {"type": "eq", "fun": constraint_4}
con_5 = {"type": "eq", "fun": constraint_5}
con_6 = {"type": "eq", "fun": constraint_6}
con_7 = {"type": "eq", "fun": constraint_7}

constraint_list = [con_1, con_2, con_3, con_4, con_5, con_6, con_7]
# initial_values = np.array([21, 35, 3, 75, 93, 86, 12, 64, 22, 53, 78, 31, 42, 55, 64])
initial_values = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

bounds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None),
          (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))


def minimize_objectives():
    obj_1_result = minimize(objective_function_1, initial_values, bounds=bounds, constraints=constraint_list)
    obj_2_result = minimize(objective_function_2, initial_values, bounds=bounds, constraints=constraint_list)
    return obj_1_result, obj_2_result

def apply_weights(obj1, obj2):
    max_i = 0
    sum_max = 0
    outcomes = []
    for i in np.arange(0, 1, 0.1):
        curr_1 = obj1 * i
        curr_2 = obj2 * (1-i)
        summed_function = curr1 + curr_2
        result = minimize(summed_function, initial_values, bounds=bounds, constraints=constraint_list)
        curr_fun = result.fun
        outcomes.append(curr_fun)
        if outcome_i > sum_max:
            sum_max = outcome_i
            max_i = i
    return max_i, (1-max_i), sum_max, outcomes 

def normalize_results(min_obj_1, min_obj_2, x_set_1, x_set_2):
    # f_i, f_i_o, f_i_max - parameters?
    # f_i ^ norm = (f_i(x) - f_i ^ o) / f_i ^ max - f_i ^ o  -> 18.2.6 in Arora
    
    f_1_x_2 = objective_function_1(x_set_2)
    f_2_x_1 = objective_function_2(x_set_1)
    
    f_1_norm = (objective_function_1 - min_obj_1) / (max(f_1_x_2, min_obj_1) - min_obj_1)
    f_2_norm = (objective_function_2 - min_obj_2) / (max(f_2_x_1, min_obj_2) - min_obj_2)
    
    return f_1_norm, f_2_norm
    
def plot_pareto_optimal_frontier(obj1, obj2):
    i_values = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] 
    _,_,_,outcomes = apply_weights(obj1, obj2)
    plt.plot(i_values, outcomes)

    # Adding labels and title
    plt.xlabel('Weights')
    plt.ylabel('Outcomes')
    plt.title('Graphing of Pareto Optimal Frontier')

    # Display the plot
    plt.show()
    
def main():
    ## get results of scipy minimize for each objective
    result1, result2 = minimize_objectives()
    
    # ## print results
    # print("Result of minimization 1:")
    # print(result1)
    # print()
    # print("Result of minimization 2:")
    # print(result2)

    x_1 = result1.x
    x_2 = result2.x
    
    fun_1 = result1.fun
    fun_2 = result2.fun
    
    
    ## normalize the results of the minimize calls
    new_f_1, new_f_2 = normalize_results(fun_1, fun_2, x_1, x_2)

    ## apply weighted sum approach to all the weights
    weight_1, weight_2, result, outcomes = apply_weights(new_f_1, new_f_2)
    
    # ## print results
    # print("Optimal weight for objective 1: " + weight_1)
    # print("Optimal weight for objective 2: " + weight_2)
    # print("Optimal weight result: " + result)
    
    ## plot pareto optimal frontier
    # plot_pareto_optimal_frontier(result1, result2)
    
main()