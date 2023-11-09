from scipy.optimize import minimize
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

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


def minimize_objectives(initial_values, bounds, constraint_list):
    obj_1_result = minimize(objective_function_1, initial_values, bounds=bounds, constraints=constraint_list)
    obj_2_result = minimize(objective_function_2, initial_values, bounds=bounds, constraints=constraint_list)
    return obj_1_result, obj_2_result


################################### GLOBALS ##################################

con_1 = {"type": "eq", "fun": constraint_1}
con_2 = {"type": "eq", "fun": constraint_2}
con_3 = {"type": "eq", "fun": constraint_3}
con_4 = {"type": "eq", "fun": constraint_4}
con_5 = {"type": "eq", "fun": constraint_5}
con_6 = {"type": "eq", "fun": constraint_6}
con_7 = {"type": "eq", "fun": constraint_7}
constraint_list = [con_1, con_2, con_3, con_4, con_5, con_6, con_7]

initial_values = np.array([21, 35, 3, 75, 93, 86, 12, 64, 22, 53, 78, 31, 42, 55, 64])

bounds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None),
        (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))

result1, result2 = minimize_objectives(initial_values=initial_values, bounds=bounds, constraint_list=constraint_list)

x_1 = result1.x
x_2 = result2.x

fun_1 = result1.fun
fun_2 = result2.fun
    
################################### END GLOBALS ##################################

    
def normalize_1(x):
    # f_i, f_i_o, f_i_max - parameters?
    # f_i ^ norm = (f_i(x) - f_i ^ o) / f_i ^ max - f_i ^ o  -> 18.2.6 in Arora
    x1 = x[9]
    x2 = x[10]
    x3 = x[11]
    x4 = x[12]
    x5 = x[13]
    x6 = x[14]

    f_1_x_2 = objective_function_1(x_2)
    denom_1 = (max(f_1_x_2, fun_1) - fun_1)
    
    return (((0.15*(x1**2)) + (x2 + 0.1*(x2**2)) + (7*x3 + 0.05*(x3**2)) + (15*x4 + 0.15*(x4**2)) +( 5*x5 + 0.1*(x5**2)) + (8*x6 + 0.05*(x6**2))) - fun_1) / denom_1
    

def normalize_2(x):
    x1 = x[9]
    x2 = x[10]
    x3 = x[11]
    x4 = x[12]
    x5 = x[13]
    x6 = x[14]

    f_2_x_1 = objective_function_2(x_1)
    denom_2 = (max(f_2_x_1, fun_2) - fun_2)

    return (((0.3 * (x1**2)) + (x2 + 0.2*(x2**2)) + (7*x3 + 0.1 * (x3**2)) + (15*x4 + 0.3 * (x4**2)) + (5*x5 + 0.2*(x5**2)) + (8*x6 + 0.1*(x6**2))) - fun_2) / denom_2

    
def weighted_function(x,i):
    return (normalize_1(x) * i) + (normalize_2(x) * (1-i))


def plot_pareto_optimal_frontier(y_vals, highlight_pt):
    i_values = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] 
    plt.plot(i_values, y_vals, label='Pareto Optimal Frontier', marker='o')

    highlight_index = i_values.index(highlight_pt)
    plt.scatter(highlight_x, y_vals[highlight_index], color='red', label=f'Highlight ({highlight_x}, {y_vals[highlight_index]})')

    
    # Adding labels and title
    plt.xlabel('Weights')
    plt.ylabel('Outcomes')
    plt.title('Graphing of Pareto Optimal Frontier')

    # Display the plot
    plt.show()

def plot_pareto_optimal_frontier(y_vals):
    i_values = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] 
    plt.plot(i_values, y_vals, label='Pareto Optimal Frontier', marker='o')
    
    # Adding labels and title
    plt.xlabel('Weights')
    plt.ylabel('Outcomes')
    plt.title('Graphing of Pareto Optimal Frontier')

    # Display the plot
    plt.show()



def main():
    ## normalize the results of the minimize calls
    normal_1 = normalize_1(initial_values)
    normal_2 = normalize_2(initial_values)     
    max_output = 0
    outputs = []
    for i in np.arange(0, 1.1, 0.1):
        extra_args = (i)
        curr_result = minimize(weighted_function, initial_values, constraints=constraint_list, bounds=bounds, args=extra_args)
        curr_fun = curr_result.fun
        outputs.append(curr_fun)
        if curr_fun > max_output:
            max_output = curr_fun
    
    print(max_output)
    # plot_pareto_optimal_frontier(outputs, max_output)
    plot_pareto_optimal_frontier(outputs)
    

main()