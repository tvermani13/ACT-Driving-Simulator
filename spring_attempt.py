from sklearn import preprocessing
import numpy as np
from scipy.optimize import minimize


def objective_function_1(x):
    f1_4 = x[0]  # 15 decision variables, 9 for path flows and 6 for link flows
    f1_5 = x[1]
    f1_6 = x[2]
    f2_4 = x[3]
    f2_5 = x[4]
    f2_6 = x[5]
    f3_4 = x[6]
    f3_5 = x[7]
    f3_6 = x[8]
    x1 = x[9]
    x2 = x[10]
    x3 = x[11]
    x4 = x[12]
    x5 = x[13]
    x6 = x[14]
    x1_squared = x1 ** 2
    x2_squared = x2 ** 2
    x3_squared = x3 ** 2
    x4_squared = x4 ** 2
    x5_squared = x5 ** 2
    x6_squared = x6 ** 2
    return (0.15*x1_squared + x2 + 0.1*x2_squared + 7*x3 + 0.05*x3_squared + 15*x4 + 0.15*x4_squared + 5*x5 + 0.1*x5_squared + 8*x6 + 0.05*x6_squared)


def objective_function_2(x):
    f1_4 = x[0]  # 15 decision variables, 9 for path flows and 6 for link flows
    f1_5 = x[1]
    f1_6 = x[2]
    f2_4 = x[3]
    f2_5 = x[4]
    f2_6 = x[5]
    f3_4 = x[6]
    f3_5 = x[7]
    f3_6 = x[8]
    x1 = x[9]
    x2 = x[10]
    x3 = x[11]
    x4 = x[12]
    x5 = x[13]
    x6 = x[14]
    x1_squared = x1 ** 2
    x2_squared = x2 ** 2
    x3_squared = x3 ** 2
    x4_squared = x4 ** 2
    x5_squared = x5 ** 2
    x6_squared = x6 ** 2
    return (0.3*x1_squared + x2 + 0.2*x2_squared + 7*x3 + 0.1*x3_squared + 15*x4 + 0.3*x4_squared + 5*x5 + 0.2*x5_squared + 8*x6 + 0.1*x6_squared)

def objective_function_3(x):
    f1_4 = x[0]  # 15 decision variables, 9 for path flows and 6 for link flows
    f1_5 = x[1]
    f1_6 = x[2]
    f2_4 = x[3]
    f2_5 = x[4]
    f2_6 = x[5]
    f3_4 = x[6]
    f3_5 = x[7]
    f3_6 = x[8]
    x1 = x[9]
    x2 = x[10]
    x3 = x[11]
    x4 = x[12]
    x5 = x[13]
    x6 = x[14]
    x1_squared = x1 ** 2
    x2_squared = x2 ** 2
    x3_squared = x3 ** 2
    x4_squared = x4 ** 2
    x5_squared = x5 ** 2
    x6_squared = x6 ** 2
    f1_normal = (((0.15*x1_squared + x2 + 0.1*x2_squared + 7*x3 + 0.05*x3_squared + 15*x4 + 0.15*x4_squared + 5*x5 + 0.1*x5_squared + 8*x6 + 0.05*x6_squared) - 4494.55)) / (4543 - 4494.55)
    f2_normal = (((0.3*x1_squared + x2 + 0.2*x2_squared + 7*x3 + 0.1*x3_squared + 15*x4 + 0.3*x4_squared + 5*x5 + 0.2*x5_squared + 8*x6 + 0.1*x6_squared) - 6774.5)) / (6876.7 - 6774.5)
    array_of_weights_1 = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    array_of_weights_2 = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]
    weighted_sum = 0
    # for weight1, weight2 in array_of_weights_1, array_of_weights_2:
    #     weighted_sum += (weight1 * f1_normal) + (weight2 * f2_normal)
    # return weighted_sum

    for i in range(len(array_of_weights_1)):
        weighted_sum += (array_of_weights_1[i] * f1_normal) + (array_of_weights_2[i] * f2_normal)
    return weighted_sum

def constraint_1(x):
    f1_4 = x[0]  # 15 decision variables, 9 for path flows and 6 for link flows
    f1_5 = x[1]
    f1_6 = x[2]
    f2_4 = x[3]
    f2_5 = x[4]
    f2_6 = x[5]
    f3_4 = x[6]
    f3_5 = x[7]
    f3_6 = x[8]
    return (f1_4 + f1_5 + f1_6 + f2_4 + f2_5 + f2_6 + f3_4 + f3_5 + f3_6 - 200)

def constraint_2(x):
    f1_4 = x[0]  # 15 decision variables, 9 for path flows and 6 for link flows
    f1_5 = x[1]
    f1_6 = x[2]
    f2_4 = x[3]
    f2_5 = x[4]
    f2_6 = x[5]
    f3_4 = x[6]
    f3_5 = x[7]
    f3_6 = x[8]
    x1 = x[9]
    x2 = x[10]
    x3 = x[11]
    x4 = x[12]
    x5 = x[13]
    x6 = x[14]
    return (x1 - f1_4 - f1_5 - f1_6)

def constraint_3(x):
    f1_4 = x[0]  # 15 decision variables, 9 for path flows and 6 for link flows
    f1_5 = x[1]
    f1_6 = x[2]
    f2_4 = x[3]
    f2_5 = x[4]
    f2_6 = x[5]
    f3_4 = x[6]
    f3_5 = x[7]
    f3_6 = x[8]
    x1 = x[9]
    x2 = x[10]
    x3 = x[11]
    x4 = x[12]
    x5 = x[13]
    x6 = x[14]
    return (x2 - f2_4 - f2_5 - f2_6)

def constraint_4(x):
    f1_4 = x[0]  # 15 decision variables, 9 for path flows and 6 for link flows
    f1_5 = x[1]
    f1_6 = x[2]
    f2_4 = x[3]
    f2_5 = x[4]
    f2_6 = x[5]
    f3_4 = x[6]
    f3_5 = x[7]
    f3_6 = x[8]
    x1 = x[9]
    x2 = x[10]
    x3 = x[11]
    x4 = x[12]
    x5 = x[13]
    x6 = x[14]
    return (x3 - f3_4 - f3_5 - f3_6)

def constraint_5(x):
    f1_4 = x[0]  # 15 decision variables, 9 for path flows and 6 for link flows
    f1_5 = x[1]
    f1_6 = x[2]
    f2_4 = x[3]
    f2_5 = x[4]
    f2_6 = x[5]
    f3_4 = x[6]
    f3_5 = x[7]
    f3_6 = x[8]
    x1 = x[9]
    x2 = x[10]
    x3 = x[11]
    x4 = x[12]
    x5 = x[13]
    x6 = x[14]
    return (x4 - f1_4 - f2_4 - f3_4)

def constraint_6(x):
    f1_4 = x[0]  # 15 decision variables, 9 for path flows and 6 for link flows
    f1_5 = x[1]
    f1_6 = x[2]
    f2_4 = x[3]
    f2_5 = x[4]
    f2_6 = x[5]
    f3_4 = x[6]
    f3_5 = x[7]
    f3_6 = x[8]
    x1 = x[9]
    x2 = x[10]
    x3 = x[11]
    x4 = x[12]
    x5 = x[13]
    x6 = x[14]
    return (x5 - f1_5 - f2_5 - f3_5)

def constraint_7(x):
    f1_4 = x[0]  # 15 decision variables, 9 for path flows and 6 for link flows
    f1_5 = x[1]
    f1_6 = x[2]
    f2_4 = x[3]
    f2_5 = x[4]
    f2_6 = x[5]
    f3_4 = x[6]
    f3_5 = x[7]
    f3_6 = x[8]
    x1 = x[9]
    x2 = x[10]
    x3 = x[11]
    x4 = x[12]
    x5 = x[13]
    x6 = x[14]
    return (x6 - f1_6 - f2_6 - f3_6)


constraint_type_1 = {"type": "eq", "fun": constraint_1}
constraint_type_2 = {"type": "eq", "fun": constraint_2}
constraint_type_3 = {"type": "eq", "fun": constraint_3}
constraint_type_4 = {"type": "eq", "fun": constraint_4}
constraint_type_5 = {"type": "eq", "fun": constraint_5}
constraint_type_6 = {"type": "eq", "fun": constraint_6}
constraint_type_7 = {"type": "eq", "fun": constraint_7}

list_of_constraints = [constraint_type_1, constraint_type_2, constraint_type_3, constraint_type_4, constraint_type_5, constraint_type_6, constraint_type_7]

bounds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None),
          (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))

x0 = np.array([21, 35, 3, 75, 93, 86, 12, 64, 22, 53, 78, 31, 42, 55, 64])

result1 = minimize(objective_function_1, x0, bounds=bounds,
                   constraints=list_of_constraints)
result2 = minimize(objective_function_2, x0, bounds=bounds,
                   constraints=list_of_constraints)
result3 = minimize(objective_function_3, x0, bounds = bounds, 
                   constraints = list_of_constraints)


# x0 = x0.reshape(-1,1)
# normalized = preprocessing.normalize(x0)
# print(normalized)

# print(result1)
# print(result2)
print(result3)