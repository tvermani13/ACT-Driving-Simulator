import numpy as np
from scipy.optimize import linprog, minimize


## objective function

def f(x):
    return x[0]* 0.4 + x[1] * 0.8

## constraint function for calories
def constraint1(x):
    return(x[0] * 800) + (x[1] * 1000) - 8000

## constraint function for units of vitamins
def constraint2(x):
    return x[0] * 140 + x[1] * 70 - 700

def constraint3(x):
    total = x[0] + x[1]
    return 0.33 - (x[0] / total)

con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3}
const = ([con1,con2,con3])

bnds = ((0, None), (0, None))

initial = np.array([3,6])

solution = minimize(f, initial, bounds=bnds, constraints=const)

print("When x = %f, y = %f" % tuple(solution.x))
print("f(x) has the minimum value %f." % solution.fun)
