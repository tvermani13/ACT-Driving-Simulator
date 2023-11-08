#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.optimize import minimize

### First we define the objective function.
### Note that this is a maximization problem, but scipy.optimize handles with minimization problems
### So, we multiply the objective function with -1 to convert the problem to a minimization problem
def f(x):
    return -(x[0]*x[1] + x[1]*x[2]) # Here, x[0] is x_1, x[1] is x_2, x[2] is x_3


### Next we define the constraint functions
# Note that scipy.optimize only handles ">=" (greater than or equal). 
# So, if the problem has less than or equal type constraints, convert them to greater than or equal type
def constraint1(x):
    return x[0] + 2*x[1] - 6
def constraint2(x):
    return x[0] - 3*x[2]

### Next we specify each constraint type            
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'eq', 'fun': constraint2}
const = ([con1,con2])
            
### Now we specify the bounds
bnds = ((0,4),(0,3),(0,2))

### Initial guess chosen based on prior knowledge or randomly
x0 = np.array([2,2,0.67])

### Now we are ready to use the minimize function            
res = minimize(f, x0, bounds=bnds, constraints=const)

print("When x = %f, y = %f, z = %f," % tuple(res.x))
print("f(x) has the maximum value %f." % -res.fun)
