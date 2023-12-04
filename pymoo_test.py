import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

import matplotlib.pyplot as plt


class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=15,
                         n_obj=2,
                         n_ieq_constr=7,
                         xl = [-2,-2, -2,-2, -2,-2, -2,-2, -2,-2, -2,-2, -2,-2, -2],
                         xu = [2,2, 2,2, 2,2, 2,2, 2,2, 2,2, 2,2, 2])

    def _evaluate(self, x, out, *args, **kwargs):
    	f1 = ((0.15*(x[9]**2)) + (x[10] + 0.1*(x[10]**2)) + (7*x[11] + 0.05*(x[11]**2)) + (15*x[12] + 0.15*(x[12]**2)) +( 5*x[13] + 0.1*(x[13]**2)) + (8*x[14] + 0.05*(x[14]**2)))
    	f2 = (0.3 * (x[9]**2)) + (x[10] + 0.2*(x[10]**2)) + (7*x[11] + 0.1 * (x[11]**2)) + (15*x[12] + 0.3 * (x[12]**2)) + (5*x[13] + 0.2*(x[13]**2)) + (8*x[14] + 0.1*(x[14]**2))
    	g1 = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] - 200
    	g2 = x[9] - (x[0] + x[1] + x[2])
    	g3 = x[10] - (x[3] + x[4] + x[5])
    	g4 = x[11] - (x[6] + x[7] + x[8])
    	g5 = x[12] - (x[0] + x[3] + x[6])
    	g6 = x[13] - (x[1] + x[4] + x[7])
    	g7 = x[14] - (x[2] + x[5] + x[8])
    	out["F"] = [f1, f2]
    	out["G"] = [g1, g2, g3, g4, g5, g6, g7]

problem = MyProblem()

algorithm = NSGA2(
	pop_size=40,
	n_offsprings=10,
	sampling=FloatRandomSampling(),
	crossover=SBX(prob=0.9, eta=15),
	mutation=PM(eta=20),
	eliminate_duplicates=True
)

termination = get_termination("n_gen", 40)

res = minimize(problem,
	algorithm,
	termination,
	seed=1,
	save_history=True,
	verbose=True)

X = res.X
F = res.F

