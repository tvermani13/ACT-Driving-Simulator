from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


################## INSTRUCTIONS FROM KIBRIA #######################

# Solve the same multi-objective problem that you solved last time using the weighted average method. However, this time, first linearize the two objective functions using first-order approximation. Then, use these linearized objectives to solve the problem.
# Solve the same problem using the NSGA-II algorithm using Pymoo (https://pymoo.org/)
# Compare the Pareto frontiers you obtained after solving the same problem using three different approaches. 

###################################################################



## Example problem ##

# problem = get_problem("zdt1")

# algorithm = NSGA2(pop_size=100)

# res = minimize(problem,
#                algorithm,
#                ('n_gen', 200),
#                seed=1,
#                verbose=False)

# plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, facecolor="none", edgecolor="red")
# plot.show()


