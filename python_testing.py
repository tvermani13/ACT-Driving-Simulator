import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
from scipy.interpolate import approximate_taylor_polynomial

# # Example lists
# x_values = [1, 2, 3, 4, 5]  # Replace with your x-axis values
# y_values = [2, 4, 6, 8, 10]  # Replace with your y-axis values

# # Plotting the lists
# plt.plot(x_values, y_values)

# # Adding labels and title
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Line Graph of Two Lists')

# # Display the plot
# plt.show()


# def objective_function_1(x):
#     x1 = x[9]
#     x2 = x[10]
#     x3 = x[11]
#     x4 = x[12]
#     x5 = x[13]
#     x6 = x[14]
#     return ((0.15*(x1**2)) + (x2 + 0.1*(x2**2)) + (7*x3 + 0.05*(x3**2)) + (15*x4 + 0.15*(x4**2)) + (5*x5 + 0.1*(x5**2)) + (8*x6 + 0.05*(x6**2)))

# def transform_function(x):
#     return (objective_function_1(x) - 4494.55) / (4543 - 4494.55)

# # Example usage:
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# result = transform_function(x)
# print(result)



# import sympy as sp
# x = sp.Symbol('x')
# f = sp.sin(x)
# n = 5
# taylor_series = f.series(x, 0, n)
# print(taylor_series)


x = np.linspace(-10.0, 10.0, num=100)
plt.plot(x, np.sin(x), label="sin curve")
for degree in np.arange(1, 15, step=2):
    sin_taylor = approximate_taylor_polynomial(np.sin, 0, degree, 1,
                                               order=degree + 2)
    plt.plot(x, sin_taylor(x), label=f"degree={degree}")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0.0, shadow=True)
plt.tight_layout()
plt.axis([-10, 10, -10, 10])
plt.show()