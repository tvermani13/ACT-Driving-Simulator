import matplotlib.pyplot as plt

# Example lists
x_values = [1, 2, 3, 4, 5]  # Replace with your x-axis values
y_values = [2, 4, 6, 8, 10]  # Replace with your y-axis values

# Plotting the lists
plt.plot(x_values, y_values)

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Graph of Two Lists')

# Display the plot
plt.show()
