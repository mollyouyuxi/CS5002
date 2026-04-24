import numpy as np
import matplotlib.pyplot as plt
from Gradient_Descent_Multi_Variables import gradient_descent

def load_data(filename):
    """
    Load the data into two arrays
    Args:
    filename: string representing the name of the file
    containing the data (x,y)
    Return:
    array containing the values of x
    array containing the values of y
    """


    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]

def data_scaling(x, y):
    """
    Scale the dataset 
    by substacting every data by the mean and deviding by the standard deviation
    Args:
    x: the original value of x 
    y: the original value of y
    Return: 
    array containing the scaled value of x
    array containing the scaled value of y
    """
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    std_x = np.std(x)
    std_y = np.std(y)
    x_scaled = (x - mean_x) / std_x
    y_scaled = (y - mean_y) / std_y
    return x_scaled,y_scaled


def cost_func (a, b, x, y):
    """
    compute the cost function g(a,b)=sum[(a*xi-b-yi)^2]
    Args:
    a: parameter a of the cost function
    b: parameter b of the cost function
    x: dataset x
    y: dataset y
    """
    g = 0

    for i in range (len(x)):
        g = g + (a * x[i] + b - y[i])**2

    return g

def deriv_cost_func(a, b, x, y):
    """
    compute the derivatives of the cost function
    """
    deriv_a = 0
    deriv_b = 0
    for i in range (len(x)):
        deriv_a = deriv_a + 2 * (a * x[i] + b - y[i]) * x[i]
        deriv_b = deriv_b + 2* (a * x[i] + b - y[i])
    return [deriv_a, deriv_b]


def deriv_g(a, b):
    """
    Helper method on the orginal dataset to accomondate the gradient descent algorith
    """
    return deriv_cost_func(a,b,x,y)

def deriv_g_scaled(a,b):
    """
    Helper method on the scaled dataset to accomondate the gradient descent algorith
    """
    return deriv_cost_func(a=a, b=b, x=x_scaled, y = y_scaled)

def plot_data_and_line(
    x,
    y,
    a_star,
    b_star,
    xlabel="Total Cholesterol Level(mmol/L)",
    ylabel="Diastolic Blood Pressure(mm Hg)",
    title="Diastolic Blood Pressure vs Cholesterol"
):
    """
    Plot the data points and the optimized linear model.
    """
    plt.figure()

    # Plot original data points
    plt.scatter(x, y, color="steelblue", label="Data points")

    # Plot the optimal line y = a*x + b
    y_pred = a_star * x + b_star
    plt.plot(x, y_pred, color="red", label="Optimal Line")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()



# Compute the x,y
x,y = load_data('Sources/data_chol_dias_pressure.txt')
# Compute the scaled x, y
x_scaled, y_scaled = data_scaling(x,y)

# Compute the optimal value and draw the plot
optimizers, steps = gradient_descent(
    initialpts=[0, 0],
    derivative=deriv_g,
    alpha=0.0001,
    epsilon=0.00001,
    iter_max=1000000
)
a_star = optimizers[0]
b_star = optimizers[1]
steps = steps
cost = cost_func(a_star, b_star, x, y)
print(
    f"\nGradient descent result for the original dataset:\n"
    f"  alpha = {1e-06}\n"
    f"  epsilon = {1e-06}\n"
    f"  iter_max = 1000000\n"
    f"  steps taken = {steps}\n"
    f"  optimal a = {a_star:.6f}\n"
    f"  optimal b = {b_star:.6f}\n"
    f"  cost = {cost:.6f}"
)
# Plot the data points and the optimal line obtained by the original value of x and y 
plot_data_and_line(x=x, y=y, a_star=a_star, b_star=b_star)


# Compute the optimal values of a and b by using the gradient_descent method 
# Instantiate a set of parameters to find the convergence 
for alpha in {1e-5, 1e-6, 5e-6}:
    for epsilon in {1e-6, 1e-7, 5e-7,1e-8}:
        optimizers, steps = gradient_descent(
            initialpts=[0, 0],
            derivative=deriv_g,
            alpha=alpha,
            epsilon=epsilon,
            iter_max=500000
        )

        a_star = optimizers[0]
        b_star = optimizers[1]
        cost = cost_func(a_star, b_star, x, y)

        print(f"\nParameter Tuning:")
        print(f"alpha = {alpha}, epsilon = {epsilon}, iter_max = 500000")
        print(f"steps taken = {steps}, optimal a = {a_star:.6f}, optimal b = {b_star:.6f}, cost = {cost:.6f}")

# Using the parameter find above to compute the optimal value and draw the plot
optimizers, steps = gradient_descent(
    initialpts=[0, 0],
    derivative=deriv_g,
    alpha=1e-06,
    epsilon=1e-06,
    iter_max=1000000
)
a_star = optimizers[0]
b_star = optimizers[1]
steps = steps
cost = cost_func(a_star, b_star, x, y)
print(
    f"\nGradient descent result for the original dataset:\n"
    f"  alpha = {1e-06}\n"
    f"  epsilon = {1e-06}\n"
    f"  iter_max = 1000000\n"
    f"  steps taken = {steps}\n"
    f"  optimal a = {a_star:.6f}\n"
    f"  optimal b = {b_star:.6f}\n"
    f"  cost = {cost:.6f}"
)
# Plot the data points and the optimal line obtained by the original value of x and y 
plot_data_and_line(x=x, y=y, a_star=a_star, b_star=b_star)


# Using the scaled data to compute the optimal value 
optimizers, steps = gradient_descent(
    initialpts=[0, 0],
    derivative=deriv_g_scaled,
    alpha=0.001,
    epsilon=0.0001,
    iter_max=1000
)
a_star_scaled = optimizers[0]
b_star_scaled = optimizers[1]
steps = steps
cost_scaled = cost_func(a_star_scaled, b_star_scaled, x_scaled, y_scaled)
print(
    f"\nGradient descent result for the scaled dataset:\n"
    f"  alpha = {0.001}\n"
    f"  epsilon = {0.0001}\n"
    f"  iter_max = 1000\n"
    f"  steps taken = {steps}\n"
    f"  optimal a = {a_star_scaled:.6f}\n"
    f"  optimal b = {b_star_scaled:.6f}\n"
    f"  cost = {cost_scaled:.6f}"
)
# Plot the data points and the optimal line obtained by the scaled value of x and y 
plot_data_and_line(
    x=x_scaled,
    y=y_scaled,
    a_star=a_star_scaled,
    b_star=b_star_scaled,
    xlabel="Scaled Total Cholesterol Level",
    ylabel="Scaled Diastolic Blood Pressure",
    title="Scaled Diastolic Blood Pressure vs Cholesterol"
)
