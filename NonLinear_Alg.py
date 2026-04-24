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
    compute the cost function 
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

def deriv_g_scaled(a,b):
    """
    Helper method for the scaled dataset.
    """
    return deriv_cost_func(a=a, b=b, x=x_scaled, y=y_scaled)

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

def plot_data_and_quadratic(x, y, a_star, b_star, c_star):
    """
    Plot the data points and the optimized quadratic model.
    """
    plt.figure()

    plt.scatter(x, y, color="steelblue", label="Data points")

    x_curve = np.linspace(min(x), max(x), 200)
    y_curve = a_star * (x_curve ** 2) + b_star * x_curve + c_star
    plt.plot(x_curve, y_curve, color="darkorange", label="Optimal Quadratic Curve")

    plt.xlabel("Scaled Total Cholesterol Level")
    plt.ylabel("Scaled Diastolic Blood Pressure")
    plt.title("Scaled Diastolic Blood Pressure vs Cholesterol")
    plt.legend()
    plt.show()

def plot_data_and_cubic(x, y, a_star, b_star, c_star, d_star):
    """
    Plot the data points and the optimized cubic model.
    """
    plt.figure()

    plt.scatter(x, y, color="steelblue", label="Data points")

    x_curve = np.linspace(min(x), max(x), 200)
    y_curve = a_star * (x_curve ** 3) + b_star * (x_curve ** 2) + c_star * x_curve + d_star
    plt.plot(x_curve, y_curve, color="green", label="Optimal Cubic Curve")

    plt.xlabel("Scaled Total Cholesterol Level")
    plt.ylabel("Scaled Diastolic Blood Pressure")
    plt.title("Scaled Diastolic Blood Pressure vs Cholesterol")
    plt.legend()
    plt.show()



# Compute the x,y
x,y = load_data('Sources/data_chol_dias_pressure_non_lin.txt')
x_scaled, y_scaled = data_scaling(x, y)

# Compute the optimal values of a and b for a linear fit on the new dataset.
optimizers, steps = gradient_descent(
    initialpts = [0, 0],
    derivative = deriv_g_scaled,
    alpha=0.001,
    epsilon = 0.0001,
    iter_max = 1000
)
a_star_scaled = optimizers[0]
b_star_scaled = optimizers[1]

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

plot_data_and_line(
    x=x_scaled,
    y=y_scaled,
    a_star=a_star_scaled,
    b_star=b_star_scaled,
    xlabel="Scaled Total Cholesterol Level",
    ylabel="Scaled Diastolic Blood Pressure",
    title="Scaled Diastolic Blood Pressure vs Cholesterol"
)






# Find a non-linear predictive model
def cost_func_2 (a, b, c, x, y):
    """
    compute the cost function g(a,b,c) = sum[(axi^2 - bxi -c -yi)^2]
    Args:
    a: parameter a of x^2 term of the cost func
    b: parameter of x term of the cost func
    c: constant parameter of the cost func
    x: dataset x
    y: dataset y
    """
    g = 0

    for i in range (len(x)):
        g = g + (a * (x[i]**2) + b * x[i] + c - y[i])**2

    return g

def deriv_cost_func_2(a, b, c, x, y):

    deriv_a = 0
    deriv_b = 0
    deriv_c = 0
    for i in range (len(x)):
        deriv_a = deriv_a + 2 * (a * x[i]**2 + b * x[i] + c - y[i]) * (x[i]**2) 
        deriv_b = deriv_b + 2 * (a * x[i]**2 + b * x[i] + c - y[i]) * x[i]
        deriv_c = deriv_c + 2 * (a * x[i]**2 + b * x[i] + c - y[i])
    return [deriv_a, deriv_b, deriv_c]

def deriv_g2_scaled(a,b,c):
    """
    Helper method for the scaled dataset.
    """
    return deriv_cost_func_2(a=a, b=b, c=c, x=x_scaled, y=y_scaled)

# Compute the optimal values of a, b and c for a non linear fit on the new dataset.
optimizers, steps = gradient_descent(
    initialpts = [0, 0, 0],
    derivative = deriv_g2_scaled,
    alpha=0.001,
    epsilon = 0.0001,
    iter_max = 1000
)
a_star_scaled = optimizers[0]
b_star_scaled = optimizers[1]
c_star_scaled = optimizers[2]

cost_scaled_2 = cost_func_2(a_star_scaled, b_star_scaled, c_star_scaled, x_scaled, y_scaled)

print(
    f"\nGradient descent result for quadratic cost function on the scaled dataset:\n"
    f"  alpha = {0.001}\n"
    f"  epsilon = {0.0001}\n"
    f"  iter_max = 1000\n"
    f"  steps taken = {steps}\n"
    f"  optimal a = {a_star_scaled:.6f}\n"
    f"  optimal b = {b_star_scaled:.6f}\n"
    f"  optimal c = {c_star_scaled:.6f}\n"
    f"  cost = {cost_scaled_2:.6f}"
)

plot_data_and_quadratic(
    x=x_scaled,
    y=y_scaled,
    a_star=a_star_scaled,
    b_star=b_star_scaled,
    c_star=c_star_scaled
)


# Find a cubic predictive model
def cost_func_3(a, b, c, d, x, y):
    """
    Compute the cost function for the cubic model y = ax^3 + bx^2 + cx + d.
    """
    g = 0

    for i in range(len(x)):
        g = g + (a * (x[i] ** 3) + b * (x[i] ** 2) + c * x[i] + d - y[i]) ** 2

    return g

def deriv_cost_func_3(a, b, c, d, x, y):
    """
    Compute the derivatives of the cubic cost function.
    """
    deriv_a = 0
    deriv_b = 0
    deriv_c = 0
    deriv_d = 0

    for i in range(len(x)):
        residual = a * (x[i] ** 3) + b * (x[i] ** 2) + c * x[i] + d - y[i]
        deriv_a = deriv_a + 2 * residual * (x[i] ** 3)
        deriv_b = deriv_b + 2 * residual * (x[i] ** 2)
        deriv_c = deriv_c + 2 * residual * x[i]
        deriv_d = deriv_d + 2 * residual

    return [deriv_a, deriv_b, deriv_c, deriv_d]

def deriv_g3_scaled(a, b, c, d):
    """
    Helper method for the scaled dataset.
    """
    return deriv_cost_func_3(a=a, b=b, c=c, d=d, x=x_scaled, y=y_scaled)

# Compute the optimal values of a, b, c and d for a cubic fit on the new dataset.
optimizers, steps = gradient_descent(
    initialpts=[0, 0, 0, 0],
    derivative=deriv_g3_scaled,
    alpha=0.001,
    epsilon=0.0001,
    iter_max=1000
)
a_star_cubic_scaled = optimizers[0]
b_star_cubic_scaled = optimizers[1]
c_star_cubic_scaled = optimizers[2]
d_star_cubic_scaled = optimizers[3]

cost_scaled_3 = cost_func_3(
    a_star_cubic_scaled,
    b_star_cubic_scaled,
    c_star_cubic_scaled,
    d_star_cubic_scaled,
    x_scaled,
    y_scaled
)

print(
    f"\nGradient descent result for cubic cost function on the scaled dataset:\n"
    f"  alpha = {0.001}\n"
    f"  epsilon = {0.0001}\n"
    f"  iter_max = 1000\n"
    f"  steps taken = {steps}\n"
    f"  optimal a = {a_star_cubic_scaled:.6f}\n"
    f"  optimal b = {b_star_cubic_scaled:.6f}\n"
    f"  optimal c = {c_star_cubic_scaled:.6f}\n"
    f"  optimal d = {d_star_cubic_scaled:.6f}\n"
    f"  cost = {cost_scaled_3:.6f}"
)

plot_data_and_cubic(
    x=x_scaled,
    y=y_scaled,
    a_star=a_star_cubic_scaled,
    b_star=b_star_cubic_scaled,
    c_star=c_star_cubic_scaled,
    d_star=d_star_cubic_scaled
)
