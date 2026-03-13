import matplotlib.pyplot as plt
import numpy as np
import math



def gradient_descent(initialpts, derivative, alpha = 0.1, epsilon = 0.001, iter_max = 1000):
    """
    Compute the gradient descent of a function, the funtion could be of single variable or multi variables

    Args:
        initialpts: inintial point to start the function, it can be either single-dimensional or multi-dimensional points 
        derivative: derivative of the function 
        alpha: the step size, 0.1 by default
        epsilon: the torlerance for stopping, 0.001 by default
        iter_max: maximum number of iteration, 1000 by default

    Returns: 
        the final position after gradient descent, nums of steps it takes to reach the position  
    """
    
    #Cast the single-dimensional point into a list
    if isinstance (initialpts, (int, float)):
        initialpts = [initialpts]

    # keep track of the current position, starting from the initial position 
    cur = initialpts.copy()
    # Keep track of the steps we've took, initialize it to zero
    steps = 0 
    # Initialize an array to store the updated position of each coordinates
    update = [0] * len(initialpts)
    
    while steps < iter_max:

        #Compute the derivatives for every coordinate respectively, using approximiate derivatives function 
        derivs = derivative(*cur)

        # Update each coordinates by using gradient descent rule calculated by approximate derivatives 
        for i in range(len(initialpts)):
            update[i] = cur[i] - alpha * derivs[i]        
        # Tracking the steps, for each round of update, steps increment by 1 
        steps = steps+1

        # Check if we've reached the tolerence after each round of update
        # Set convergence is True by default. if any of the coordinates do not converge, set convergence to False 
        convergence = True
        for j in range(len(initialpts)):
            if abs(update[j]-cur[j])>epsilon:
                convergence = False
        # Continue the gradient descent iteration if any of the coordinates do not converge, else, convergence is true, break the while loop, return the current position and steps
        if convergence == True:
            # update the current position 
            cur = update.copy()
            break

        cur = update.copy()
    
    # Check whether we've reach the maximum iteration number 
    if steps == iter_max:
        print("Reach the maximum iterations!")

    return cur, steps   


# Create helper method to compute the approximate derivative
# This helper method allow us to maintiain the signature and implementation of gradient_descent method

def approximateDerivatives(initials, function, h=0.001):
    """
    Compute the derivatives approximately by limh→0 [f(x+h)-f(x)]/h

    Args:
        initials: the value of initial points 
        function: the function to compute the dirivates of 
        h: 0.001 by default

    Returns:
        the derivatives of the function at point x    
    """
    # Cast the initial point to list for single variable condition
    if isinstance(initials, (int, float)):
        initials = [initials]

    # Initialize derivative list 
    derivatives = [0] * len(initials)

    # Derivative approximation for each coordinates 
    for i in range(len(initials)):
        # Creating a copy of the initial points which allow us to modify the ith coordinates 
        temp = initials.copy()
        temp[i] = temp[i] + h
        derivatives[i] = (function(*temp) - function(*initials)) / h
    return derivatives
    


def f4(x,y):
    return x**2 + y**2

def approx_f4(x,y):
    return approximateDerivatives(initials=[x,y], function=f4)

# # Designing the additional test function and compute its approximate derivative
# def testf1(x):
#     return x**2

# def approx_testf1(x):
#     return approximateDerivatives(initials=[x], function=testf1)

# def testf2(x,y,z):
#     return x**2+y**2+z**2

# def approx_testf2(x,y,z):
#     return approximateDerivatives(initials=[x,y,z], function=testf2)


def plot_2var(initialpts, function, derivative, title, lower_domain = -10, upper_domain = 10, alpha= 0.1, epsilon = 0.001):
    """
    Plot the x point where we reach the gradient descent minimum, and the function

    Args: 
        x_inis: initial point to start to find the optimizer, it can be a single point or a list of points 
        function: the function to be plotted 
        derivative: the derivative of this function 
        title: title of this plot
        lower_domain: the lower bound of the x domain, -10 by defaut
        upper_domain: the upper bound of the x domain, 10 by defaut 

    """
    # Initializing a new figure 
    two_var_plot=plt.figure()

    # generate value for each coordinate within range [-10, 10]
    x = np.linspace(lower_domain, upper_domain, 60)
    y = np.linspace(lower_domain, upper_domain, 60)

    # Build combination for 2-D point
    X, Y = np.meshgrid(x,y)
    # Compute value for the function
    func = function(X, Y)

    # creating 3-D axes
    ax = two_var_plot.add_subplot(projection = '3d')
    # plot the function 
    ax.plot_surface(X, Y, func)


    # compute the minimum point by gradient desecent
    local_optimizer, steps = gradient_descent(initialpts=initialpts, derivative=derivative, alpha = alpha, epsilon = epsilon) # calculate the x axis of local minimum by gradient desecent algorithm 
    x_min = local_optimizer[0]
    y_min = local_optimizer[1]
    func_min = function(*local_optimizer) 
    # plot the optimizer 
    ax.scatter(x_min, y_min, func_min, color="red")
    # mark the optimizer 
    ax.text(x_min+5, y_min, func_min-2, f"Min ({x_min:.2f}, {y_min:.2f}, {func_min:.2f})", color="red")


    # # Setting the plot range, lable and title
    # plt.xticks(np.arange(lower_domain, upper_domain, 1))
    # plt.yticks(np.arange(0,11,5))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    ax.set_title(title)

    plt.show()





def main():
    # Generate output for f(x, y) = x^2 + y^2
    optimizers, steps = gradient_descent(initialpts=[3,3], derivative=approx_f4)
    # Rounded the output for further implementation 
    rounded_optimizers = [math.ceil(x * 1000) / 1000 for x in optimizers]
    print(f"For function f(x,y)=x^2+y^2, with fixed step size=0.1, tolerance=0.001.\nStarting at x=3, y=3, the algorithm converges to the point {rounded_optimizers} after {steps} steps")
    # Plot f(x, y) = x^2 + y^2 and its optimizer
    plot_2var(initialpts=[3,3], function=f4, derivative=approx_f4, title="f=x^2+y^2")

#     # Testing the algorithm by implement single-varaible function and triple-variable function
#     optimizers_test1, steps_test1 = gradient_descent(initialpts=3, derivative=approx_testf1)
#     rounded_optimizers_test1 = [math.ceil(x * 100) / 100 for x in optimizers_test1]
#     print(f"For function f(x)=x^2, we start at x=3, and reach the optimizer {rounded_optimizers_test1} after {steps_test1} steps")
#    # Testing the algorithm by implement single-varaible function and triple-variable function
#     optimizers_test2, steps_test2 = gradient_descent(initialpts=[3,3,3], derivative=approx_testf2)
#     rounded_optimizers_test2 = [math.ceil(x * 100) / 100 for x in optimizers_test2]
#     print(f"For function f(x)=x^2+y^2+z^2, we start at x=3, y=3, z=3 and reach the optimizer {rounded_optimizers_test2} after {steps_test2} steps")


if __name__ == "__main__":
    # Run the test
    main()
