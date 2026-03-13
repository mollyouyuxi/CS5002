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

    steps = 0 # the number of gradient descent iterations (also the number of steps) 
    x = initialpts # current point is at initial point 
    while steps < iter_max: # maximum number of iteration to avoid infinete loop 
        x_update = x - alpha * derivative(x) # update location 
            
        if abs(x_update-x) < epsilon: # check if we've reach the tolerence, if yes, we've reach local minimum, end the loop 
                return x_update, steps+1 
            
        x = x_update # update the current point after taking one step 
        steps+=1 # update number of steps 
    
    if steps == iter_max: #check if maximum iteration force the loop to end 
        print("Reach the maximum iterations!") 
        
    return x, steps # reach the maximum iteration number, return the x
    

def f1(x):
    """
    cumpute the value of quadratic function f(x) = x^2

    Args: 
        x: input value of x

    Returns:
        the result of the quadratic function
    """
    return x**2


def deriv_f1(x):
    """
    cumpute the value of quadratic function f(x) = x^2

    Args: 
        x: input value of x

    Returns:
        the result of the quadratic function
    """
    return 2*x


def f2(x):
    """
    cumpute the value of quadratic function f(x) = x^2-2x+3

    Args: 
        x: input value of x

    Returns:
        the result of the quadratic function
    """
    return x**2-2*x+3


def deriv_f2(x):
    """
    Compute the derivative of quadratic function f2(x)

    Args:
        x: input value of x
    Rerturns
        the result of the f2(x) derivative at given x
    """
    return 2*x-2


def f3(x):
    """
    Compute function f(x) = sin(x) + cos(√2x), for 0 <= x <= 10. Raise error if x is out of range 

    Args:
        x: input value of x

    Returns: the result of function f(x) = sin(x) + cos(√2x)
    """
    if x < 0 or x > 10: 
        raise ValueError("x must be in range [0, 10]")
    return math .sin(x) + math.cos(math.sqrt(2)*x)


def deriv_f3(x):
    return math.cos(x) - math.sqrt(2)*math.sin(math.sqrt(2)*x)

# Create helper method to compute the approximate derivative
# This helper method allow us to maintiain the signature and implementation of gradient_descent method
def approximateDerivatives(initials, function, h=0.001):
    """
    Compute the derivatives approximately by limh→0 [f(x+h)-f(x)]/h

    Args:
        x: the value of x
        function: the function to compute the dirivates of 
        h: 0.001 by default

    Returns:
        the derivatives of the function at point x    
    """
    derivative = (function(initials+h) - function(initials)) / h
    return derivative
    


def approx_f1(x):
    return approximateDerivatives(initials=x, function=f1)
def approx_f2(x):
    return approximateDerivatives(initials=x, function=f2)
def approx_f3(x):
    return approximateDerivatives(initials=x, function=f3)
     

def plot_f3():  
    # Initializing a new figure 
    plt.figure()
    
    # Generate x value in range [0, 10]
    x = np.linspace(0, 10, 60)
    
    # Compute y
    y = [f3(xi) for xi in x]

    # Plot the function
    plt.plot(x, y)

    # Setting the plot range, lable and title
    plt.xticks(np.arange(-2, 11, 2))
    plt.yticks(np.arange(0,5,1))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("f(x) = sin(x) + cos(√2x)")
    plt.show()


def plot_opt(x_inis, function, derivative, title, lower_domain = -10, upper_domain = 10, alpha= 0.1, epsilon = 0.001):
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
    plt.figure()

    # generate x values within range [lower_domain, upper_domain]
    x = np.linspace(lower_domain, upper_domain, 60)

    # compute y values
    y = [function(xi) for xi in x]

    # plot the quadratic function 
    plt.plot(x, y)

    # cast the x_ini into a list
    if isinstance(x_inis, (float, int)):
        x_inis = [x_inis]

    # compute opitimizer for each different starting point 
    for x_ini in x_inis:
        # compute the minimum point by gradient desecent
        x_min, steps = gradient_descent(x_ini, derivative, alpha = alpha, epsilon = epsilon) # calculate the x axis of local minimum by gradient desecent algorithm 
        y_min = function(x_min) # compute the y axis of local minimum 


        # plot the optimizer 
        plt.scatter(x_min, y_min, color="red",)
        # mark the optimizer 
        plt.annotate(f"Min. ({x_min:.1f}, {y_min:.1f})", (x_min, y_min), xytext=(15,-5), textcoords="offset points")



    # Setting the plot range, lable and title
    plt.xticks(np.arange(lower_domain, upper_domain, 1))
    plt.yticks(np.arange(0,11,5))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)



def main():
    # plotting the function and its optimizer 
    plot_opt(x_inis=-5, function=f1, derivative=deriv_f1, title = "f(x) = x^2")
    plot_opt(x_inis=-5, function=f2, derivative=deriv_f2, title = "f(x) = x^2-2x+3")


    print(f"\n********Generated Output for Gradient Desecent Algorithm********\n") 

    print(f"***Fixing step size = 0.1, tolerance = 0.001***\n") 

    # Compute the optimizers with different initial value -3, 3

    #Compute optimizers for quadratic function f(x) = x^2 of starting at x = -3
    f1_minx1, steps = gradient_descent(initialpts=-3, derivative=deriv_f1)
    f1_miny1 = f1(f1_minx1)

    # Compute optimizers for quadratic function f(x) = x^2 of starting at x = 3
    f1_minx2, steps = gradient_descent(initialpts=3, derivative=deriv_f1)
    f1_miny2 = f1(f1_minx2)

    # Compute the optimizer for quadratic function f(x) = x^2-2x+3 of starting at x = -3
    f2_minx1, steps = gradient_descent(initialpts=-3, derivative=deriv_f2)
    f2_miny1 = f2(f2_minx1)

    # Compute the optimizer for quadratic function f(x) = x^2-2x+3 of starting at x = -3
    f2_minx2, steps = gradient_descent(initialpts=3, derivative=deriv_f2)
    f2_miny2 = f2(f2_minx2)

    # Generate the output of optimizers with different initial value
    print(f"For f(x) = x^2, when starting at x = 3, the optimizer is ({f1_minx1:.3f},{f1_miny1:.3f}).\nHowever, when starting at x = -3, the optimizer is ({f1_minx2:.3f}, {f1_miny2:.3f}).\n")
    print(f"For f(x) = x^2-2x+3, when starting at x = 3, the optimizer is ({f2_minx1:.3f},{f2_miny1:.3f}).\nHowever, when starting at x = -3, the optimizer is ({f2_minx2:.3f}, {f2_miny2:.3f}).\n")


    # Compute optimizers with different step size 1, 0,001, 0.0001
    print(f"***Starting at x = 3, with fixed tolerance = 0.001***\n") 
    stepSizes = [1, 0.001, 0.0001]
    x_ini = 3
    
    # Compute optimizers for quadratic function f(x) = x^2 with step size 1, 0.001, 0.0001 respectively 
    print(f"For f(x) = x ^2\n")
    for f1_size in stepSizes:
        f1_minx_dffstep, steps = gradient_descent(initialpts=x_ini, derivative=deriv_f1, alpha=f1_size)
        f1_miny_dffstep = f1(f1_minx_dffstep)
        print(f"When step size is {f1_size}, after taking {steps} steps, we've reached the optimizer ({f1_minx_dffstep:.3f},{f1_miny_dffstep:.3f})\n")
    
    # Compute optimizers for quadratic function f(x) = x^2-2x+3 with step size 1, 0,001, 0.0001 respectively 
    print(f"For f(x) = x^2 -2x+3\n")
    for f2_size in stepSizes:
        f2_minx_dffstep, steps = gradient_descent(initialpts=x_ini, derivative=deriv_f2, alpha=f2_size)
        f2_miny_dffstep=f2(f2_minx_dffstep)
        print(f"When step size is {f2_size}, after taking {steps} steps, we've reached the optimizer ({f2_minx_dffstep:.3f},{f2_miny_dffstep:.3f})\n")


    # Compute the optimizers with different tolerence 0.1, 0.01, 0.0001
    print(f"***Starting at x = 3 with fixed step size = 0.1***\n")
    tolerence = [0.1, 0.01, 0.0001]
    x_ini = 3 

    # Generate optimizer for quadratic function f(x) = x^2 with tolerence 1, 0.001, 0.0001 respectively 
    print(f"For f(x) = x^2:\n")
    for epsilon in tolerence:
        f1_minx_dfftol, steps = gradient_descent(initialpts=x_ini, derivative=deriv_f1, epsilon=epsilon)
        f1_miny_dfftol = f1(f1_minx_dfftol)
        print(f"When tolerence is {epsilon}, after taking {steps} steps, we've reached the optimizer ({f1_minx_dfftol:.3f},{f1_miny_dfftol:.3f})\n")

    # Generate optimizer for quadratic function f(x) = x^2-2x+3 with tolerence 1, 0.001, 0.0001 respectively 
    print(f"For f(x) = x^2-2x+3:\n")
    for epsilon in tolerence:
        f2_minx_dfftol, steps = gradient_descent(initialpts=x_ini, derivative=deriv_f2, epsilon=epsilon)
        f2_miny_dfftol = f2(f2_minx_dfftol)
        print(f"When tolerence is {epsilon}, after taking {steps} steps, we've reached the optimizer ({f2_minx_dfftol:.3f},{f2_miny_dfftol:.3f})\n")       

    
    # Compute optimizers for function f(x) = sin(x) + cos(√2x) with different starting point x = 1, 4, 5 ,7
    print(f"***For f(x) = sin(x) + cos(√2x) with fixing step size = 0.1, tolerance = 0.0001***\n") 
    for x in {1,4,5,7}:
        f3_x, f3_steps = gradient_descent(initialpts=x, derivative=deriv_f3, epsilon=0.0001)
        f3_y = f3(f3_x)
        print(f"when starting at x = {x}, the optimizer is ({f3_x:.3f},{f3_y:.3f}).\n")

    # # plot the function f(x) = sin(x) + cos(√2x)
    # plot_f3()

    # Plot the optimizer for function f(x) = sin(x) + cos(√2x), staring at x = 1, 4, 5 ,7
    plot_opt(x_inis=[1, 4, 5, 7], function=f3, derivative=deriv_f3, epsilon=0.0001, lower_domain=0, upper_domain=10, title= "f(x) = sin(x) + cos(√2x)")

    # Compute the approximate derivative 
    print(f"***Derivative Approximation for single-variable function***\n") 
    # Compute the actual and approximate dirivatives for f(x) = x^2
    print(f"For f(x)=x^2:\n")
    for x in {1,4,5,7}:
        # Compute the actual derivative
        actual_dev = deriv_f1(x)
        # Compute the approximate derivative
        approx_dev = approximateDerivatives(x, function=f1)
        # Generate the output
        print(f"When x = {x}, the acutual derivative is {actual_dev:.3f}, the approxiamate derivative is {approx_dev:.3f}\n")

    # Compute the actual and approximate dirivatives for f(x) = x^2
    print(f"\nFor f(x)=x^2-2x+3:\n")
    for x in {1,4,5,7}:
        # Compute the actual derivative
        actual_dev = deriv_f2(x)
        # Compute the approximate derivative
        approx_dev = approximateDerivatives(x, function=f2)
        # Generate the output
        print(f"When x = {x}, the acutual derivative is {actual_dev:.3f}, the approxiamate derivative is {approx_dev:.3f}\n")

    # Compute the gradient descent optimizer by using the approximate derivative 
    print(f"***Gradient Descent Optimizer by Using Derivative Approximation***\n") 
    
    # Compute the gradient descent optimizer by using the approximate derivative for f(x) = x^2
    f1_appx, steps = gradient_descent(initialpts=3, derivative=approx_f1)
    f1_appy = f1(f1_appx)
    print(f"***For f(x) = x^2, starting at x = 3 with fixing step size = 0.1, tolerance = 0.001***\nBy taking {steps} steps, we've reach the optimizer ({f1_appx:.3f},{f1_appy:.3f})\n") 

    # Compute the gradient descent optimizer by using the approximate derivative for f(x) = x^2-2x+3
    f2_appx, steps = gradient_descent(initialpts=3, derivative=approx_f2)
    f2_appy = f2(f2_appx)
    print(f"***For f(x) = x^2-2x+3, starting at x = 3 with fixing step size = 0.1, tolerance = 0.001***\nBy taking {steps} steps, we've reach the optimizer ({f2_appx:.3f},{f2_appy:.3f})\n") 

    plt.show()



if __name__ == "__main__":
    # # Run the test
    main()

