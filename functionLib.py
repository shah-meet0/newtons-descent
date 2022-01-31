from math import *
import numpy as np

# Functions were sourced from
# https://en.wikipedia.org/wiki/Test_functions_for_optimization
# Contains mostly scalable functions.


def cross_multiple(x): # Minimised at 0,0
    return x[0] * x[1]


def ols_loss_fn(beta, dep_var, regressors):
    dep_var = np.array(dep_var)
    regressors = np.array(regressors)
    if len(beta) == 1:
        residuals = dep_var - beta * regressors
    else:
        residuals = dep_var - np.matmul(regressors, beta)
    ressq = residuals ** 2
    return sum(ressq)/len(beta)


def sum_of_squares(x):  # Minimized at 0 for each x
    val = 0
    for i in range(len(x)):
        val += x[i] ** 2
    return val


def rastrigin_func(x):  # Minimized at 0 for each x
    n = len(x)
    a = 10
    val = 0
    for i in range(n):
        val += x[i] ** 2 - a * cos(2 * pi * x[i])

    return a * n + val


def rosenbrock_func(x):  # Minimum at [1,1, ..., 1]
    n = len(x)
    val = 0
    for i in range(n-1):
        val += 100 * (x[i+1] - x[i] ** 2) ** 2 + (1-x[i]) ** 2
    return val


