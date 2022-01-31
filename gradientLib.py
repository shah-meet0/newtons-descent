# Gradient Library contains functions which help estimate
# The first and second order derivatives of functions.

import numpy as np
from pathos.multiprocessing import ProcessPool
import functools
import time
import os


def time_it(func):
    """
    A decorator which when applied to a function makes it return the value and the time taken for execution.

    :param func: Function whose time is to be recorded.
    :return: Tuple containing function output and time taken to execute function.
    """

    @functools.wraps(func)
    def time_wrapper(*args, **kwargs):
        start_time = time.time()
        val = func(*args, **kwargs)
        end_time = time.time()
        tot_time = end_time - start_time
        # print(f'{func.__name__} took {tot_time} seconds to run.')
        return val, tot_time

    return time_wrapper


def partial_derivative(func, x, index, epsilon=10 ** -6, *args, **kwargs):
    """
    Calculates the numerical partial derivative using the finite difference method.

    :param func: Function whose partial derivative is to be calculated.
    :param x: List containing the argument values at which the partial derivative is to be calculated.
    :param index: The variable in x with respect to which the partial derivative is to be calculated. (Starts at 0).
    :param epsilon: Epsilon controls the error of estimation. Lower the better - but numerical errors will arise if it is too low.
    :param args: Additional arguments to be passed on to func.
    :param kwargs: Additional keyword arguments to be passed on to func.
    :return: Partial derivative of func at x.
    """
    upper = x.copy()
    lower = x.copy()
    upper[index] = upper[index] + epsilon
    lower[index] = lower[index] - epsilon
    difference = func(upper, *args, **kwargs) - func(lower, *args, **kwargs)
    return difference / 2 / epsilon


def gradient(func, x, epsilon=10 ** -6, *args, **kwargs):
    """
    Calculates the gradient [vector containing partial derivatives for all variables] for func at x.
    Not Parallelized.

    :param func: Function whose gradient is to be calculated.
    :param x: List containing the argument values at which the gradient is to be calculated.
    :param epsilon: Epsilon controls the error of estimation. Lower the better - but numerical errors will arise if it is too low.
    :param args: Additional arguments to be passed on to func.
    :param kwargs: Additional keyword arguments to be passed on to func.
    :return: Gradient for func at x.
    """
    n = len(x)
    gradient_vector = np.zeros(n)
    for index in range(n):
        index_derivative = partial_derivative(func, x, index, epsilon, *args, **kwargs)
        gradient_vector[index] = index_derivative

    return np.array(gradient_vector, dtype=np.float64)


def gradient_parallel(func, x, epsilon=10 ** -6, *args, **kwargs):
    """
    Calculates the gradient [vector containing partial derivatives for all variables] for func at x.
    Parallelized.

    :param func: Function whose gradient is to be calculated.
    :param x: List containing the argument values at which the gradient is to be calculated.
    :param epsilon: Epsilon controls the error of estimation. Lower the better - but numerical
    errors will arise if it is too low.
    :param args: Additional arguments to be passed on to func.
    :param kwargs: Additional keyword arguments to be passed on to func.
    :return: Gradient for func at x.
    """
    n = len(x)

    def gradient_finder(index):
        return partial_derivative(func, x, index, epsilon, *args, **kwargs)

    with ProcessPool(os.cpu_count() - 1) as p:
        gradient_vector = p.map(gradient_finder, range(n))

    return np.array(gradient_vector, dtype=np.float64)


def hessian(func, x, epsilon=10 ** -6, *args, **kwargs):
    """
    Calculates the Hessian [Matrix containing cross derivatives] of func at x.
    Not Parallelized.

    :param func: Function whose Hessian is to be calculated.
    :param x: List containing the argument values at which the Hessian is to be calculated.
    :param epsilon: Epsilon controls the error of estimation. Lower the better - but numerical
    errors will arise if it is too low.
    :param args: Additional arguments to be passed on to func.
    :param kwargs: Additional keyword arguments to be passed on to func.
    :return: Hessian of func at x
    """
    n = len(x)
    hessian_matrix = np.zeros((n, n))

    def gradient_getter(val):
        return gradient(func, val, epsilon, *args, **kwargs)

    for index in range(n):
        partial_gradient = partial_derivative(gradient_getter, x, index)
        hessian_matrix[index] = partial_gradient

    return hessian_matrix


def hessian_parallel(func, x, epsilon=10 ** -6, *args, **kwargs):
    """
    Calculates the Hessian [Matrix containing cross derivatives] of func at x.
    Parallelized.

    :param func: Function whose Hessian is to be calculated.
    :param x: List containing the argument values at which the Hessian is to be calculated.
    :param epsilon: Epsilon controls the error of estimation. Lower the better - but numerical
    errors will arise if it is too low.
    :param args: Additional arguments to be passed on to func.
    :param kwargs: Additional keyword arguments to be passed on to func.
    :return: Hessian of func at x
    """

    n = len(x)
    epsilon = 10 ** -6
    indices = [[i, j] for i in range(n) for j in range(i, n)]
    hessian_matrix = np.zeros((n, n))

    def second_partial_derivative(set_of_indexes):
        i = set_of_indexes[0]
        j = set_of_indexes[1]

        upper_j = x.copy()
        upper_j[j] = upper_j[j] + epsilon

        lower_j = x.copy()
        lower_j[j] = lower_j[j] - epsilon

        f_prime_i_upper = partial_derivative(func, upper_j, i, epsilon, *args, **kwargs)
        f_prime_i_lower = partial_derivative(func, lower_j, i, epsilon, *args, **kwargs)

        difference = f_prime_i_upper - f_prime_i_lower
        val = difference / 2 / epsilon
        return val

    with ProcessPool(os.cpu_count() - 1) as p:
        vec_hessian = p.map(second_partial_derivative, indices)
    hessian_matrix[np.triu_indices(n)] = vec_hessian  # Only calculate each entry once
    # so need to assign it to the lower triangle
    hessian_matrix = hessian_matrix + np.tril(hessian_matrix.T, -1)

    return hessian_matrix


def hessian_parallel_partial(func, x, epsilon=10 ** -6, *args, **kwargs):
    """
    Calculates the Hessian [Matrix containing cross derivatives] of func at x.
    Parallelized Partially.

    :param func: Function whose Hessian is to be calculated.
    :param x: List containing the argument values at which the Hessian is to be calculated.
    :param epsilon: Epsilon controls the error of estimation. Lower the better - but numerical
    errors will arise if it is too low.
    :param args: Additional arguments to be passed on to func.
    :param kwargs: Additional keyword arguments to be passed on to func.
    :return: Hessian of func at x
    """
    n = len(x)

    def gradient_getter(val):
        return gradient(func, val, epsilon, *args, **kwargs)

    def hessian_finder(index):
        return partial_derivative(gradient_getter, x, index, epsilon, *args, **kwargs)

    with ProcessPool(os.cpu_count() - 1) as p:
        hessian_matrix = p.map(hessian_finder, range(n))

    hessian_matrix = np.array(hessian_matrix, dtype=np.float64)
    return np.array(hessian_matrix, dtype=np.float64)
