# optimLib contains implementations of gradient descent and Newton's Method for functions
# In parallel and without parallel

from gradientLib import *
import numpy as np


class OptimalSummary:

    """
    OptimalSummary is a helper class which holds the Optimization Results from Gradient Descent and Newton's method
    and has a simple string representation for them.
    """

    def __init__(self, x, val, method, note):
        self.x_min = x
        self.value = val
        self.method = method
        self.note = note

    def __str__(self):
        a1 = f'Optimal value found at x = {np.around(self.x_min,4)} \n'
        a2 = f'The value is equal to {round(self.value, 4)}\n'
        a3 = f'The method used was {self.method}\n'
        return a1 + a2 + a3 + self.note


def l2norm(x_old, x_new):
    """
    Calculates the l2norm of the difference of the arguments.
    """
    diff = x_new - x_old
    diff_sq = diff ** 2
    sum_sq = np.sum(diff_sq)
    return np.sqrt(sum_sq)


def gradient_descent(func, x0, t=0.1, gradient_func=None, tol=10 ** -6, max_iter=1000, epsilon=10 ** -6,  *args, **kwargs):
    """
    Calculates the local minimum of a convex function using Gradient Descent.
    Not Parallelized.

    :param func: function whose local minimum is to be calculated.
    :param x0: Initial value from where to descend.
    :param t: Step size - Too low will cause slow convergence, too high may result in non-convergence. Default 0.1.
    :param gradient_func: Optional - Function which gives analytical gradient of func.
    :param tol: Tolerance level for successive iterations.
    :param max_iter: Maximum number of iterations before algorithm stops.
    :param epsilon: parameter used to calculate gradient
    :param args: Additional arguments to be passed on to func and gradient func.
    :param kwargs: Additional keyword arguments to be passed on to func and gradient func.
    :return: OptimalSummary containing Optimal X, Function Value at Optimal X, and more information about algorithm behaviour.
    """
    if gradient_func is None:
        x_old = np.array(x0)
        x_new = x_old - t * gradient(func, x0, epsilon=epsilon, *args, **kwargs)
        iterations = 1
        while l2norm(x_old, x_new) > tol and iterations < max_iter:
            x_old = x_new
            x_new = x_old - t * gradient(func, x_old, epsilon=epsilon, *args, **kwargs)
            iterations += 1

        if iterations == max_iter:
            print(f'Reached max iteration bound without convergence')
            return OptimalSummary(x_new, func(x_new, *args, **kwargs), 'Gradient Descent (Non Parallel)',
                                  'Max Iterations Reached \nGradient Estimated')
        else:
            return OptimalSummary(x_new, func(x_new, *args, **kwargs), 'Gradient Descent (Non Parallel)',
                                  f'Number of iterations = {iterations} \nGradient Estimated')
    else:
        x_old = np.array(x0)
        x_new = x_old - t * np.array(gradient_func(x0, *args, **kwargs))
        iterations = 1
        while l2norm(x_old, x_new) > tol and iterations < max_iter:
            x_old = x_new
            x_new = x_old - t * np.array(gradient_func(x_old, *args, **kwargs))
            iterations += 1

        if iterations == max_iter:
            print(f'Reached max iteration bound without convergence')
            return OptimalSummary(x_new, func(x_new, *args, **kwargs), 'Gradient Descent (Non Parallel)',
                                  'Max Iterations Reached')
        else:
            return OptimalSummary(x_new, func(x_new, *args, **kwargs), 'Gradient Descent (Non Parallel)',
                                  f'Number of iterations = {iterations}')


def gradient_descent_parallel(func, x0, t=0.1, tol=10 ** -6, max_iter=1000, epsilon=10 ** -6, *args, **kwargs):
    """
    Calculates the local minimum of a convex function using Gradient Descent.
    Gradient Estimation is Parallelized.

    :param func: function whose local minimum is to be calculated.
    :param x0: Initial value from where to descend.
    :param t: Step size - Too low will cause slow convergence, too high may result in non-convergence. Default 0.1.
    :param tol: Tolerance level for successive iterations.
    :param max_iter: Maximum number of iterations before algorithm stops.
    :param epsilon: parameter used to calculate gradient
    :param args: Additional arguments to be passed on to func.
    :param kwargs: Additional keyword arguments to be passed on to func. Must contain epsilon if args are passed.
    :return: OptimalSummary containing Optimal X, Function Value at Optimal X, and more information about algorithm behaviour.
    """
    x_old = np.array(x0)
    x_new = x_old - t * gradient_parallel(func, x0, epsilon=epsilon, *args, **kwargs)
    iterations = 1
    while l2norm(x_old, x_new) > tol and iterations < max_iter:
        x_old = x_new
        x_new = x_old - t * gradient_parallel(func, x_old, epsilon=epsilon, *args, **kwargs)
        iterations += 1

    if iterations == max_iter:
        print(f'Reached max iteration bound without convergence')
        return OptimalSummary(x_new, func(x_new, *args, **kwargs), 'Gradient Descent (Parallel)',
                              'Max Iterations Reached \nGradient Estimated')
    else:
        return OptimalSummary(x_new, func(x_new, *args, **kwargs), 'Gradient Descent (Parallel)',
                              f'Number of iterations = {iterations} \nGradient Estimated')


def newton_step(func, x, *args, **kwargs):
    """
    Calculates the Newtons Method iteration of a function.
    Non Parallelized.

    :param func: Function for which the iteration is to be calculated.
    :param x: Point at which the iteration is to be calculated.
    :param args: Additional arguments to be passed on to func.
    :param kwargs: Additional arguments to be passed on to func. Must contain epsilon if args are passed.
    :return: hesssian_inverse * gradient
    """
    grad = gradient(func, x, *args, **kwargs)
    hes = hessian(func, x, *args, **kwargs)
    try:
        hes_inv = np.linalg.inv(hes)
    except np.linalg.LinAlgError:
        print('Hessian not invertible, using Pseudo-Inverse')
        hes_inv = np.linalg.pinv(hes)
    # Could have alternatively used a perturbed inverse: H + c * I with c sufficiently large, I identity matrix.
    return np.matmul(hes_inv, grad)


def newton_step_given(x, grad_func, hes_func, *args, **kwargs):
    """
    Calculates the Newton step given an analytical expression for gradient and hessian functions.

    :param x: point at which step is to be calculated.
    :param grad_func: Analytical function for gradient.
    :param hes_func: Analyitcal function for Hessian.
    :param args: Additional arguments to be passed to grad_func and hes_func.
    :param kwargs: Additional arguments to be passed to grad_func and hes_func.
    :return: hessian_inverse * gradient.
    """
    grad = grad_func(x, *args, **kwargs)
    hes = hes_func(x, *args, **kwargs)
    n = len(x)
    if len(grad) != n or len(hes) != n or len(hes[0] != n):
        raise ValueError('Incorrect Length for provided Gradient or Hessian')

    try:
        hes_inv = np.linalg.inv(hes)
    except np.linalg.LinAlgError:
        print('Hessian not invertible, using Pseudo-inverse')
        hes_inv = np.linalg.pinv(hes)

    return np.matmul(hes_inv, grad)


def newton_step_parallel(func, x, *args, **kwargs):
    """
    Calculates the Newtons Method iteration of a function.
    Parallelized.

    :param func: Function for which the iteration is to be calculated.
    :param x: Point at which the iteration is to be calculated.
    :param args: Additional arguments to be passed on to func.
    :param kwargs: Additional arguments to be passed on to func. Must contain epsilon if args are passed.
    :return: hessian_inverse * gradient
    """
    grad = gradient_parallel(func, x, *args, **kwargs)
    hes = hessian_parallel(func, x, *args, **kwargs)
    try:
        hes_inv = np.linalg.inv(hes)
    except np.linalg.LinAlgError:
        print('Hessian not invertible, using Pseudo-inverse')
        hes_inv = np.linalg.pinv(hes)

    return np.matmul(hes_inv, grad)


def newtons_method(func, x0, grad_func=None, hes_func=None, tol=10**-6, max_iter=1000, epsilon=10 ** -6, *args, **kwargs):
    """
    Calculates the local minimum of a convex function using Newton's Method.
    Estimated Hessian and Gradient not parallelized.

    :param func: function whose local minimum is to be calculated.
    :param x0: Point at which Newton's Descent is started.
    :param grad_func: Optional: Function which gives the analytical gradient for func. Must provide hes_func if provided.
    :param hes_func: Optional: Function which gives the analytical Hessian for func. Must provide grad_func if provided.
    :param tol: Tolerance level for successive iterations.
    :param max_iter: Maximum number of iterations before algorithm stops.
    :param epsilon: parameter used to calculate gradient and Hessian
    :param args: Additional arguments to be passed on to func.
    :param kwargs: Additional keyword arguments to be passed on to func.
    :return: OptimalSummary containing Optimal X, Function Value at Optimal X, and more information about algorithm behaviour.
    """
    if grad_func is not None and hes_func is not None:
        x_old = np.array(x0)
        x_new = x_old - newton_step_given(x_old, grad_func, hes_func, *args, **kwargs)
        iterations = 1

        while l2norm(x_old, x_new) > tol and iterations < max_iter:
            x_old = x_new
            x_new = x_old - newton_step_given(x_old, grad_func, hes_func, *args, **kwargs)
            iterations += 1

        if iterations == max_iter:
            print(f'Reached max iteration bound without convergence')
            return OptimalSummary(x_new, func(x_new, *args, **kwargs), "Newton's Method (Non Parallel)",
                                  'Max Iterations Reached')
        else:
            return OptimalSummary(x_new, func(x_new, *args, **kwargs), "Newton's Method (Non Parallel)",
                                  f'Number of Iterations = {iterations}')
    else:
        x_old = np.array(x0)
        x_new = x_old - newton_step(func, x_old, *args, **kwargs)
        iterations = 1

        while l2norm(x_old, x_new) > tol and iterations < max_iter:
            x_old = x_new
            x_new = x_old - newton_step(func, x_old, epsilon=epsilon, *args, **kwargs)
            iterations += 1

        if iterations == max_iter:
            print(f'Reached max iteration bound without convergence')
            return OptimalSummary(x_new, func(x_new, *args, **kwargs), "Newton's Method (Non Parallel)",
                                  'Max Iterations Reached \nGradient and Hessian Estimated')
        else:
            return OptimalSummary(x_new, func(x_new, *args, **kwargs), "Newton's Method (Non Parallel)",
                                  f'Number of Iterations = {iterations} \nGradient and Hessian Estimated')


def newtons_method_parallel(func, x0, tol=10**-6, max_iter=1000, epsilon=10 ** -6, *args, **kwargs):
    """
    Calculates the local minimum of a convex function using Newton's Method.
    Estimated Hessian and Gradient parallelized.

    :param func: function whose local minimum is to be calculated.
    :param x0: Point at which Newton's Descent is started.
    :param tol: Tolerance level for successive iterations.
    :param max_iter: Maximum number of iterations before algorithm stops.
    :param epsilon: parameter used to calculate gradient and Hessian
    :param args: Additional arguments to be passed on to func.
    :param kwargs: Additional keyword arguments to be passed on to func.
    :return: OptimalSummary containing Optimal X, Function Value at Optimal X, and more information about algorithm behaviour.
    """
    x_old = np.array(x0)
    x_new = x_old - newton_step_parallel(func, x_old, epsilon=epsilon, *args, **kwargs)
    iterations = 1

    while l2norm(x_old, x_new) > tol and iterations < max_iter:
        x_old = x_new
        x_new = x_old - newton_step_parallel(func, x_old, epsilon=epsilon, *args, **kwargs)
        iterations += 1

    if iterations == max_iter:
        print(f'Reached max iteration bound without convergence')
        return OptimalSummary(x_new, func(x_new, *args, **kwargs), "Newton's Method (Parallel)",
                              'Max Iterations Reached \nGradient and Hessian Estimated')
    else:
        return OptimalSummary(x_new, func(x_new, *args, **kwargs), "Newton's Method (Parallel)",
                              f'Number of Iterations = {iterations} \nGradient and Hessian Estimated')

