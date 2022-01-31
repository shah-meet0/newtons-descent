from gradientLib import *
from optimLib import *
from functionLib import *

if __name__ == '__main__':

    '''''''''''''''''''''''''''' Gradient Calculation + Gradient Descent '''''''''''''''''''''''''''''''''
    print(f'Gradient Calculation')
    print('All results for the sum of squares (sphere) function.')
    input('Press Enter to continue')
    print('Please wait...')

    initial_x0 = [i for i in range(300)]
    initial_x1 = [i for i in range(2000)]

    gradient_timed = time_it(gradient)  # Makes it so that time is returned alongside value.
    gradient_parallel_timed = time_it(gradient_parallel)

    gradient0_np, time0_np = gradient_timed(sum_of_squares, initial_x0)
    gradient0_p, time0_p = gradient_parallel_timed(sum_of_squares, initial_x0)

    gradient1_np, time1_np = gradient_timed(sum_of_squares, initial_x1)
    gradient1_p, time1_p = gradient_parallel_timed(sum_of_squares, initial_x1)

    time_improved_0 = (time0_np - time0_p)/time0_np * 100
    time_improved_1 = (time1_np - time1_p)/time1_np * 100

    print(f'When size of x is {len(initial_x0)}, parallelization improves gradient calculation'
          f' performance by {round(time_improved_0, 2)}%')
    print(f'When size of x is {len(initial_x1)}, parallelization improves gradient calculation'
          f' performance by {round(time_improved_1, 2)}%\n')

    input('Press Enter to continue')
    print('Gradient Descent\n')
    print('Example output')

    print(gradient_descent(sum_of_squares, t = 0.3, x0= [1, 2, 3]))
    print()
    input('Press Enter to continue')
    print('Please wait...')

    gradient_descent_timed = time_it(gradient_descent)
    gradient_descent_parallel_timed = time_it(gradient_descent_parallel)

    initial_x2 = [i for i in range(100)]
    initial_x3 = [i for i in range(500)]

    gradient_des_2_np, time2_np = gradient_descent_timed(sum_of_squares, initial_x2, t=0.3)
    gradient_des_2_p, time2_p = gradient_descent_parallel_timed(sum_of_squares, initial_x2, t=0.3)

    gradient3_np, time3_np = gradient_descent_timed(sum_of_squares, initial_x3, t=0.3)
    gradient3_p, time3_p = gradient_descent_parallel_timed(sum_of_squares, initial_x3, t=0.3)

    time_improved_2 = (time2_np - time2_p) / time2_np * 100
    time_improved_3 = (time3_np - time3_p) / time3_np * 100

    print(f'When size of x is {len(initial_x2)}, parallelization improves gradient descent'
          f' performance by {round(time_improved_2, 2)}%')
    print(f'When size of x is {len(initial_x3)}, parallelization improves gradient descent'
          f' performance by {round(time_improved_3, 2)}%\n')

    '''''''''''''''''''''''''''' Hessian Calculation + Newton's Method '''''''''''''''''''''''''''''''''
    print(f'Hessian Calculation')
    input('Press Enter to continue')
    print('Please wait...')

    initial_x4 = [1 for i in range(30)]
    initial_x5 = [1 for i in range(100)]

    hessian_timed = time_it(hessian)
    hessian_parallel_timed = time_it(hessian_parallel)

    hessian4_np, time4_np = hessian_timed(sum_of_squares, initial_x4)
    hessian4_p, time4_p = hessian_parallel_timed(sum_of_squares, initial_x4)

    hessian5_np, time5_np = hessian_timed(sum_of_squares, initial_x5)
    hessian5_p, time5_p = hessian_parallel_timed(sum_of_squares, initial_x5)

    time_improved_4 = (time4_np - time4_p) / time4_np * 100
    time_improved_5 = (time5_np - time5_p) / time5_np * 100

    print(f'When size of x is {len(initial_x4)}, parallelization improves Hessian calculation'
          f' performance by {round(time_improved_4, 2)}%')
    print(f'When size of x is {len(initial_x5)}, parallelization improves Hessian calculation'
          f' performance by {round(time_improved_5, 2)}%\n')

    input('Press Enter to continue')
    print("Newton's Method\n")
    print('Example output')

    print(newtons_method(sum_of_squares, [1, 2, 3]))
    print()
    input('Press Enter to continue')
    print('Please wait...')

    newtons_timed = time_it(newtons_method)
    newtons_parallel_timed = time_it(newtons_method_parallel)

    initial_x6 = [1 for i in range(30)]
    initial_x7 = [1 for i in range(100)]

    newton6_np, time6_np = newtons_timed(sum_of_squares, initial_x6)
    newton6_p, time6_p = newtons_parallel_timed(sum_of_squares, initial_x6)

    newton7_np, time7_np = newtons_timed(sum_of_squares, initial_x7)
    newton7_p, time7_p = newtons_parallel_timed(sum_of_squares, initial_x7)

    time_improved_6 = (time6_np - time6_p) / time6_np * 100
    time_improved_7 = (time7_np - time7_p) / time7_np * 100

    print(f"When size of x is {len(initial_x6)}, parallelization improves Newton's method"
          f' performance by {round(time_improved_6, 2)}%')
    print(f"When size of x is {len(initial_x7)}, parallelization improves Newton's method"
          f' performance by {round(time_improved_7, 2)}%\n')

    print('Thank you!')