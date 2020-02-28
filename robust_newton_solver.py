import numpy as np
import numpy.linalg as la

# Our imports:
from project_functions import *


def find_step_direction(P, f, df, ddf, xk, l):
    try:  # Try to find the optimal Newton direction.
        p = np.linalg.solve(ddf(P, xk, l), -df(P, xk, l))
        # print("\n\nFound Newton step.")
        if p@df(P, xk, l) > 0:
            # print("Newton not decreasing, using direct descent")
            p = -df(P, xk, l)
    except la.linalg.LinAlgError:  # If the Hessian is singular, perform a gradient descent step.
        # print("\n\nNewton not working, taking gradient descent step.\n")
        p = -df(P, xk, l)

    # print("descent:  ", p@df(P, xk, l))
    p = p/la.norm(p, 2)
    return p


def anulus_radii(line_lengths):
    # assert all(line_lengths > 0)  # Safety
    outer_radius = np.sum(np.abs(line_lengths))
    inner_radius = 2*np.max(np.abs(line_lengths)) - outer_radius
    if inner_radius < 0.0:
        inner_radius = 0.0
    return inner_radius, outer_radius


def line_search_step(P, f, df, ddf, theta_k, line_lengths, c1=0.01, c2=0.8, max_iter=50):
    # Finding step direction through Newton-method, or reverting back to Gradient descent:
    p = find_step_direction(P, f, df, ddf, theta_k, line_lengths)

    # print(f"The step direction is: {p}\n")

    # initial step length
    a_max = 1e100
    a_min = 0
    a = 1
    j = 0

    # Finding an acceptable step length 'a' for the Wolfe conditions:
    while j < max_iter:
        j += 1

        f_at_step = f(P, theta_k + a * p, line_lengths)
        f_descent = f(P, theta_k, line_lengths) + a * c1 * df(P, theta_k, line_lengths) @ p

        directional_derivative_at_step = df(P, theta_k + a * p, line_lengths).T @ p
        directional_derivative_reduction = c2 * df(P, theta_k, line_lengths).T @ p

        if (f_at_step > f_descent)  or (directional_derivative_at_step > -directional_derivative_reduction):
            # Meaning (first wolfe-requirement was not passed)
            # or (The strong part of the second wolfe-req not passed)
            a_max = a
            a = 1 * a_max / 4 + 3 * a_min / 4

        elif directional_derivative_at_step <= directional_derivative_reduction: # Meaning second wolfe-requirement was not passed
            a_min = a
            if a_max > 1e99:
                a *= 2
            else:
                a = (a_max + a_min) / 2
        else:
            break

    return p, a


def second_order_step(P, f, ddf, thetas, line_lengths):
    """
    Function finding a descent direction for the Hessian of a function, through eigenvalue decomposition.
    :param H: Hessian of a function.
    :return: (Hopefully) Unit-length eigenvector of the Hessian, with the lowest value eigenvalue.
    """
    H = ddf(P, thetas, line_lengths)
    w, v = la.eigh(H, 'L')

    p, a = np.array(v[:, 0]), 1.0

    # If the lowest value eigenvalue is not negative, we are not at a saddle point. Thus we are at a minimzer.
    if w[0] < 0.0:
        # With a short enough step size, we are guaranteed to reduce the function value when we are at a saddle point
        # and go in a direction p with a negative eigenvalue with respect to the hessian H.
        while f(P, thetas + a * p, line_lengths) >= f(P, thetas, line_lengths):
            a *= 0.5
        return p, a, False

    else:
        return p, a, True


def robust_minimizer(P, f, df, ddf, theta_k, line_lengths, vtol=1e-8, gtol=1.0e-8, max_iter=50, c1=0.01, c2=0.8):
    # f is the function to be minimized
    # P is point to be reached
    # df is "nabla"f (a function)
    # B(xk) is a function returning Bk,
    # line_lengths is an array of the line lengths for each arm.
    # the Bk is the matrix used in defining the direction vector p in step k.
    # Gradient descent method is Bk = Id. Newtons method uses Bk = hessian(fk).
    # vtol: Value tolerance, for the value of f.
    # gtol: Gradient tolerance, for the norm of the gradient of f.

    l = line_lengths
    i = 0

    anulus_inner, anulus_outer = anulus_radii(line_lengths)
    target_feasible = anulus_inner <= la.norm(P, 2) < anulus_outer
    converged = False

    # We know that f is non-negative, bounded from below by zero: Assuming that our target point P
    # is in the feasible domain, we should be able to reach it with the robot arms.

    if target_feasible:
        while i < max_iter and converged is not True:
            i += 1

            function_value = f(P, theta_k, l)

            # Check if we have a feasible point, and if close enough, return.
            if function_value < vtol:  # or (not target_feasible and grad_norm < gtol):
                # We have found a minimizer inside the feasible domain:
                break
            elif f(P, theta_k, l) > vtol and la.norm(df(P, theta_k, l)) < gtol:
                # We are at a stationary point. If there are no negative eigenvalues, we are at a minimizer.
                print("Taking second order step!\n")
                p, a, converged = second_order_step(P, f, ddf, theta_k, l)
                theta_k += a * p
                continue

            else:
                # If none of the above apply, perform regular step:
                p, a = line_search_step(P, f, df, ddf, theta_k, line_lengths)
                theta_k += a * p
    else:
        while i < max_iter:
            i += 1

            # Target point not feasible, cannot easily detect saddle points, and revert to simpler method.
            # Stopping criterion based on the norm of the gradient:
            grad_norm = la.norm(df(P, theta_k, l), 2)
            if grad_norm < gtol:
                break
            p, a = line_search_step(P, f, df, ddf, theta_k, line_lengths)
            theta_k += a * p

    print(f"Solution found. Number of steps: {i}, function value: {f(P, theta_k, l):.3e}")
    return theta_k, f(P, theta_k, l)
