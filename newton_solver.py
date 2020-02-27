# optimizing using gradient descent and newton
import numpy as np
import numpy.linalg as la
import visualization
from matplotlib import pyplot as plt


def find_step_direction(f, df, xk, B, P, l):
    try:  # Try to find the optimal Newton direction.
        p = np.linalg.solve(B(P, xk, l), -df(P, xk, l))
        print("\n\nFound Newton step.")
        if p@df(P, xk, l) > 0:
            print("Newton not decreasing, using direct descent")
            p = -df(P, xk, l)
    except la.linalg.LinAlgError:  # If the Hessian is singular, perform a gradient descent step.
        print("\n\nNewton not working, taking gradient descent step.\n")
        p = -df(P, xk, l)

    print("descent:  ", p@df(P, xk, l))
    p = p/la.norm(p, 2)
    return p


def _id(P,x,l):
    n = x.size
    return np.identity(n)


def find_minimum(f, P, df, theta_0, B, line_lengths, tol=1e-6, max_iter=50, c1=0.01, c2=0.8):
    # f is function to be minimized
    # P is point to be reached
    # df is "nabla"f (a function)
    # B(xk) is a function returning Bk,
    # line_lengths is an array of the line lengths for each arm.
    # the Bk is the matrix used in defining the direction vector p in step k.
    # Gradient descent method is Bk = Id. Newtons method uses Bk = hessian(fk).
    l = line_lengths
    i = 0
    theta_k = theta_0

    # We know that f is non-negative, bounded from below by zero:
    while i < max_iter and la.norm(df(P, theta_k, l)) > tol and f(P, theta_k, l) > tol:
        i += 1
        p = find_step_direction(f, df, theta_k, B, P, line_lengths)

        print(f"The step direction is: {p}\n")

        # initial step length
        a_max = 1e100
        a_min = 0
        a = 1
        j = 0

        # Plot first wolfe tests
        A=[]
        F=[]
        b=a
        # plt.plot(np.array([0,b]),np.array([f(P,theta_k,l),f(P,theta_k,l)+b*df(P,theta_k,l)@p]), label=r'derivative at $\theta^{(k)}$')
        # plt.plot(np.array([0,b]),np.array([f(P,theta_k,l),f(P,theta_k,l)+c1*b*df(P,theta_k,l)@p]), label="First wolfe condition")

        # Finding an acceptable step length 'a' for the Wolfe conditions:
        while j < max_iter:
            # For wolfe plotting
            A.append(a)
            F.append(f(P, theta_k+a*p, l))

            print(f'\nj={j}')
            print(f'a={a}')
            print(f'possible improvement={f(P,theta_k,l)-f(P,theta_k+a*p,l)}')
            print(f'First wolfe requirement: {f(P,theta_k+a*p,l)}, <= {f(P,theta_k,l)} + {a*c1*df(P,theta_k,l)@p}')

            # visualization.display_robot_arm(line_lengths=l, turn_angles=theta_k, target_point=P)

            j += 1
            if f(P, theta_k + a*p, l) > f(P, theta_k, l) + a*c1*df(P, theta_k, l)@p:
                print("First wolfe=Not passed")
                a_max = a
                a = a_max/2 + a_min/2
            elif df(P, theta_k + a*p, l).T@p <= c2*df(P, theta_k, l).T@p:
                a_min = a
                if a_max > 1e99:
                    a *= 2
                else:
                    a = (a_max + a_min)/2
            else:
                break

        # Wolfe plotting
        # plt.plot(np.array(A), np.array(F))
        # plt.legend()
        # plt.show()
        print(f"Norm of gradient at step {i}: {np.linalg.norm(df(P, theta_k, l))}")
        theta_k = theta_k + a*p

    print("Solution found. Number of steps=", i)
    return theta_k, f(P, theta_k, l)
