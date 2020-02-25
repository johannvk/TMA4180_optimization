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


def id(P,x,l):
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
    improvement = tol+1

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

        # improvement = f(P, theta_k, l) - f(P, theta_k + a*p, l)
        # print("improvement from this step=", improvement, "a=", a)
        print(f"Norm of gradient at step {i}: {np.linalg.norm(df(P, theta_k, l))}")
        theta_k = theta_k + a*p

    print("Solution found. Number of steps=", i)
    return theta_k, f(P, theta_k, l)


def x_fnc(theta, l, active_index=0):
    # Initial value for x:
    x = 0.0
    # Sum up the first "k-1" theta-values if the active index is not the first one:
    phi = np.sum(theta[0:active_index])
    for i in range(active_index, len(l)):
        li = l[i]
        phi += theta[i]
        x += li*np.cos(phi)
    return x


def dx(k, theta, l):
    #  Can do as before, as the y_fnc has been modified, with the "active_index" arguments,
    #  which decides when to count the sum from. But also adding the right theta-values to start with.
    return -y_fnc(theta, l, active_index=k)


def ddx(k, m, theta, l):
    max_index = max(k, m)
    return -x_fnc(theta, l, active_index=max_index)


def y_fnc(theta, l, active_index=0):
    # Initial value for y:
    y = 0.0
    # Sum up the first "k-1" theta-values if the active index is not the first one:
    phi = np.sum(theta[0:active_index])

    for i in range(active_index, len(l)):
        li = l[i]
        phi += theta[i]
        y += li*np.sin(phi)
    return y


def dy(k, theta, l):
    #  Can do as before, as the y_fnc has been modified, with the "active_index" arguments,
    #  which decides when to count the sum from. But also adding the right theta-values to start with.
    return x_fnc(theta, l, active_index=k)


def ddy(k, m, theta, l):
    max_index = max(k, m)
    return -y_fnc(theta, l, active_index=max_index)


def f(p, theta, l):
    return (1/2.0)*((p[0]-x_fnc(theta, l))**2 + (p[1]-y_fnc(theta, l))**2)


def df(p, theta, l):
    Df = np.array([-(p[0]-x_fnc(theta, l))*dx(i, theta, l) - (p[1]-y_fnc(theta, l))*dy(i, theta, l)
                   for i in range(len(l))])
    return Df


def ddf(p, theta, l):
    n = len(l)
    DDf = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x_terms = dx(i, theta, l)*dx(j, theta, l) - (p[0] - x_fnc(theta, l))*ddx(i, j, theta, l)
            y_terms = dy(i, theta, l)*dy(j, theta, l) - (p[1] - y_fnc(theta, l))*ddy(i, j, theta, l)
            DDf[i, j] = x_terms + y_terms

    return DDf


def testing_code():
    # print(f"X-value-test:\n{x_fnc(np.array([np.pi/2.0]*3), [1, 2, 3], active_index=0)} ")
    # print(f"X-deriative:\n{dx(1, np.array([np.pi/2.0]*3), [1, 2, 3])}")

    # print("problem 1")
    # n=3
    # theta_0 = np.ones(n)*(1.0/2)
    # l=np.array([3,2,2])
    # p=(3,2)
    # theta, f_min = find_minimum(f,p,df,theta_0, ddf, line_lengths=l)
    # print("theta, f(theta)=", theta, f_min)
    # visualization.display_robot_arm(l, theta, p)

    # print("\nproblem 2")
    # n=3
    # theta_0 = np.ones(n)*(np.pi/3)
    # l=np.array([1,4,1])
    # p=(1,1)
    #
    # theta, f_min = find_minimum(f,p,df,theta_0, ddf, line_lengths=l)
    # print("theta, f(theta)=", theta, f_min)
    # visualization.display_robot_arm(l, theta, p)

    # print("\nproblem 3")
    # n = 4
    # theta_0 = np.ones(n)*(np.pi/4)
    # l=np.array([3,2,1,1])
    # p=(3,2)
    # theta, f_min = find_minimum(f,p,df,theta_0, ddf, line_lengths=l)
    # print("theta, f(theta)=", theta, f_min)
    # visualization.display_robot_arm(l, theta, p)

    # Problems with singular matrix:
    print("\nproblem 4")
    n = 4
    p = (0.0, 0.0)
    theta_0 = np.ones(n) * (np.pi / 3)
    l = np.array([3, 2, 1, 1])
    theta, f_min = find_minimum(f, p, df, theta_0, ddf, line_lengths=l)
    print("theta, f(theta)=", theta, f_min)
    visualization.display_robot_arm(l, theta, p)


if __name__ == "__main__":
    testing_code()
