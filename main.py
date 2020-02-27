# Main file, for staging and exectuting code

# import numpy as np


# Our imports:
from newton_solver import find_minimum
from project_functions import *
import visualization


def problem_1():
    print("problem 1")
    n = 3
    theta_0 = np.ones(n) * (1.0 / 2)
    l = np.array([3, 2, 2])
    p = (3, 2)
    theta, f_min = find_minimum(f, p, df, theta_0, ddf, line_lengths=l, tol=1.0e-10)
    print("theta, f(theta)=", theta, f_min)
    visualization.display_robot_arm(l, theta, p)


def problem_2():
    print("\nproblem 2")
    n=3
    theta_0 = np.ones(n)*(np.pi/3)
    l=np.array([1,4,1])
    p=(1,1)

    theta, f_min = find_minimum(f,p,df,theta_0, ddf, line_lengths=l, tol=1.0e-10)
    print("theta, f(theta)=", theta, f_min)
    visualization.display_robot_arm(l, theta, p)


def problem_3():
    print("\nproblem 3")
    n = 4
    theta_0 = np.ones(n) * (np.pi / 4)
    l = np.array([3, 2, 1, 1])
    p = (3, 2)
    theta, f_min = find_minimum(f, p, df, theta_0, ddf, line_lengths=l, tol=1.0e-10)
    print("theta, f(theta)=", theta, f_min)
    visualization.display_robot_arm(l, theta, p)


def problem_4():
    # Problems with singular matrix when target point is origo:
    print("\nproblem 4")
    n = 4
    p = (0.0, 0.0)
    theta_0 = np.ones(n) * (np.pi / 3)
    l = np.array([3, 2, 1, 1])
    theta, f_min = find_minimum(f, p, df, theta_0, ddf, line_lengths=l, tol=1.0e-10, max_iter=100)
    print("theta, f(theta)=", theta, f_min)
    visualization.display_robot_arm(l, theta, p)


def testing_code():
    print(f"X-value-test:\n{x_fnc(np.array([np.pi/2.0]*3), [1, 2, 3], active_index=0)} ")
    print(f"X-deriative:\n{dx(1, np.array([np.pi/2.0]*3), [1, 2, 3])}")


def main():
    problem_1()
    problem_2()
    problem_3()
    problem_4()


if __name__ == "__main__":
    main()
