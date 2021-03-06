import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from robust_newton_solver import robust_minimizer
from project_functions import *
import visualization


def problem_1():
    print("problem 1")
    n = 3
    theta_1 = np.ones(n) * (1.0 / 2)
    l1 = np.array([3, 2, 2])
    p1 = (3, 2)

    theta, f_min = robust_minimizer(p1, f, df, ddf, theta_1, line_lengths=l1)
    print("theta: {}, f(theta) = {:.3e}".format(theta, f_min))
    visualization.display_robot_arm(l1, theta, p1)


def problem_2():
    print("\nproblem 2")
    n = 3
    theta_2 = np.array([np.pi/3, -np.pi/3, np.pi/3.0])
    l2 = np.array([1, 4, 1])
    p2 = (1, 1)

    theta, f_min = robust_minimizer(p2, f, df, ddf, theta_2, line_lengths=l2)
    print("theta: {}, f(theta) = {:.3e}".format(theta, f_min))
    visualization.display_robot_arm(l2, theta, p2)


def problem_3():
    print("\nproblem 3")
    n = 4
    theta_3 = np.ones(n) * (0.0/4)
    l3 = np.array([3, 2, 1, 1])
    p3 = (3, 2)

    theta, f_min = robust_minimizer(p3, f, df, ddf, theta_3, line_lengths=l3)
    print("theta: {}, f(theta) = {:.3e}".format(theta, f_min))
    visualization.display_robot_arm(l3, theta, p3)


def problem_4():
    # Problems with singular matrix when target point is origo:
    print("\nproblem 4")
    n = 4
    p4 = (0.0, 0.0)
    theta_4 = np.ones(n) * (np.pi / 3)
    l4 = np.array([3, 2, 1, 1])

    theta, f_min = robust_minimizer(p4, f, df, ddf, theta_4, line_lengths=l4)
    print("theta: {}, f(theta) = {:.3e}".format(theta, f_min))
    visualization.display_robot_arm(l4, theta, p4)


def problem_5():
    # Larger problem with many arms:
    np.random.seed(2)

    n = 20
    p5 = (-10.0, 10.0)

    theta_5 = np.random.randn(n)
    l5 = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], n, replace=True)
    print(f"Outer anulus radius: {np.linalg.norm(l5, 1)}\nPoint distance: {np.linalg.norm(p5, 2)}")

    theta, f_min, delta_xs, delta_fs = robust_minimizer(p5, f, df, ddf, theta_5, l5, convergence_data=True)
    norm_delta_fs = np.array([np.linalg.norm(delta_f) for delta_f in delta_fs])
    decrease_slope = stats.linregress(np.arange(len(delta_fs)), np.log10(norm_delta_fs))[0]

    function_decreases = norm_delta_fs[1:] / norm_delta_fs[0:-1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # ax.plot(np.arange(len(delta_xs)), [np.log(np.linalg.norm(delta_x)) for delta_x in delta_xs])
    label_text = f"log(f_k), with slope {decrease_slope:.3f}\nMean f_(k+1)/f_k = {np.mean(function_decreases):.3f}"
    ax.plot(np.arange(len(delta_fs)), np.log10(norm_delta_fs), label=label_text)
    fig.suptitle(r"Convergence of $f(\theta)$", fontsize=15)
    ax.set_ylabel(r"$log(f(\vec{\theta}_k))$", fontsize=12)
    ax.set_xlabel("Iteration Number, k", fontsize=12)
    ax.legend(loc="best")
    ax.grid(linestyle="--")
    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.96])
    plt.show()
    # print("theta: {}, f(theta) = {:.3e}".format(theta, f_min))
    # visualization.display_robot_arm(l5, theta, p5)


def convergence_plot():
    pass


def main():
    # problem_1()
    # problem_2()
    # problem_3()
    # problem_4()
    problem_5()


if __name__ == "__main__":
    main()
