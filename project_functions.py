import numpy as np


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
