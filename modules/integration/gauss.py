from math import cos, pi
from typing import Callable, Tuple
import numpy as np

def legendre(t: float, m: int) -> Tuple[float, float]:
    p0 = 1.; p1 = t

    for k in range(1, m):
        p = ((2.*k + 1.)*t*p1 - k*p0)/(1. + k)
        p0 = p1; p1 = p

    dp = m*(p0 - t*p1)/(1. - t**2)

    return p, dp

def nodes(m: int, tol = 10e-9, max_iter = 30):
    A = np.zeros(m)
    x = np.zeros(m)

    nRoots = int((m + 1)/2)

    for i in range(nRoots):
        t = cos(pi*(i + 0.75)/(m + 0.5))

        dt = 2*tol; iter = 0

        while iter < max_iter and abs(dt) > tol:
            p, dp = legendre(t, m)

            dt = -p/dp; t = t + dt
            iter += 1

        x[i] = t; x[m - i - 1] = -t
        A[i] = 2./(1. - t**2)/dp/dp
        A[m - i - 1] = A[i]

    return x, A

def integral(
    f: Callable[[float], float], a: float, b: float, k: int,
    tol = 10e-9, max_iter = 30
) -> float:
    c1 = (b + a)/2.
    c2 = (b - a)/2.

    x, A = nodes(k, tol, max_iter)

    sum = 0.

    for i in range(len(x)):
        sum = sum + A[i]*f(c1 + c2*x[i])

    return c2*sum
        