import numpy as np
import math
import matplotlib.pyplot as plt
from modules.slae.solver.direct.gauss_elimination import gauss_elimination

def poly_fit(xd: np.ndarray[np.double], yd: np.ndarray[np.double], m: int) -> np.ndarray[np.double]:
    a = np.zeros((m + 1, m + 1))
    b = np.zeros(m + 1)
    s = np.zeros(2*m + 1)

    for i in range(len(xd)):
        t = yd[i]

        for j in range(m + 1):
            b[j] = b[j] + t
            t = t * xd[i]

        t = 1.

        for j in range(2*m + 1):
            s[j] = s[j] + t
            t = t * xd[i]

    for i in range(m + 1):
        for j in range(m + 1):
            a[i, j] = s[i + j]

    return gauss_elimination(a, b)

def polynom(c: np.ndarray[np.double], x: np.double) -> np.double:
    m = len(c) - 1
    p = c[m]

    for j in range(m):
        p = p*x + c[m - j - 1]

    return p

def std_dev(c: np.ndarray[np.double], xd: np.ndarray[np.double], yd: np.ndarray[np.double]) -> np.double:
    n = len(xd) - 1
    m = len(c) - 1

    sigma = 0.

    for i in range(n + 1):
        p = polynom(c, xd[i])
        sigma = sigma + (yd[i] - p)**2

    sigma = math.sqrt(sigma / (n - m))

    return sigma

def plot_poly(xd: np.ndarray[np.double], yd: np.ndarray[np.double], c: np.ndarray[np.double]):
    m = len(c)
    
    x1 = np.min(xd)
    x2 = np.max(xd)

    dx = (x2 - x1) / 20.

    x = np.arange(x1, x2 + dx / 10., dx)
    y = np.zeros(len(x)) * 1.

    for i in range(m):
        y = y + c[i] * x ** i

    plt.plot(xd, yd, 'x', x, y, '-')        
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend(['Data', f'Polynom of {m - 1} degree'], loc='lower right')
    plt.show()
