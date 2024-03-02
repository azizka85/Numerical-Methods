import numpy as np
from modules.slae.solver.direct.lu_decomposition import lu_3_diag_decomposition, lu_3_diag_solve

def curvatures(xd: np.ndarray[np.double], yd: np.ndarray[np.double]) -> np.ndarray[np.double]:
    n = len(xd) - 1
    
    c = np.zeros(n)
    d = np.ones(n + 1)
    e = np.zeros(n)
    k = np.zeros(n + 1)

    c[0:n-1] = xd[0:n-1] - xd[1:n]
    d[1:n] = 2.*(xd[0:n-1] - xd[2:n+1])
    e[1:n] = xd[1:n] - xd[2:n+1]

    k[1:n] = 6.*(yd[0:n-1]-yd[1:n])/(xd[0:n-1] - xd[1:n]) \
            - 6.*(yd[1:n] - yd[2:n+1])/(xd[1:n]-xd[2:n+1])
    
    ct, dt, et = lu_3_diag_decomposition(c, d, e)
    k = lu_3_diag_solve(ct, dt, et, k)

    return k

def find_segment(xd: np.ndarray[np.double], x: np.double) -> np.double:
    il = 0
    ir = len(xd) - 1

    while True:
        if (ir - il) <= 1: return il

        i = (ir + il) // 2

        if x < xd[i]: ir = i
        else: il = i

def polynom(xd: np.ndarray[np.double], yd: np.ndarray[np.double], k: np.ndarray[np.double], x: np.double) -> np.double:
    i = find_segment(xd, x)
    h = xd[i] - xd[i+1]

    return ((x - xd[i+1])**3/h - (x - xd[i+1])*h)*k[i]/6. \
            - ((x - xd[i])**3/h - (x - xd[i])*h)*k[i+1]/6. \
            + (yd[i]*(x - xd[i+1]) - yd[i+1]*(x - xd[i]))/h
