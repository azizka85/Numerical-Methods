import numpy as np
import math

def gauss_seidel(
    a: np.ndarray[np.double, np.double], 
    b: np.ndarray[np.double], 
    tol: np.double, max_iter: np.uint32
) -> tuple[np.ndarray[np.double], np.uint32]:
    n = len(b)
    x = np.zeros(n)    
    dx = 2 * tol

    iter = 0

    while dx > tol and iter < max_iter:
        x_old = x.copy()

        for i in range(n):
            x[i] = (b[i] - np.dot(a[i, :], x) + a[i, i] * x[i]) / a[i, i]

        dx = math.sqrt(np.dot(x - x_old, x - x_old))        
        iter += 1

    return x, iter

def gauss_seidel_relax(
    a: np.ndarray[np.double, np.double], 
    b: np.ndarray[np.double], 
    tol: np.double, max_iter: np.uint32
) -> tuple[np.ndarray[np.double], np.uint32]:
    n = len(b)
    x = np.zeros(n)    
    dx = 2 * tol

    omega = 1.
    k = 10
    p = 1

    iter = 0

    while dx > tol and iter < max_iter:
        x_old = x.copy()

        for i in range(n):
            x[i] = omega * (b[i] - np.dot(a[i, :], x) + a[i, i] * x[i]) / a[i, i] + (1 - omega) * x[i]

        dx = math.sqrt(np.dot(x - x_old, x - x_old))        
        iter += 1

        if iter % k == 0:
            dx1 = dx

        if iter % (k + p) == 0:
            dx2 = dx
            omega = 2./(1. + math.sqrt(1. - (dx2 / dx1)**(1./p)))

    return x, iter
