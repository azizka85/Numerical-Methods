import numpy as np
import math

def choleski_decomposition(a: np.ndarray[np.double, np.double]) -> np.ndarray[np.double, np.double]:
    n = len(a)

    a = a.copy()

    for k in range(n):
        a[k, k] = math.sqrt(a[k, k] - np.dot(a[k, 0:k], a[k, 0:k]))

        for i in range(k + 1, n):
            a[i, k] = (a[i, k] - np.dot(a[i, 0:k], a[k, 0:k])) / a[k, k]

    for k in range(1, n):
        a[0:k, k] = 0.

    return a

def choleski_solve(a: np.ndarray[np.double, np.double], b: np.ndarray[np.double]) -> np.ndarray[np.double]:
    n = len(b)

    b = b.copy()

    for k in range(n):
        b[k] = (b[k] - np.dot(a[k, 0:k], b[0:k])) / a[k, k]

    for k in range(n - 1, -1, -1):
        b[k] = (b[k] - np.dot(a[k + 1:n, k], b[k + 1:n])) / a[k, k]

    return b
