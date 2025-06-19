import numpy as np

def lu_decomposition(a: np.ndarray[np.double, np.double]) -> np.ndarray[np.double, np.double]:
    n = len(a)

    a = a.copy()

    for k in range(0, n - 1):
        for i in range(k + 1, n):
            l = a[i, k] / a[k, k]
            a[i, k + 1:n] = a[i, k + 1:n] - l * a[k, k + 1:n]
            a[i, k] = l

    return a

def lu_solve(a: np.ndarray[np.double, np.double], b: np.ndarray[np.double]) -> np.ndarray[np.double]:
    n = len(a)

    b = b.copy()

    for k in range(1, n):
        b[k] = b[k] - np.dot(a[k, 0:k], b[0:k])

    b[n - 1] = b[n - 1] / a[n - 1, n - 1]

    for k in range(n - 2, -1, -1):
        b[k] = (b[k] - np.dot(a[k, k + 1:n], b[k + 1:n])) / a[k, k]

    return b

def lu_3_diag_decomposition(c: np.ndarray[np.double], d: np.ndarray[np.double], e: np.ndarray[np.double]):
    n = len(d)

    d = d.copy()
    c = c.copy()

    for k in range(1, n):
        l = c[k - 1] / d[k - 1]
        d[k] = d[k] - l * e[k - 1]
        c[k - 1] = l

    return c, d, e

def lu_3_diag_solve(c: np.ndarray[np.double], d: np.ndarray[np.double], e: np.ndarray[np.double], b: np.ndarray[np.double]):
    n = len(d)

    b = b.copy()

    for k in range(1, n):
        b[k] = b[k] - c[k - 1] * b[k - 1]

    b[n - 1] = b[n - 1] / d[n - 1]

    for k in range(n - 2, -1, -1):
        b[k] = (b[k] - e[k] * b[k + 1]) / d[k]

    return b
