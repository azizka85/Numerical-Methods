import numpy as np

def gauss_elimination(a: np.ndarray[np.double, np.double], b: np.ndarray[np.double]) -> np.ndarray[np.double]:
    n = len(b)

    a = a.copy()
    b = b.copy()

    for k in range(0, n - 1):
        for i in range(k + 1, n):
            l = a[i, k] / a[k, k]
            a[i, k + 1:n] = a[i, k + 1:n] - l * a[k, k + 1:n]
            b[i] = b[i] - l * b[k]

    for k in range(n - 1, -1, -1):
        b[k] = (b[k] - np.dot(a[k, k + 1:n], b[k + 1:n])) / a[k, k]

    return b
