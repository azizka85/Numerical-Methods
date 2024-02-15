import numpy as np

def vandermode(v: np.ndarray[np.double]) -> np.ndarray[np.double]:
    n = len(v)
    a = np.zeros((n, n))

    for j in range(n):
        a[:, j] = v ** (n - j - 1)

    return a