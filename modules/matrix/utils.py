import numpy as np

def matrix_3_diag_mult_vector(
    c: np.ndarray[np.double], d: np.ndarray[np.double], e: np.ndarray[np.double], 
    x: np.ndarray[np.double]
) -> np.ndarray[np.double]:
    n = len(d)
    y = np.array([0] * n)

    for i in range(n):
        ce = c[i - 1] if i > 0 else 0
        xc = x[i - 1] if i > 0 else 0

        de = d[i]
        xd = x[i]

        ee = e[i] if i < n - 1 else 0
        xe = x[i + 1] if i < n - 1 else 0

        y[i] = ce * xc + de * xd + ee * xe

    return y