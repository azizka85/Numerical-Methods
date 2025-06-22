import numpy as np

def polynom(xd: np.ndarray[np.double], yd: np.ndarray[np.double], x: np.double) -> np.double:
    m = len(xd)
    y = yd.copy()

    for k in range(1, m):
        y[0:m-k] = ((x - xd[k:m])*y[0:m-k] + (xd[0:m-k] - x)*y[1:m-k+1]) / (xd[0:m-k] - xd[k:m])

    return y[0]