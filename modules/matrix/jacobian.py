import numpy as np
from typing import Callable, Tuple

def jacobian(
    f: Callable[[np.ndarray[np.double]], np.ndarray[np.double]], 
    x: np.ndarray[np.double], h: np.double
) -> Tuple[np.ndarray[np.double, np.double], np.ndarray[np.double]]:
    n = len(x)    
    jac = np.zeros((n, n))
    f0 = f(x)

    for i in range(n):
        t = x[i]
        x[i] = t + h
        f1 = f(x)
        x[i] = t

        jac[:, i] = (f1 - f0) / h

    return jac, f0
