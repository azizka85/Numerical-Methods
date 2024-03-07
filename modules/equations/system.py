import numpy as np
from typing import Callable, Tuple
from modules.matrix.jacobian import jacobian
from modules.slae.solver.direct.gauss_elimination import gauss_elimination

def newton_raphson_system(
    f: Callable[[np.ndarray[np.double]], np.ndarray[np.double]], 
    x: np.ndarray[np.double],
    h: np.double, tol: np.double, max_iter: int
) -> Tuple[np.ndarray[np.double], int]:
    dx = 2 * tol * np.ones(len(x))
    iter = 0

    while max(abs(dx)) > tol and iter < max_iter:
        jac, f0 = jacobian(f, x, h)
        dx = gauss_elimination(jac, -f0)

        x = x + dx
        iter += 1
        
    return x, iter