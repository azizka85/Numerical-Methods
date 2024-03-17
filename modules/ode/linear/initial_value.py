import numpy as np
from typing import Callable, Tuple

def euler(
    f: Callable[[float, np.ndarray[float]], np.ndarray[float]],
    a: float, b: float, ya: np.ndarray[float], n: int
) -> Tuple[np.ndarray[float], np.ndarray[float, float]]:
    m = len(ya)
    
    x = np.linspace(a, b, n)
    y = np.zeros((n, m))

    y[0] = ya
    h = x[1] - x[0]

    for i in range(1, n):
        y[i] = y[i - 1] + h*f(x[i-1], y[i-1])

    return x, y

def modified_euler(
    f: Callable[[float, np.ndarray[float]], np.ndarray[float]],
    a: float, b: float, ya: np.ndarray[float], n: int
) -> Tuple[np.ndarray[float], np.ndarray[float, float]]:
    m = len(ya)
    
    x = np.linspace(a, b, n)
    y = np.zeros((n, m))

    y[0] = ya
    h = x[1] - x[0]

    for i in range(1, n):
        y[i] = y[i - 1] + h*f(x[i-1] + h/2, y[i-1] + h*f(x[i-1], y[i-1])/2)

    return x, y
