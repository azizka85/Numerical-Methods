from typing import Callable, Tuple
from numpy import sign
from math import sqrt

def root_search(
    f: Callable[[float], float], 
    a: float, b: float, dx: float
) -> Tuple[float, float]:
    x1 = a
    f1 = f(x1)

    x2 = x1 + dx
    f2 = f(x2)

    while sign(f1) == sign(f2):
        if x1 >= b:
            return None, None
        
        x1 = x2
        f1 = f2

        x2 = x1 + dx
        f2 = f(x2)
    else:
        return x1, x2
    
def bisection(
    f: Callable[[float], float],
    a: float, b: float, tol: float
) -> float:
    if abs(f(a)) <= tol:
        return a
    
    if abs(f(b)) <= tol:
        return b
    
    x1 = a
    f1 = f(x1)

    x2 = b
    f2 = f(x2)

    if sign(f1) == sign(f2):
        return None
    
    while x2 - x1 > tol:
        x3 = 0.5 * (x1 + x2)
        f3 = f(x3)

        if abs(f3) <= tol:
            return x3
        
        if sign(f2) != sign(f3):
            x1 = x3
            f1 = f3
        else:
            x2 = x3
            f2 = f3

    return (x1 + x2) / 2.

def ridder(
    f: Callable[[float], float],
    a: float, b: float, 
    tol: float, max_iter: int
) -> Tuple[float, int]:
    if abs(f(a)) <= tol:
        return a, 0
    
    if abs(f(b)) <= tol:
        return b, 0
    
    x1 = a
    f1 = f(x1)

    x2 = b
    f2 = f(x2)    
    
    x = 1.
    dx = 2 * tol

    iter = 0

    while abs(dx) > tol and iter < max_iter:
        x3 = 0.5 * (x1 + x2)
        f3 = f(x3)

        s = sqrt(f3**2 - f1*f2)

        if s == 0.:
            return None, iter

        dx = (x3 - x1) * f3 / s

        if f1 < f2:
            dx = -dx

        x = x3 + dx
        fx = f(x)

        if sign(f3) == sign(fx):
            if sign(f1) != sign(fx):
                x2 = x
                f2 = fx
            else:
                x1 = x
                f1 = fx
        else:
            x1 = x3
            f1 = f3

            x2 = x            
            f2 = fx

        iter += 1

    return x, iter
        
def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    a: float, b: float, 
    tol: float, max_iter: int
) -> Tuple[float, int]:
    if abs(f(a)) <= tol:
        return a, 0
    
    if abs(f(b)) <= tol:
        return b, 0
    
    x1 = a
    f1 = f(x1)

    x2 = b
    f2 = f(x2)    

    if sign(f1) == sign(f2):
        return None, 0
    
    x = 0.5 * (x1 + x2)
    dx = 2 * tol

    iter = 0

    while abs(dx) > tol and iter < max_iter:
        fx = f(x)

        if abs(fx) <= tol:
            return x, iter
        
        if sign(f1) != sign(fx):
            x2 = x
        else:
            x1 = x

        dfx = df(x)

        if abs(dfx) <= tol:
            return None, iter
        
        dx = -fx / dfx
        x = x + dx

        if (x2 - x)*(x - x1) < 0.:
            dx = 0.5 * (x2 - x1)
            x = x1 + dx

        iter += 1

    return x, iter

