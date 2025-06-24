from typing import Callable


def integral(
        f: Callable[[float], float], a: float, b: float, k: int) -> float:
    if k == 1: 
        return (f(a) + f(b))*(b - a)/2
    else:
        Iold = integral(f, a, b, k - 1)

        n = 2**(k - 2)
        h = (b - a)/n
        x = a + h/2

        sum = 0.

        for _ in range(n):
            sum = sum + f(x)
            x = x + h

        return (Iold + h*sum)/2
