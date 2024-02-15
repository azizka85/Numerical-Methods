from typing import List
from modules.matrix.utils import matrix_sym_triplet_mult_vector
import numpy as np
import math

def conjugate_gradient(
    a: np.ndarray[np.double, np.double], 
    b: np.ndarray[np.double],
    tol: np.double, max_iter: np.uint32    
) -> tuple[np.ndarray[np.double], np.uint32]:
    n = len(b)
    x = np.zeros(n)
    r = b - np.dot(a, x)
    s = r.copy()

    iter = 0

    while math.sqrt(np.dot(r, r)) > tol and iter < max_iter:
        u = np.dot(a, s)
        alpha = np.dot(s, r) / np.dot(s, u)
        x = x + alpha * s
        r = b - np.dot(a, x)
        beta = -np.dot(r, u) / np.dot(s, u)
        s = r + beta * s

        iter += 1

    return x, iter

def conjugate_gradient_sym_triplet(
    t: List[tuple[np.uint32, np.uint32, np.double]], 
    b: np.ndarray[np.double],
    tol: np.double, max_iter: np.uint32    
) -> tuple[np.ndarray[np.double], np.uint32]:
    n = len(b)
    x = np.zeros(n)
    r = b - matrix_sym_triplet_mult_vector(t, x)
    s = r.copy()

    iter = 0

    while math.sqrt(np.dot(r, r)) > tol and iter < max_iter:
        u = matrix_sym_triplet_mult_vector(t, s)
        alpha = np.dot(s, r) / np.dot(s, u)
        x = x + alpha * s
        r = b - matrix_sym_triplet_mult_vector(t, x)
        beta = -np.dot(r, u) / np.dot(s, u)
        s = r + beta * s

        iter += 1

    return x, iter
