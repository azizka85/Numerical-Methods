# %% [markdown]
# # Итерационные методы решения СЛАУ

# %%
import sys

sys.path.append('..')

# %% [markdown]
# ### 1. Метод Гаусса-Зейделя

# %%
import numpy as np
from modules.slae.solver.iterative.gauss_seidel import gauss_seidel

a = np.array([
    [4., -1., 1.],
    [-1., 4., -2.],
    [1., -2., 4.]
])

b = np.array([12., -1., 5.])

x, iter = gauss_seidel(a, b, 10**-9, 1000)

print(f'iterations: {iter}')

x

# %% [markdown]
# ##### Проверим найденное решение
# 
# Если $x$ решение $A * x = b$, то $r = A * x - b = 0$

# %%
r = np.dot(a, x) - b

r

# %% [markdown]
# ### 2. Метод Гаусса-Зейделя c релаксацией

# %%
from modules.slae.solver.iterative.gauss_seidel import gauss_seidel_relax

a = np.array([
    [4., -1., 1.],
    [-1., 4., -2.],
    [1., -2., 4.]
])

b = np.array([12., -1., 5.])

x, iter = gauss_seidel_relax(a, b, 10**-9, 1000)

print(f'iterations: {iter}')

x

# %%
r = np.dot(a, x) - b

r

# %% [markdown]
# ### 3. Метод сопряженных градиентов

# %%
from modules.slae.solver.iterative.conjugate_gradient import conjugate_gradient

a = np.array([
    [4., -1., 1.],
    [-1., 4., -2.],
    [1., -2., 4.]
])

b = np.array([12., -1., 5.])

x, iter = conjugate_gradient(a, b, 10**-9, 1000)

print(f'iterations: {iter}')

x

# %%
r = np.dot(a, x) - b

r

# %% [markdown]
# ### 4. Метод сопряженных градиентов для разреженной матрицы

# %%
from modules.slae.solver.iterative.conjugate_gradient import conjugate_gradient_sym_triplet

t = [
    [0, 0, 3.],
    [0, 2, -1.],
    [1, 1, 4.],
    [1, 2, -2.],
    [2, 2, 5.]
]

b = np.array([4., 10., -10.])

x, iter = conjugate_gradient_sym_triplet(t, b, 10**-9, 1000)

print(f'iterations: {iter}')

x

# %%
from modules.matrix.utils import matrix_sym_triplet_mult_vector

r = matrix_sym_triplet_mult_vector(t, x) - b

r


