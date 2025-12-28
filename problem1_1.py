import numpy as np
from sklearn.datasets import make_spd_matrix

np.random.seed(0)
A = make_spd_matrix(2, random_state=0)
x_star = np.random.random(2) # истинные веса
b = np.dot(A, x_star)
print(np.allclose(веса_которые_нашёл_алгоритм, x_star))
