# conjugate gradient algorithm for solving linear systems.
import numpy as np
from sklearn.datasets import make_spd_matrix

def LinearCG(A, b, x0, tol=1e-5):
    xk = x0
    rk = np.dot(A, xk) - b
    pk = -rk
    rk_norm = np.linalg.norm(rk)
    
    num_iter = 0
    curve_x = [xk]
    while rk_norm > tol:
        apk = np.dot(A, pk)
        rkrk = np.dot(rk, rk)
        
        alpha = rkrk / np.dot(pk, apk)
        xk = xk + alpha * pk
        rk = rk + alpha * apk
        beta = np.dot(rk, rk) / rkrk
        pk = -rk + beta * pk
        
        num_iter += 1
        curve_x.append(xk)
        rk_norm = np.linalg.norm(rk)
        print('Iteration: {} \t x = {} \t residual = {:.4f}'.
              format(num_iter, xk, rk_norm))
    
    print('\nSolution: \t x = {}'.format(xk))
        
    return np.array(curve_x)


# Problem 1

# A is a 2 × 2 symmetric positive-definite matrix and b is a 2 × 1 vector.


A= [[ 2.54086605,-0.01128187], [-0.01128187, 0.52868286]] 
b = [1.38639293, 0.37191672]

x0 = np.array([ЛЮБОЕ_ЧИСЛО_ПО_ОСИ_Х, ЛЮБОЕ_ЧИСЛО_ПО_ОСИ_Y]) # НАЧАЛЬНОЕ ПРИБЛЕЖЕНИЕ по весам

Запускайте алгоритм, всё готово!


# РАНДОМНАЯ МАТРИЦА -ЕСЛИ НАДО поэксперементировать
#np.random.seed(0)
#A = make_spd_matrix(2, random_state=0)
#x_star = np.random.random(2) веса которые мы не знаем
#b = np.dot(A, x_star)
##############

