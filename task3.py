A = np.array([[4, 1], [1, 2]], dtype=float)
b = np.array([1, 1], dtype=float)

def f3(x):
    return 0.5 * x @ A @ x - b @ x
  
x0=[0, 0]

res = minimize()
print("f(x*):", res.fun)
