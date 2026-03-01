# stepik: https://stepik.org/a/260000


# Conjugate Gradient & Sparse CG Solver Course

[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Website](https://img.shields.io/badge/website-live-blue.svg)](https://senatorovai.github.io/conjugate-gradient-sparse-cg-solver-course/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18818738.svg)](https://doi.org/10.5281/zenodo.18820931)


> 🚀 Professional implementation and mathematical explanation of **Conjugate Gradient (CG)** and **Sparse Conjugate Gradient** methods for large-scale linear systems.

---

## 🔥 Project Overview

This repository provides a complete course-style implementation of:

- Conjugate Gradient (CG) algorithm
- Preconditioned Conjugate Gradient
- Sparse Conjugate Gradient
- Large-scale linear system solving
- Numerical stability analysis
- Optimization perspective of CG

---

## Keywords

```

conjugate gradient
conjugate gradient method
sparse conjugate gradient
pcg solver
cg solver python
large scale linear systems
numerical linear algebra
iterative solver
preconditioned conjugate gradient
optimization solver
python cg implementation

```

---

## 📚 Mathematical Background

### Linear System Problem

We solve:

$$
Ax = b
$$

Where:

- $$A$$ — symmetric positive definite matrix
- $$x$$ — unknown vector
- $$b$$ — right-hand side

---

## 🔵 Conjugate Gradient Method

CG minimizes quadratic function:

$$
f(x) = \frac{1}{2}x^T A x - b^T x
$$

Update rule:

$$
x_{k+1} = x_k + \alpha_k p_k
$$

Where:

- $$p_k$$ — conjugate direction
- $$\alpha_k$$ — optimal step size

Step size:

$$
\alpha_k =
\frac{r_k^T r_k}
{p_k^T A p_k}
$$

---

### Residual Update

$$
r_k = b - Ax_k
$$

Direction update:

$$
p_{k+1} = r_{k+1} + \beta_k p_k
$$

Where:

$$
\beta_k =
\frac{r_{k+1}^T r_{k+1}}
{r_k^T r_k}
$$

---

## ⚡ Sparse Conjugate Gradient

For sparse matrices:

- Store matrix in CSR/CSC format
- Avoid dense multiplication
- Reduce memory complexity

Advantages:

✅ Memory efficient  
✅ Faster computation  
✅ Scalable to large systems  

---

## 🧠 Why This Project Is Important

CG is used in:

- Finite element methods
- Physics simulations
- Machine learning
- PDE solvers
- Large sparse systems
- Scientific computing

It is one of the most important iterative solvers.

---

## 🏗 Project Structure

```

conjugate-gradient-sparse-cg-solver-course/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
│
├── src/
│   ├── cg_solver.py
│   ├── sparse_cg.py
│   ├── preconditioner.py
│
├── examples/
│   └── demo.py
│
├── docs/
│   ├── theory.md
│   ├── convergence.md
│
├── images/
│   └── convergence_plot.png
│
└── index.html

````

Clean structure improves:

✔ Search ranking  
✔ Professional appearance  
✔ Research credibility  

---

## 🐍 Example — Basic Conjugate Gradient Implementation

```python
import numpy as np

def conjugate_gradient(A, b, x0=None, tol=1e-8, max_iter=1000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0

    r = b - A @ x
    p = r.copy()

    for _ in range(max_iter):
        Ap = A @ p
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p

        r_new = r - alpha * Ap

        if np.linalg.norm(r_new) < tol:
            break

        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new

    return x
````

---

## 🚀 Installation

```bash id="install-cg"
pip install -r requirements.txt
```

Run example:

```bash id="run-cg"
python examples/demo.py
```

---

## 📊 Visualization (Highly Recommended)

Add:

* Residual norm vs iteration
* Convergence curve
* Sparse matrix structure plot

Example:

```python id="plot-cg"
import matplotlib.pyplot as plt

plt.plot(residual_history)
plt.xlabel("Iteration")
plt.ylabel("Residual Norm")
plt.title("CG Convergence")
plt.show()
```



