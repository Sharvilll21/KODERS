import numpy as np


def gradient(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_f = x.copy()
        x_b = x.copy()

        x_f[i] += eps
        x_b[i] -= eps

        grad[i] = (f(x_f) - f(x_b)) / (2 * eps)

    return grad


def jacobian(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    y = f(x)

    J = np.zeros((len(y), len(x)))

    for i in range(len(x)):
        x_f = x.copy()
        x_b = x.copy()

        x_f[i] += eps
        x_b[i] -= eps

        J[:, i] = (f(x_f) - f(x_b)) / (2 * eps)

    return J