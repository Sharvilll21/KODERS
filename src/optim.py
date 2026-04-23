import numpy as np


def gradient(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_forward = x.copy()
        x_backward = x.copy()

        x_forward[i] += eps
        x_backward[i] -= eps

        grad[i] = (f(x_forward) - f(x_backward)) / (2 * eps)

    return grad


def jacobian(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    y = f(x)

    J = np.zeros((len(y), len(x)))

    for i in range(len(x)):
        x_forward = x.copy()
        x_backward = x.copy()

        x_forward[i] += eps
        x_backward[i] -= eps

        J[:, i] = (f(x_forward) - f(x_backward)) / (2 * eps)

    return J