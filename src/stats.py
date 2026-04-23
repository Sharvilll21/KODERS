import numpy as np


def mean(x):
    x = np.asarray(x, dtype=float)
    return np.mean(x)


def variance(x):
    x = np.asarray(x, dtype=float)
    return np.var(x)


def standard_deviation(x):
    x = np.asarray(x, dtype=float)
    return np.std(x)


def histogram(x, bins=10):
    x = np.asarray(x, dtype=float)
    return np.histogram(x, bins=bins)


def quantiles(x, q):
    x = np.asarray(x, dtype=float)
    return np.quantile(x, q)