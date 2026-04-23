import numpy as np


def mean(x):
    x = np.asarray(x)
    return np.sum(x) / x.size


def variance(x):
    x = np.asarray(x)
    m = mean(x)
    return np.sum((x - m) ** 2) / x.size


def standard_deviation(x):
    return np.sqrt(variance(x))


def histogram(x, bins=10):
    x = np.asarray(x)
    counts, edges = np.histogram(x, bins=bins)
    return counts, edges


def quantiles(x, q):
    x = np.asarray(x)
    return np.quantile(x, q)