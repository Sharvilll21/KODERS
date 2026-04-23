import numpy as np


def top_k(x, k):
    x = np.asarray(x)

    if not 1 <= k <= x.size:
        raise ValueError("k must be between 1 and array size")

    indices = np.argpartition(x, -k)[-k:]
    sorted_indices = indices[np.argsort(x[indices])[::-1]]

    return x[sorted_indices]


def binary_search(x, target):
    x = np.asarray(x)

    left, right = 0, len(x) - 1

    while left <= right:
        mid = (left + right) // 2

        if x[mid] == target:
            return mid
        if x[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1