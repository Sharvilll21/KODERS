import numpy as np


def top_k(x, k):
    x = np.asarray(x)

    if k <= 0 or k > x.size:
        raise ValueError("k must be between 1 and the size of the array")

    # get indices of k largest elements (not fully sorted)
    indices = np.argpartition(x, -k)[-k:]

    # sort those indices by value (descending)
    sorted_indices = indices[np.argsort(x[indices])[::-1]]

    return x[sorted_indices]


def binary_search(x, target):
    x = np.asarray(x)

    left, right = 0, x.size - 1

    while left <= right:
        mid = (left + right) // 2

        if x[mid] == target:
            return mid
        elif x[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1