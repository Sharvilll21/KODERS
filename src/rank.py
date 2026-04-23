import numpy as np


def rank_data(x):
    x = np.asarray(x)

    # get sorted order
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)

    sorted_x = x[order]
    n = len(x)

    i = 0
    while i < n:
        j = i

        # find range of equal values (ties)
        while j + 1 < n and sorted_x[j] == sorted_x[j + 1]:
            j += 1

        # assign average rank for ties
        avg_rank = (i + j) / 2 + 1
        ranks[order[i:j+1]] = avg_rank

        i = j + 1

    return ranks


def percentile(x):
    x = np.asarray(x)
    ranks = rank_data(x)
    return 100 * (ranks - 1) / (len(x) - 1)