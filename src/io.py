import numpy as np


def read_csv(filepath, delimiter=",", skip_header=True):
    data = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    if skip_header:
        lines = lines[1:]

    for line in lines:
        row = []
        for val in line.strip().split(delimiter):
            val = val.strip().replace('"', '')

            if val == "" or val.lower() == "nan":
                row.append(np.nan)
            else:
                row.append(float(val))

        data.append(row)

    return np.array(data)


def fill_missing_mean(X):
    X = np.asarray(X, dtype=float)

    col_means = np.nanmean(X, axis=0)

    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    return X