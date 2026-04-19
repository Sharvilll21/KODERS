import numpy as np

class StandardScaler:
    def _init_(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X = np.array(X)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        # avoid division issues if a column has zero variance
        self.std[self.std == 0] = 1

    def transform(self, X):
        X = np.array(X)
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)