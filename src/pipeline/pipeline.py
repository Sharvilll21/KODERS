class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y):
        data = X

        # run all preprocessing steps first
        for _, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                data = step.fit_transform(data)
            else:
                step.fit(data)
                data = step.transform(data)

        model = self.steps[-1][1]
        model.fit(data, y)

    def predict(self, X):
        data = X

        for _, step in self.steps[:-1]:
            data = step.transform(data)

        model = self.steps[-1][1]
        return model.predict(data)