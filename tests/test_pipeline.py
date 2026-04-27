import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.pipeline import Pipeline
from src.preprocessing.scaler import StandardScaler


class DummyTransformer:
    """Simple transformer that subtracts column mean"""
    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class DummyModel:
    """Model that always predicts 1 - for pipeline structure testing only"""
    def fit(self, X, y):
        self.trained_ = True

    def predict(self, X):
        return np.ones(X.shape[0])


class BrokenTransformer:
    """Transformer deliberately missing transform() method"""
    def fit(self, X, y=None):
        pass


class BrokenModel:
    """Model deliberately missing predict() method"""
    def fit(self, X, y):
        pass


class TestPipeline:

    def test_basic_fit_predict(self):
        """Pipeline should fit and predict without errors"""
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])
        y = np.array([0, 1, 0])
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  DummyModel()),
        ])
        pipe.fit(X, y)
        predictions = pipe.predict(X)
        assert len(predictions) == 3

    def test_transformer_without_transform_raises(self):
        """Transformer missing transform() should raise TypeError"""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 0])
        pipe = Pipeline([
            ('broken', BrokenTransformer()),
            ('model',  DummyModel()),
        ])
        with pytest.raises(TypeError):
            pipe.fit(X, y)

    def test_model_without_predict_raises(self):
        """Final step missing predict() should raise TypeError"""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 0])
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  BrokenModel()),
        ])
        with pytest.raises(TypeError):
            pipe.fit(X, y)

    def test_predict_output_length(self):
        """Predict should return same number of rows as input"""
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0],
                      [7.0, 8.0]])
        y = np.array([0, 1, 0, 1])
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  DummyModel()),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(X)

    def test_multiple_transformers(self):
        """Pipeline with multiple transformers should chain correctly"""
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])
        y = np.array([0, 1, 0])
        pipe = Pipeline([
            ('t1',    DummyTransformer()),
            ('scaler', StandardScaler()),
            ('model', DummyModel()),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == 3

    def test_single_step_pipeline(self):
        """Pipeline with just a model and no transformers should work"""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 0])
        pipe = Pipeline([
            ('model', DummyModel()),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == 3