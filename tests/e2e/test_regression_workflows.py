import numpy as np
import pytest

from src.metrics.metrics import mse
from src.models.linear_regression import LinearRegression
from src.pipeline.pipeline import Pipeline
from src.preprocessing.scaler import MinMaxScaler, SimpleImputer, StandardScaler


class TestRegressionWorkflowE2E:
    def test_standard_scaled_linear_regression_pipeline_learns_known_relationship(self):
        """Full workflow: scale features, fit a model, predict, and evaluate error."""
        X = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ])
        y = 4.0 + 2.0 * X[:, 0] - 0.5 * X[:, 1]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert predictions.shape == y.shape
        assert np.allclose(predictions, y, atol=1e-8)
        assert mse(y, predictions) < 1e-20

    def test_impute_then_minmax_scale_then_regress_handles_missing_values(self):
        """An end-to-end workflow should stay finite even when raw data has NaNs."""
        X = np.array([
            [1.0, np.nan],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
        ])
        y = np.array([3.0, 6.0, 9.0, 12.0])

        pipeline = Pipeline([
            ("imputer", SimpleImputer(fill_value=2.0)),
            ("scaler", MinMaxScaler()),
            ("model", LinearRegression()),
        ])

        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert np.all(np.isfinite(predictions))
        assert np.allclose(predictions, y, atol=1e-8)

    def test_pipeline_uses_training_scaler_statistics_for_unseen_predictions(self):
        """Prediction data must be transformed with fitted training statistics only."""
        X_train = np.array([[0.0], [5.0], [10.0], [15.0]])
        y_train = 10.0 + 3.0 * X_train.ravel()
        X_new = np.array([[20.0], [25.0]])

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_new)

        assert np.allclose(predictions, [70.0, 85.0], atol=1e-8)

    def test_pipeline_rejects_transformer_without_transform_method(self):
        class NotATransformer:
            def fit(self, X):
                return self

        pipeline = Pipeline([
            ("broken_step", NotATransformer()),
            ("model", LinearRegression()),
        ])

        with pytest.raises(TypeError, match="broken_step must implement transform"):
            pipeline.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))

    def test_pipeline_rejects_final_step_without_predict_method(self):
        class NotAModel:
            def fit(self, X, y):
                return self

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("not_model", NotAModel()),
        ])

        with pytest.raises(TypeError, match="not_model must implement predict"):
            pipeline.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
