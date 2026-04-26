import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.scaler import StandardScaler


class TestStandardScaler:

    def test_basic_scaling(self):
        """After scaling, mean should be ~0 and std should be ~1"""
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-6)
        assert np.allclose(X_scaled.std(axis=0),  1.0, atol=1e-6)

    def test_zero_variance_column(self):
        """Column with all same values should not produce NaN or inf"""
        X = np.array([[5.0, 1.0],
                      [5.0, 2.0],
                      [5.0, 3.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.all(np.isfinite(X_scaled)), "Expected finite values, got NaN or inf"

    def test_transform_before_fit_raises(self):
        """Calling transform before fit should raise an error"""
        scaler = StandardScaler()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(Exception):
            scaler.transform(X)

    def test_single_row(self):
        """Single row input should not crash and preserve shape"""
        X = np.array([[1.0, 2.0, 3.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert X_scaled.shape == (1, 3)

    def test_fit_transform_matches_fit_then_transform(self):
        """fit_transform should give identical result to fit then transform"""
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])
        s1 = StandardScaler()
        result1 = s1.fit_transform(X)

        s2 = StandardScaler()
        s2.fit(X)
        result2 = s2.transform(X)

        assert np.allclose(result1, result2)

    def test_transform_uses_training_stats(self):
        """Transform on test data should use stats from training data only"""
        X_train = np.array([[1.0], [3.0], [5.0]])
        X_test  = np.array([[7.0]])
        scaler  = StandardScaler()
        scaler.fit(X_train)
        X_test_scaled = scaler.transform(X_test)
        mean     = np.mean([1.0, 3.0, 5.0])
        std      = np.std([1.0, 3.0, 5.0])
        expected = (7.0 - mean) / std
        assert np.isclose(X_test_scaled[0, 0], expected)

    def test_negative_values(self):
        """Should handle negative values correctly"""
        X = np.array([[-3.0], [0.0], [3.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.isclose(X_scaled.mean(), 0.0, atol=1e-6)

    def test_large_values(self):
        """Should handle large values without overflow"""
        X = np.array([[1e10, 2e10],
                      [3e10, 4e10],
                      [5e10, 6e10]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.all(np.isfinite(X_scaled))