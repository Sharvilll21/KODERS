import numpy as np
import pytest

from src.metrics.metrics import (
    accuracy,
    auc,
    confusion_matrix,
    f1,
    mse,
    precision,
    recall,
    roc_curve,
)
from src.stats import histogram, mean, quantiles, standard_deviation, variance


class TestMetricsAndStatsE2E:
    def test_binary_classification_report_values_are_consistent(self):
        """Evaluate a realistic classifier output across all binary metrics."""
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

        assert accuracy(y_true, y_pred) == pytest.approx(0.75)
        assert precision(y_true, y_pred) == pytest.approx(3 / 4)
        assert recall(y_true, y_pred) == pytest.approx(3 / 4)
        assert f1(y_true, y_pred) == pytest.approx(0.75)
        assert np.array_equal(confusion_matrix(y_true, y_pred), np.array([[3, 1], [1, 3]]))

    def test_metrics_handle_no_positive_predictions_without_crashing(self):
        """The small denominator guard should keep undefined precision/f1 finite."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0])

        assert precision(y_true, y_pred) == pytest.approx(0.0)
        assert recall(y_true, y_pred) == pytest.approx(0.0)
        assert f1(y_true, y_pred) == pytest.approx(0.0)
        assert np.isfinite(precision(y_true, y_pred))

    def test_regression_mse_matches_manual_squared_error_average(self):
        y_true = np.array([2.5, 0.0, 2.0, 8.0])
        y_pred = np.array([3.0, -0.5, 2.0, 7.0])

        expected = ((0.5 ** 2) + ((-0.5) ** 2) + 0.0 + ((-1.0) ** 2)) / 4

        assert mse(y_true, y_pred) == pytest.approx(expected)

    def test_roc_curve_thresholds_and_auc_for_ranked_scores(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        assert np.array_equal(thresholds, np.array([0.8, 0.4, 0.35, 0.1]))
        assert np.all(np.diff(fpr) >= -1e-12)
        assert np.all((0.0 <= tpr) & (tpr <= 1.0))
        assert auc(fpr, tpr) == pytest.approx(0.75)

    def test_descriptive_statistics_match_numpy_reference_values(self):
        values = np.array([1.0, 2.0, 2.0, 4.0, 9.0])

        assert mean(values) == pytest.approx(np.mean(values))
        assert variance(values) == pytest.approx(np.var(values))
        assert standard_deviation(values) == pytest.approx(np.std(values))
        assert quantiles(values, [0.25, 0.5, 0.75]) == pytest.approx(np.quantile(values, [0.25, 0.5, 0.75]))

    def test_histogram_returns_counts_and_bin_edges_for_distribution(self):
        values = np.array([0, 1, 1, 2, 3, 3, 3, 4], dtype=float)

        counts, bin_edges = histogram(values, bins=4)

        assert np.array_equal(counts, np.array([1, 2, 1, 4]))
        assert bin_edges == pytest.approx(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
