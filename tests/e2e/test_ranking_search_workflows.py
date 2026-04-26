import numpy as np
import pytest

from src.rank import percentile, rank
from src.sort_search import binary_search, stable_sort, topk


class TestRankingAndSearchWorkflows:
    def test_average_rank_assigns_mean_position_to_ties(self):
        scores = np.array([80, 70, 80, 90])

        ranks = rank(scores, method="average")

        assert ranks == pytest.approx(np.array([2.5, 1.0, 2.5, 4.0]))

    def test_dense_rank_does_not_leave_gaps_after_ties(self):
        scores = np.array([100, 80, 80, 70, 100])

        ranks = rank(scores, method="dense")

        assert ranks == pytest.approx(np.array([3.0, 2.0, 2.0, 1.0, 3.0]))

    def test_ordinal_rank_uses_sorted_position_even_when_values_tie(self):
        scores = np.array([5, 5, 1])

        ranks = rank(scores, method="ordinal")

        assert ranks == pytest.approx(np.array([2.0, 3.0, 1.0]))

    def test_rank_rejects_unknown_tie_method(self):
        with pytest.raises(ValueError, match="method must be"):
            rank([1, 2, 2], method="competition")

    def test_percentile_delegates_to_numpy_percentile_method(self):
        values = np.array([10, 20, 30, 40, 50], dtype=float)

        assert percentile(values, 50) == pytest.approx(30.0)
        assert percentile(values, 25, interpolation="nearest") == pytest.approx(20.0)

    def test_topk_returns_largest_values_in_descending_order_with_indices(self):
        values = np.array([4, 10, -1, 7, 10, 3])

        selected, indices = topk(values, 3, largest=True, return_indices=True)

        assert np.array_equal(selected, np.array([10, 10, 7]))
        assert np.array_equal(values[indices], selected)

    def test_topk_returns_smallest_values_in_ascending_order(self):
        values = np.array([4, 10, -1, 7, 10, 3])

        selected = topk(values, 3, largest=False)

        assert np.array_equal(selected, np.array([-1, 3, 4]))

    def test_topk_rejects_k_outside_array_bounds(self):
        with pytest.raises(ValueError, match="k must be between"):
            topk([1, 2, 3], 0)

        with pytest.raises(ValueError, match="k must be between"):
            topk([1, 2, 3], 4)

    def test_binary_search_finds_existing_value_and_missing_insertion_point(self):
        values = np.array([2, 4, 6, 8, 10])

        found_index, was_found = binary_search(values, 6)
        missing_index, missing_found = binary_search(values, 7)

        assert (found_index, was_found) == (2, True)
        assert (missing_index, missing_found) == (3, False)

    def test_stable_sort_returns_sorted_values_without_mutating_input(self):
        values = np.array([3, 1, 2, 1])

        sorted_values = stable_sort(values)

        assert np.array_equal(sorted_values, np.array([1, 1, 2, 3]))
        assert np.array_equal(values, np.array([3, 1, 2, 1]))
