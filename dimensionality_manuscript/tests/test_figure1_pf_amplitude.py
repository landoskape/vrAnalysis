import numpy as np

from dimensionality_manuscript.figure_scripts.figure1.pf_amplitude import PlacefieldPeakAmplitude


class _Results:
    param_axes = {}

    def __init__(self, arrays):
        self.arrays = arrays

    def sel(self, *, keys, **kwargs):
        return {key: self.arrays[key] for key in keys}


def _viewer_with_cached_peaks():
    arrays = {
        "pf_peak": np.array([[[0.5, 1.5, 2.5], [2.5, 1.5, 0.25]]]),
        "pf_peak_hist": np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]),
        "pf_peak_n": np.array([[3.0, 3.0]]),
        "pf_peak_hist_edges": np.array([[0.0, 1.0, 2.0, 3.0]]),
        "reliability_slot": np.array([[[0.31, 0.3, np.nan], [0.9, 0.1, 0.4]]]),
        "env_slot_ids": np.array([[1.0, 2.0]]),
    }
    viewer = PlacefieldPeakAmplitude.__new__(PlacefieldPeakAmplitude)
    viewer.results = _Results(arrays)
    viewer.selection_names = ()
    viewer._load_arrays({})
    return viewer


def test_reliability_filter_is_slot_local_and_cached():
    viewer = _viewer_with_cached_peaks()

    np.testing.assert_allclose(
        viewer._peak_cache[True],
        [[[0.5, np.nan, np.nan], [2.5, np.nan, 0.25]]],
        equal_nan=True,
    )
    np.testing.assert_allclose(viewer._hist_cache[True], [[[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]])
    np.testing.assert_allclose(viewer._n_cache[True], [[1.0, 2.0]])


def test_filtered_counts_cover_pooled_slot_and_best_modes():
    viewer = _viewer_with_cached_peaks()
    state = {"filter_by_reliability": True, "env_slot": 1}

    counts, n = viewer._row_counts(0, {**state, "env_mode": "pooled"})
    np.testing.assert_allclose(counts, [2.0, 0.0, 1.0])
    assert n == 3

    counts, n = viewer._row_counts(0, {**state, "env_mode": "slot"})
    np.testing.assert_allclose(counts, [1.0, 0.0, 1.0])
    assert n == 2

    counts, n = viewer._row_counts(0, {**state, "env_mode": "best"})
    np.testing.assert_allclose(counts, [1.0, 0.0, 1.0])
    assert n == 2
