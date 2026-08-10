import numpy as np

from dimensionality_manuscript.figure_scripts.figure1.reliability import ReliabilityHistogramViewer


def _viewer_with_mice():
    viewer = ReliabilityHistogramViewer.__new__(ReliabilityHistogramViewer)
    viewer.mouse_names = np.array(["mouse_b", "mouse_a", "mouse_c", "mouse_a", "mouse_b"])
    return viewer


def test_ordered_mice_sorted_by_decreasing_mean_place_cell_fraction():
    viewer = _viewer_with_mice()
    fractions = np.array([0.2, 0.8, 0.5, 0.6, 0.4])

    mice = viewer._ordered_mice(fractions, {"mouse_order": "sorted", "random_seed": 0})

    assert mice == ["mouse_a", "mouse_c", "mouse_b"]


def test_ordered_mice_random_uses_seeded_permutation():
    viewer = _viewer_with_mice()
    fractions = np.array([0.2, 0.8, 0.5, 0.6, 0.4])
    original_order = ["mouse_b", "mouse_a", "mouse_c"]
    state = {"mouse_order": "random", "random_seed": 17}

    mice = viewer._ordered_mice(fractions, state)
    expected = np.random.default_rng(17).permutation(original_order).tolist()

    assert mice == expected
    assert viewer._ordered_mice(fractions, state) == expected
