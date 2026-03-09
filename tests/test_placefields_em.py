import numpy as np
import pytest

pytest.importorskip("sklearn")

from vrAnalysis.processors.em import _huber_irls_weights_from_pred, _mstep, ExpMaxConfig
from vrAnalysis.processors.placefields import FrameBehavior, _get_placefield_fast_sampling, get_placefield


def _make_frame_behavior() -> FrameBehavior:
    return FrameBehavior(
        position=np.array([0.1, 0.2, 1.1, 1.2], dtype=float),
        speed=np.array([2.0, 2.0, 2.0, 2.0], dtype=float),
        environment=np.array([1, 1, 1, 1], dtype=float),
        trial=np.array([0, 0, 0, 0], dtype=float),
    )


def test_get_placefield_default_weights_match_explicit_ones():
    spks = np.array([[1.0], [3.0], [5.0], [7.0]])
    frame_behavior = _make_frame_behavior()
    dist_edges = np.array([0.0, 1.0, 2.0], dtype=float)

    pf_default = get_placefield(spks, frame_behavior, dist_edges, average=True, use_fast_sampling=False)
    pf_ones = get_placefield(
        spks,
        frame_behavior,
        dist_edges,
        average=True,
        use_fast_sampling=False,
        weights=np.ones(len(frame_behavior), dtype=float),
    )

    np.testing.assert_allclose(pf_default.placefield, pf_ones.placefield)
    np.testing.assert_allclose(pf_default.count, pf_ones.count)


def test_get_placefield_weighted_average_and_counts():
    spks = np.array([[1.0], [3.0], [5.0], [7.0]])
    frame_behavior = _make_frame_behavior()
    dist_edges = np.array([0.0, 1.0, 2.0], dtype=float)
    weights = np.array([1.0, 3.0, 2.0, 2.0], dtype=float)

    pf = get_placefield(spks, frame_behavior, dist_edges, average=True, use_fast_sampling=False, weights=weights)

    expected_count = np.array([[4.0, 4.0]])
    expected_pf = np.array([[[2.5], [6.0]]])

    np.testing.assert_allclose(pf.count, expected_count)
    np.testing.assert_allclose(pf.placefield, expected_pf)


def test_get_placefield_zero_weights_exclude_samples():
    spks = np.array([[1.0], [3.0], [5.0], [7.0]])
    frame_behavior = _make_frame_behavior()
    dist_edges = np.array([0.0, 1.0, 2.0], dtype=float)
    weights = np.array([1.0, 0.0, 0.0, 2.0], dtype=float)

    pf = get_placefield(spks, frame_behavior, dist_edges, average=True, use_fast_sampling=False, weights=weights)

    expected_count = np.array([[1.0, 2.0]])
    expected_pf = np.array([[[1.0], [7.0]]])

    np.testing.assert_allclose(pf.count, expected_count)
    np.testing.assert_allclose(pf.placefield, expected_pf)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.array([[1.0], [1.0]]), "1D"),
        (np.array([1.0, 1.0, 1.0]), "same length"),
        (np.array([1.0, -1.0, 1.0, 1.0]), "non-negative"),
        (np.array([1.0, np.nan, 1.0, 1.0]), "finite"),
        (np.array([1.0, np.inf, 1.0, 1.0]), "finite"),
    ],
)
def test_get_placefield_invalid_weights(weights, message):
    spks = np.array([[1.0], [3.0], [5.0], [7.0]])
    frame_behavior = _make_frame_behavior()
    dist_edges = np.array([0.0, 1.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match=message):
        get_placefield(spks, frame_behavior, dist_edges, average=True, use_fast_sampling=False, weights=weights)


def test_get_placefield_fast_sampling_kernel_uses_frame_weights_with_sample_duration():
    placefield = np.zeros((1, 2, 1), dtype=float)
    counts = np.zeros((1, 2), dtype=float)
    spks = np.array([[2.0], [6.0]], dtype=float)
    behave_position_bin = np.array([0, 1, 0], dtype=np.int64)
    behave_env_or_trial_indices = np.array([0, 0, 0], dtype=np.int64)
    idx_behave_to_frame = np.array([0, 1, 1], dtype=np.int64)
    difference_timestamps = np.array([0.0, 0.0, 0.0], dtype=float)
    dist_cutoff = 0.5
    idx_valid_samples = np.array([True, True, True])
    idx_valid_frames = np.array([True, True])
    fast_frame = np.array([True, True])
    sample_duration = np.array([0.5, 1.0, 2.0], dtype=float)
    idx_to_spks = np.array([0, 1], dtype=np.int64)
    weights = np.array([2.0, 0.25], dtype=float)

    _get_placefield_fast_sampling(
        placefield,
        counts,
        spks,
        behave_position_bin,
        behave_env_or_trial_indices,
        idx_behave_to_frame,
        difference_timestamps,
        dist_cutoff,
        idx_valid_samples,
        idx_valid_frames,
        fast_frame,
        sample_duration,
        True,
        idx_to_spks,
        weights,
    )

    expected_counts = np.array([[1.5, 0.25]])
    expected_placefield = np.array([[[13.0 / 3.0], [6.0]]])

    np.testing.assert_allclose(counts, expected_counts)
    np.testing.assert_allclose(placefield[:, :, 0] / counts, expected_placefield)


def test_get_placefield_fast_sampling_kernel_uses_frame_weights_without_sample_duration():
    placefield = np.zeros((1, 1, 1), dtype=float)
    counts = np.zeros((1, 1), dtype=float)
    spks = np.array([[2.0], [6.0]], dtype=float)
    behave_position_bin = np.array([0, 0], dtype=np.int64)
    behave_env_or_trial_indices = np.array([0, 0], dtype=np.int64)
    idx_behave_to_frame = np.array([0, 1], dtype=np.int64)
    difference_timestamps = np.array([0.0, 0.0], dtype=float)
    dist_cutoff = 0.5
    idx_valid_samples = np.array([True, True])
    idx_valid_frames = np.array([True, True])
    fast_frame = np.array([True, True])
    sample_duration = np.array([0.5, 1.5], dtype=float)
    idx_to_spks = np.array([0, 1], dtype=np.int64)
    weights = np.array([1.0, 0.5], dtype=float)

    _get_placefield_fast_sampling(
        placefield,
        counts,
        spks,
        behave_position_bin,
        behave_env_or_trial_indices,
        idx_behave_to_frame,
        difference_timestamps,
        dist_cutoff,
        idx_valid_samples,
        idx_valid_frames,
        fast_frame,
        sample_duration,
        False,
        idx_to_spks,
        weights,
    )

    expected_count = np.array([[1.5]])
    expected_pf = np.array([[[10.0 / 3.0]]])

    np.testing.assert_allclose(counts, expected_count)
    np.testing.assert_allclose(placefield / counts[..., None], expected_pf)


def test_huber_irls_weights_downweight_outlier():
    spks = np.zeros((5, 2), dtype=float)
    pred = np.zeros((5, 2), dtype=float)
    spks[-1] = np.array([10.0, 0.0])

    weights = _huber_irls_weights_from_pred(spks, pred, k=1.5)

    assert weights.shape == (5,)
    assert np.all(np.isfinite(weights))
    np.testing.assert_allclose(weights[:4], np.ones(4))
    assert 0.0 <= weights[-1] < 1.0


def test_huber_irls_weights_perfect_prediction_are_one():
    spks = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    weights = _huber_irls_weights_from_pred(spks, spks.copy())
    np.testing.assert_allclose(weights, np.ones(len(spks)))


def test_mstep_without_weights_matches_explicit_none():
    spks = np.array([[1.0], [3.0], [5.0], [7.0]])
    frame_behavior = _make_frame_behavior()
    dist_edges = np.array([0.0, 1.0, 2.0], dtype=float)

    pf_default = _mstep(spks, frame_behavior, dist_edges, smooth_width=None)
    pf_none = _mstep(spks, frame_behavior, dist_edges, smooth_width=None, weights=None)

    np.testing.assert_allclose(pf_default.placefield, pf_none.placefield)
    np.testing.assert_allclose(pf_default.count, pf_none.count)


def test_expmax_config_defaults_enable_huber_irls():
    config = ExpMaxConfig()

    assert config.use_huber_irls is True
    assert config.huber_k == 2.5
    assert config.huber_eps == 1e-8
