"""Temporal PF / residual-gain / additive variance with covariance bookkeeping."""

import numpy as np
import pytest

from dimensionality_manuscript.configs import pfpred_quality as quality
from vrAnalysis.processors.placefields import FrameBehavior


def _assert_bookkeeping(result, prefix="", suffix=""):
    get = lambda metric: result[f"{prefix}{metric}{suffix}"]
    np.testing.assert_allclose(
        sum(get(f"var_{name}") for name in ("template", "gain", "additive", "covariance")),
        get("var_total"), atol=1e-12,
    )
    total = sum(get(f"frac_var_{name}") for name in ("template", "gain", "additive", "covariance"))
    valid = np.isfinite(total)
    np.testing.assert_allclose(total[valid], 1, atol=1e-12)


def test_temporal_components_match_explicit_trial_residual_projection():
    # Repeated positions get repeated frame weight. The additive components have zero
    # uncentered dot product with P within each trial, but nonzero means/covariances.
    p = np.array([1., 1., 2., 4., 1., 2., 2., 4.])
    trial = np.array([3] * 4 + [17] * 4)
    delta = np.repeat([-0.5, 0.75], 4)
    additive = np.array([2., -1., 1., -0.75, 2., -1., -1., 0.5])
    modulation = delta * p
    activity = p + modulation + additive
    result = quality._temporal_variance_components(activity[:, None], p[:, None], trial)
    for name, trace in (("total", activity), ("template", p), ("gain", modulation), ("additive", additive)):
        np.testing.assert_allclose(result[f"var_{name}"], [np.var(trace)])
    expected_covariance = 2 * sum(np.cov(left, right, ddof=0)[0, 1] for left, right in (
        (p, modulation), (p, additive), (modulation, additive),
    ))
    np.testing.assert_allclose(result["var_covariance"], [expected_covariance])
    assert abs(expected_covariance) > 0.01
    assert result["var_n_frames"][0] == len(p)
    _assert_bookkeeping(result)


def test_cancellation_preserves_fractions_above_one_and_negative_covariance():
    p = np.array([[2.], [4.], [2.], [4.]])
    result = quality._temporal_variance_components(p / 2, p, np.array([3, 3, 9, 9]))
    for name, value in {"template": 4., "gain": 1., "additive": 0., "covariance": -4.}.items():
        np.testing.assert_allclose(result[f"frac_var_{name}"], [value])
    _assert_bookkeeping(result)


def test_zero_template_is_all_additive_and_constant_activity_has_undefined_fractions():
    activity = np.array([[0., 1.], [1., 1.], [3., 1.]])
    result = quality._temporal_variance_components(activity, np.zeros_like(activity), np.ones(3))
    np.testing.assert_allclose(result["var_gain"], 0)
    np.testing.assert_allclose(result["var_template"], 0)
    np.testing.assert_allclose(result["var_additive"], np.var(activity, axis=0))
    assert result["frac_var_additive"][0] == 1
    for name in ("template", "gain", "additive", "covariance"):
        assert np.isnan(result[f"frac_var_{name}"][1])
    _assert_bookkeeping(result)


def test_nonfinite_frames_use_one_common_mask_per_roi():
    activity = np.array([[1., np.nan], [3., 2.], [np.inf, 4.], [5., 6.], [1e6, 1e6]])
    prediction = np.array([[2., 1.], [2., 1.], [2., np.nan], [4., 3.], [1., 1.]])
    trial = np.array([3., 3., 3., 9., np.nan])
    result = quality._temporal_variance_components(activity, prediction, trial)
    np.testing.assert_array_equal(result["var_n_frames"], [3., 2.])
    for roi, indices in enumerate(([0, 1, 3], [1, 3])):
        expected = quality._temporal_variance_components(
            activity[indices, roi:roi + 1], prediction[indices, roi:roi + 1], trial[indices],
        )
        for key in expected:
            np.testing.assert_allclose(result[key][roi], expected[key][0])
    _assert_bookkeeping(result)


@pytest.mark.parametrize("scale", [1e-8, 3.])
def test_fractions_are_invariant_to_activity_units(scale):
    p = np.array([[1.], [2.], [3.], [1.], [2.], [3.]])
    activity = np.array([[3.], [1.], [5.], [1.], [3.], [2.]])
    trial = np.repeat([3, 9], 3)
    reference = quality._temporal_variance_components(activity, p, trial)
    scaled = quality._temporal_variance_components(scale * activity, scale * p, trial)
    for name in ("template", "gain", "additive", "covariance"):
        np.testing.assert_allclose(scaled[f"frac_var_{name}"], reference[f"frac_var_{name}"])
        np.testing.assert_allclose(scaled[f"var_{name}"], scale**2 * reference[f"var_{name}"])


def _fold_data():
    trials = np.array([3, 9, 17, 25, 42])
    maps = np.array([
        [[1., 2., 4.], [2., 3., 3.], [2., 3., 7.], [5., 1., 2.], [2., 7., 5.]],
        [[2., 1., 3.], [1., 4., 2.], [3., 4., 5.], [3., 2., 4.], [6., 4., 1.]],
    ])
    # More than one ROI chunk, and uneven occupancy across position/trial.
    maps = np.tile(maps, (65, 1, 1))
    bins = np.array([0, 0, 1, 2, 0, 1, 1, 2, 0, 1, 2, 2, 0, 0, 1, 2, 0, 1, 2, 2])
    frame_trials = np.repeat(trials, 4)
    activity = np.stack([maps[:, i, bins[i * 4:(i + 1) * 4]].T for i in range(len(trials))]).reshape(-1, maps.shape[0])
    return trials, maps, bins, frame_trials, activity


def _templates(maps):
    return {fold: np.mean(values, axis=1) for fold, values in {
        "all": maps, "f1": maps[:, ::2], "f2": maps[:, 1::2],
    }.items()}


def test_fold_comparisons_use_average_trial_maps_and_actual_evaluation_frames():
    trials, maps, bins, frame_trials, activity = _fold_data()
    templates = _templates(maps)
    result = quality._placefield_variance_by_fold(activity, bins, frame_trials, trials, templates)
    assert set(result) == set(quality.pf_variance_keys())
    folds = {"all": trials, "f1": trials[::2], "f2": trials[1::2]}
    for comparison in quality.PF_VARIANCE_COMPARISONS[:5]:
        evaluation, template = comparison.split("_from_")
        indices = np.flatnonzero(np.isin(frame_trials, folds[evaluation]))
        expected = quality._temporal_variance_components(
            activity[indices], templates[template][:, bins[indices]].T, frame_trials[indices],
        )
        for metric, values in expected.items():
            np.testing.assert_allclose(result[f"pf_{metric}_{comparison}"], values)
        _assert_bookkeeping(result, "pf_", f"_{comparison}")
    for average, directions in {"cv": ("f2_from_f1", "f1_from_f2"),
                                "within_cv": ("f1_from_f1", "f2_from_f2")}.items():
        for metric in quality.PF_VARIANCE_METRICS:
            expected = sum(result[f"pf_{metric}_{direction}"] for direction in directions)
            if metric != "var_n_frames":
                expected /= 2
            np.testing.assert_allclose(result[f"pf_{metric}_{average}"], expected)
        _assert_bookkeeping(result, "pf_", f"_{average}")
    assert result["pf_var_n_frames_all_from_all"][0] == len(activity)
    assert not np.isclose(result["pf_var_total_all_from_all"][0], np.var(maps[0]))

    # Held-out changes cannot affect the training-only decomposition or training PF.
    changed = maps.copy()
    changed[:, 1::2] *= 10
    changed_activity = activity.copy()
    changed_activity[np.isin(frame_trials, trials[1::2])] *= 10
    perturbed = quality._placefield_variance_by_fold(
        changed_activity, bins, frame_trials, trials, _templates(changed),
    )
    for metric in quality.PF_VARIANCE_METRICS:
        np.testing.assert_allclose(perturbed[f"pf_{metric}_f1_from_f1"], result[f"pf_{metric}_f1_from_f1"])
    assert not np.allclose(perturbed["pf_var_gain_f2_from_f1"], result["pf_var_gain_f2_from_f1"])


def test_fold_decomposition_excludes_invalid_bins_trials_and_missing_templates():
    trials, maps, bins, frame_trials, activity = _fold_data()
    reference = quality._placefield_variance_by_fold(activity, bins, frame_trials, trials, _templates(maps))
    expanded = quality._placefield_variance_by_fold(
        np.concatenate([activity, np.full((4, activity.shape[1]), 1e6)]),
        np.r_[bins, -1, 3, 0, 0], np.r_[frame_trials, 3, 3, 1000, np.nan], trials, _templates(maps),
    )
    for key in reference:
        np.testing.assert_allclose(expanded[key], reference[key])

    templates = _templates(maps)
    templates["f2"][:] = np.nan
    result = quality._placefield_variance_by_fold(activity, bins, frame_trials, trials, templates)
    assert np.all(np.isnan(result["pf_frac_var_gain_cv"]))
    assert np.all(result["pf_var_n_frames_f1_from_f2"] == 0)


@pytest.mark.parametrize("n_trials", [0, 1])
def test_empty_evaluation_or_template_folds_remain_undefined(n_trials):
    trials = np.arange(n_trials)
    templates = {"all": np.ones((2, 3)), "f1": np.ones((2, 3)), "f2": np.full((2, 3), np.nan)}
    result = quality._placefield_variance_by_fold(
        np.ones((n_trials * 3, 2)), np.tile(np.arange(3), n_trials), np.repeat(trials, 3), trials, templates,
    )
    assert np.all(np.isnan(result["pf_var_total_f2_from_f1"]))
    assert np.all(result["pf_var_n_frames_f2_from_f1"] == 0)
    assert np.all(np.isnan(result["pf_frac_var_template_cv"]))


def test_population_summaries_cover_every_key_and_quality_subset():
    per_roi = {key: np.array([1., np.nan, 3.]) for key in quality.pf_variance_keys()}
    result = quality._summarize_metrics_across_rois(per_roi, np.array([True, True, False]))
    assert set(result) == set(quality.pf_variance_population_summary_keys())
    for key in per_roi:
        assert result[f"mean_{key}"] == 2
        assert result[f"median_quality_filtered_{key}"] == 1
        assert result[f"mean_notquality_filtered_{key}"] == 3
def test_gain_unexplained_fraction_counts_residual_mean_and_silent_trials():
    template = np.array([[1.], [2.], [1.], [2.]])
    activity = np.array([[2.], [1.], [0.], [0.]])
    result = quality._temporal_variance_components(activity, template, np.array([1, 1, 2, 2]))
    # The active trial has gain 4/5. The silent trial has gain zero.
    predicted = np.array([[0.8], [1.6], [0.], [0.]])
    expected = np.sum((activity - predicted)**2) / np.sum((activity - activity.mean())**2)
    np.testing.assert_allclose(result["frac_unexplained"], expected)
    assert result["frac_unexplained"][0] > result["frac_var_additive"][0]


def test_regional_scores_share_the_error_but_only_weighted_scores_recenter_activity():
    template = np.array([0., 1., 2., 0., 1., 2.])
    activity = np.array([1., 1., 3., 2., 0., 4.])
    trials = np.repeat([1, 2], 3)
    result = quality._temporal_variance_components(
        activity[:, None], template[:, None], trials, template_peak=np.array([4.]),
    )
    gain = (activity.reshape(2, 3) @ template[:3]) / (template[:3] @ template[:3])
    residual = activity - template * np.repeat(gain, 3)
    for region, w in (("within", template / 4), ("outside", 1 - template / 4)):
        sse = sum(w * residual**2)
        shared_sst = sum((activity - activity.mean())**2)
        regional_sst = sum(w * (activity - np.average(activity, weights=w))**2)
        np.testing.assert_allclose(result[f"residual_rms_{region}"], np.sqrt(sse / w.sum()))
        np.testing.assert_allclose(result[f"frac_unexplained_{region}_shared"], sse / shared_sst)
        np.testing.assert_allclose(result[f"frac_unexplained_{region}_weighted"], sse / regional_sst)
        np.testing.assert_allclose(
            result[f"r2_{region}_shared"], (sum(w * (activity - activity.mean())**2) - sse) / shared_sst,
        )
    np.testing.assert_allclose(
        result["frac_unexplained_within_shared"] + result["frac_unexplained_outside_shared"],
        result["frac_unexplained"],
    )
    np.testing.assert_allclose(
        result["r2_within_shared"] + result["r2_outside_shared"], 1 - result["frac_unexplained"],
    )


def test_regional_residual_rms_remains_defined_at_zero_data_variance_with_positive_weight():
    template = np.array([[1., 1., 0.], [1., 2., 0.], [1., 0., 0.]])
    result = quality._temporal_variance_components(np.ones_like(template), template, np.ones(3))
    np.testing.assert_allclose(result["residual_rms_within"], [0., np.sqrt(0.08), np.nan])
    np.testing.assert_allclose(result["residual_rms_outside"], [np.nan, np.sqrt(0.72), np.nan])
    assert np.isnan(result["r2_within_shared"]).all()


def test_gamma_reference_refits_gain_on_each_draw_using_fixed_templates():
    edges = np.arange(4.)
    bins = np.tile(np.arange(3), 4)
    trials = np.arange(4)
    frame_trials = np.repeat(trials, 3)
    behavior = FrameBehavior(bins + 0.5, np.ones(12), np.ones(12), frame_trials,
                             np.zeros(12), np.zeros(12))
    template = np.array([[0., 1., 3.]])
    fixed = dict(all=template, f1=template * 0.5, f2=template * 2)
    activity = np.tile([0., 1., 3.], 4)[:, None]
    rng_seed = 42
    result = quality._gamma_reference(
        behavior, activity, bins, trials, fixed, edges, 1., None, 4,
        np.random.default_rng(rng_seed),
    )
    rng = np.random.default_rng(rng_seed)
    samples, gain_means = [], []
    for _ in range(4):
        sim = rng.standard_gamma(template[:, bins].T)
        template_on_frames = template[:, bins].T
        components = quality._temporal_variance_components(sim, template_on_frames, frame_trials)
        samples.append(components["frac_unexplained"])
        gain_means.append(np.mean(sim.reshape(4, 3) @ template[0] / np.sum(template[0]**2)))
    np.testing.assert_allclose(result["gamma_mean_pf_frac_unexplained_all_from_all"], np.mean(samples, axis=0))
    np.testing.assert_allclose(result["gamma_std_pf_frac_unexplained_all_from_all"], np.std(samples, axis=0, ddof=1))
    np.testing.assert_allclose(result["gamma_mean_trial_gain_least_squares_mean"], np.mean(gain_means))

