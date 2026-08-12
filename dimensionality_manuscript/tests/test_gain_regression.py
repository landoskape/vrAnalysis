"""Tests for the gain-regression config's numerical helpers and bookkeeping."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dimensionality_manuscript.configs.gain_regression import (
    CURVE_KEYS,
    GAIN_TRANSFORMS,
    MAX_ENV_SLOTS,
    PLACEFIELD_SPLITS,
    GainRegressionConfig,
    _full_trial_indices,
    _stack_curves,
    apply_gain_transform,
    evaluate_null_rolls,
    gaussian_gain_matrix,
    optimize_rrr,
    roll_gain_rows,
    score_rrr,
    split_tensors,
    split_trials,
)

POSITIONS = np.linspace(0.0, 100.0, 40)


def _place_fields(n_rois: int, rng: np.random.Generator) -> np.ndarray:
    """Noiseless Gaussian place fields, ``(rois, bins)``, exactly representable by the fit."""
    centers = rng.uniform(25.0, 75.0, n_rois)
    return np.exp(-0.5 * ((POSITIONS[None, :] - centers[:, None]) / 9.0) ** 2)


def _trial_maps(place_fields: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Build ``(rois, trials, bins)`` maps as place field times per-trial gain."""
    return place_fields[:, None, :] * gain[:, :, None]


def test_full_trial_filter_ignores_bins_outside_the_required_range():
    occupancy = np.array(
        [
            [np.nan, 0.0, 1.0, 1.0, 0.0, np.nan],
            [np.nan, 0.0, 1.0, np.nan, 0.0, np.nan],
        ]
    )
    np.testing.assert_array_equal(_full_trial_indices(occupancy, np.arange(1, 5)), [0])


def test_gaussian_gain_recovers_planted_multipliers():
    rng = np.random.default_rng(0)
    place_fields = _place_fields(6, rng)
    # Gains are normalized to a mean of one per ROI so the trial-averaged place field equals the
    # underlying field and the recovered gain is the planted gain itself.
    gain = rng.uniform(0.5, 1.8, (6, 15))
    gain /= np.mean(gain, axis=1, keepdims=True)

    recovered, fitted, prediction = gaussian_gain_matrix(_trial_maps(place_fields, gain), POSITIONS)

    np.testing.assert_allclose(recovered, gain, atol=1e-6)
    np.testing.assert_allclose(fitted, place_fields, atol=1e-5)
    np.testing.assert_allclose(prediction, place_fields, atol=1e-12)


def test_gaussian_gain_is_invariant_to_per_neuron_rescaling():
    """Any per-neuron scalar cancels between the gain numerator and denominator.

    The Gaussian model is linear in its baseline and amplitude, so scaling one ROI's activity by
    ``c`` scales the fitted curve by exactly ``c`` and leaves the center and width -- and therefore
    the normalized bump used as spatial weights -- untouched. Numerator and denominator both pick
    up ``c`` and the ratio is unchanged. This is why no activity-scaling parameter belongs on this
    config: std scaling, max scaling and no scaling at all give identical gain.
    """
    rng = np.random.default_rng(12)
    maps = _trial_maps(_place_fields(8, rng), rng.uniform(0.4, 2.5, (8, 12)))
    scale = rng.uniform(0.01, 100.0, 8)  # four orders of magnitude between neurons

    baseline, _, _ = gaussian_gain_matrix(maps, POSITIONS)
    rescaled, _, _ = gaussian_gain_matrix(maps * scale[:, None, None], POSITIONS)

    np.testing.assert_allclose(rescaled, baseline, rtol=1e-6)


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
def test_gaussian_gain_returns_nan_when_the_place_field_cannot_be_fit():
    rng = np.random.default_rng(1)
    maps = _trial_maps(_place_fields(3, rng), np.ones((3, 8)))
    maps[1, :, 3:] = np.nan  # three finite bins, below the four-parameter fit's minimum

    recovered, fitted, _ = gaussian_gain_matrix(maps, POSITIONS)

    assert np.all(np.isnan(recovered[1]))
    assert np.all(np.isnan(fitted[1]))
    np.testing.assert_allclose(recovered[[0, 2]], 1.0, atol=1e-6)


def test_prediction_trials_keep_held_out_trials_out_of_their_own_denominator():
    rng = np.random.default_rng(2)
    place_fields = _place_fields(4, rng)
    train = np.arange(10)
    held_out = np.arange(10, 14)
    gain = np.ones((4, 14))
    gain[:, held_out] = 2.0  # held-out trials are uniformly twice as active

    maps = _trial_maps(place_fields, gain)
    train_based, _, _ = gaussian_gain_matrix(maps, POSITIONS, prediction_trials=train)
    all_based, _, _ = gaussian_gain_matrix(maps, POSITIONS)

    # Against a train-only fit the held-out gain is its true value of 2.
    np.testing.assert_allclose(train_based, gain, atol=1e-6)
    # Against an all-trial fit every gain is deflated by the shared all-trial mean, which is the
    # leakage the "train" placefield_split exists to avoid.
    np.testing.assert_allclose(all_based, gain / np.mean(gain, axis=1, keepdims=True), atol=1e-6)


def test_sqrt_transform_compresses_the_tail_without_displacing_zeros():
    gain = np.array([[0.0, 0.25, 1.0, 4.0, 49.0]])

    np.testing.assert_array_equal(apply_gain_transform(gain, "raw"), gain)
    transformed = apply_gain_transform(gain, "sqrt")
    np.testing.assert_allclose(transformed, [[0.0, 0.5, 1.0, 2.0, 7.0]])

    # Silent trials stay at zero and a gain of one stays at one, so the scale keeps its meaning;
    # only the tail moves. A log transform would send the zero to -inf, which is the point of sqrt.
    assert transformed[0, 0] == 0.0
    assert transformed[0, 2] == 1.0
    assert np.ptp(transformed) < np.ptp(gain)


def test_sqrt_transform_marks_negative_gain_nan_for_the_finite_screen():
    transformed = apply_gain_transform(np.array([[1.0, -0.5]]), "sqrt")

    assert transformed[0, 0] == 1.0
    assert np.isnan(transformed[0, 1])

    with pytest.raises(ValueError):
        apply_gain_transform(np.array([[1.0]]), "cube_root")


def test_split_trials_is_stratified_disjoint_and_reproducible():
    trial_environment = np.array([1] * 12 + [3] * 8)
    folds = split_trials(trial_environment, (1.0, 0.25, 0.25), seed=7)

    assert len(folds) == 3
    combined = np.concatenate(folds)
    np.testing.assert_array_equal(np.sort(combined), np.arange(trial_environment.size))
    assert combined.size == np.unique(combined).size

    # Each environment is cut independently at the normalized cumulative fractions (2/3, 1/6, 1/6).
    counts = [[int(np.sum(trial_environment[fold] == env)) for fold in folds] for env in (1, 3)]
    assert counts == [[8, 2, 2], [5, 1, 2]]

    repeated = split_trials(trial_environment, (1.0, 0.25, 0.25), seed=7)
    for first, second in zip(folds, repeated):
        np.testing.assert_array_equal(first, second)
    different = split_trials(trial_environment, (1.0, 0.25, 0.25), seed=8)
    assert any(not np.array_equal(first, second) for first, second in zip(folds, different))


def test_split_trials_restores_the_global_random_state():
    np.random.seed(123)
    expected = np.random.random(4)

    np.random.seed(123)
    split_trials(np.array([1] * 10), (1.0, 0.25, 0.25), seed=99)
    np.testing.assert_array_equal(np.random.random(4), expected)


def _low_rank_gain(n_trials: int, n_source: int, n_target: int, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Source and target gain sharing a rank-``rank`` latent, centered on a gain of one.

    Both sides carry independent private noise. Without it the source would be exactly rank
    ``rank`` and every higher rank would score identically, which says nothing about whether the
    search can find the shared dimensionality.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_trials, rank))
    source = 1.0 + 0.05 * (latent @ rng.normal(size=(rank, n_source))) + 0.02 * rng.normal(size=(n_trials, n_source))
    target = 1.0 + 0.05 * (latent @ rng.normal(size=(rank, n_target))) + 0.02 * rng.normal(size=(n_trials, n_target))
    return source, target


def test_optimize_rrr_finds_a_low_rank_solution_that_generalizes():
    rank = 3
    source, target = _low_rank_gain(360, 30, 25, rank, seed=3)
    tensors = {}
    for name, rows in (("train", slice(0, 200)), ("val", slice(200, 280)), ("test", slice(280, 360))):
        tensors[name] = (
            torch.tensor(source[rows], dtype=torch.float64),
            torch.tensor(target[rows], dtype=torch.float64),
        )

    fit = optimize_rrr(*tensors["train"], *tensors["val"], alpha_bounds=(1e-2, 1e6), max_iterations=25)

    # The intercept column raises the achievable rank by one over the source-neuron count.
    assert fit.max_rank == min(30 + 1, 25, 200)
    assert fit.rank <= 2 * rank  # a genuinely low-rank solution, far below max_rank
    assert fit.alpha > 0

    mse_test, r2_test = score_rrr(fit.model, fit.rank, *tensors["test"])
    assert r2_test > 0.5
    assert mse_test > 0

    # The rank curve is the same model scored at every rank, so it must agree at the winning rank.
    ranks, scores = fit.model.score_curve(
        *tensors["test"], ranks=list(range(1, fit.max_rank + 1)), nonnegative=True, dim=None, verbose=False
    )
    assert scores["r2"][ranks.index(fit.rank)] == pytest.approx(r2_test)
    # The planted rank already captures nearly all the predictable structure.
    assert scores["r2"][ranks.index(rank)] > 0.5


def test_optimize_rrr_prefers_the_smallest_rank_when_the_curve_is_flat():
    # A noiseless source is exactly rank 3, so with the intercept every rank from 4 up predicts
    # identically. The parsimony tie-break must report that plateau's start, not max_rank.
    rng = np.random.default_rng(11)
    latent = rng.normal(size=(300, 3))
    source = 1.0 + 0.05 * (latent @ rng.normal(size=(3, 20)))
    target = 1.0 + 0.05 * (latent @ rng.normal(size=(3, 18)))
    tensors = {
        name: (torch.tensor(source[rows], dtype=torch.float64), torch.tensor(target[rows], dtype=torch.float64))
        for name, rows in (("train", slice(0, 200)), ("val", slice(200, 300)))
    }

    fit = optimize_rrr(*tensors["train"], *tensors["val"], alpha_bounds=(1e-2, 1e6), max_iterations=25)

    # Three latent dimensions plus the constant offset: rank 4 predicts exactly, and so does every
    # rank above it. The sparse golden-section search cannot see where that plateau starts.
    assert fit.rank == 4
    assert fit.rank <= fit.rank_gss
    assert fit.max_rank == 18


def test_roll_gain_rows_shifts_every_neuron_by_its_own_nonzero_offset():
    rng = np.random.default_rng(0)
    gain = rng.normal(size=(40, 9))

    rolled = roll_gain_rows(gain, np.random.default_rng(1))

    offsets = []
    for row, rolled_row in zip(gain, rolled):
        shifts = [shift for shift in range(9) if np.array_equal(np.roll(row, shift), rolled_row)]
        # Each row is a circular shift of itself -- the marginal distribution is untouched -- and
        # never the identity, so no neuron keeps its original trial alignment.
        assert shifts == [shifts[0]] and shifts[0] != 0
        offsets.append(shifts[0])
    # Offsets are independent per neuron; a common offset would preserve all shared structure.
    assert len(set(offsets)) > 1

    single = np.ones((3, 1))
    np.testing.assert_array_equal(roll_gain_rows(single, np.random.default_rng(2)), single)


def _null_inputs(n_trials: int = 180, n_source: int = 15, n_target: int = 15, rank: int = 3, seed: int = 21):
    """A ``(neurons, trials)`` gain matrix with shared low-rank structure, plus its partitions."""
    source, target = _low_rank_gain(n_trials, n_source, n_target, rank, seed)
    gain = np.concatenate((source.T, target.T), axis=0)
    source_rows = np.arange(n_source)
    target_rows = np.arange(n_source, n_source + n_target)
    train_end, val_end = int(n_trials * 0.55), int(n_trials * 0.78)
    columns = {
        "train": np.arange(0, train_end),
        "val": np.arange(train_end, val_end),
        "test": np.arange(val_end, n_trials),
    }
    return gain, source_rows, target_rows, columns


def test_null_rolls_remove_the_shared_structure_the_real_fit_uses():
    gain, source_rows, target_rows, columns = _null_inputs()
    tensors = split_tensors(gain, source_rows, target_rows, columns)
    fit = optimize_rrr(*tensors["train"], *tensors["val"], alpha_bounds=(1e-2, 1e6), max_iterations=25)
    _, r2_test = score_rrr(fit.model, fit.rank, *tensors["test"])

    null = evaluate_null_rolls(
        gain,
        source_rows,
        target_rows,
        columns,
        alpha=fit.alpha,
        rank=fit.rank,
        ranks=fit.ranks,
        n_rolls=5,
        rng=np.random.default_rng(0),
    )

    # The rolled data keeps every neuron's own gain distribution, so the intercept still works and
    # the null is not degenerate -- but the cross-neuron latent it was predicting from is gone.
    assert r2_test > 0.5
    assert null.r2_test < 0.1
    assert null.mse_test > 0
    # The null curves live on the real fit's rank grid so the two can be read against each other.
    for key in ("mse_val_curve", "r2_val_curve", "mse_test_curve", "r2_test_curve"):
        assert getattr(null, key).shape == (len(fit.ranks),)
    assert fit.val_mse[fit.ranks.index(fit.rank)] < null.mse_val_curve[fit.ranks.index(fit.rank)]
    assert 1 <= null.rank <= fit.max_rank


def test_null_rolls_are_reproducible_and_skippable():
    gain, source_rows, target_rows, columns = _null_inputs(n_trials=140, n_source=8, n_target=8, rank=2, seed=5)
    kwargs = dict(alpha=1.0, rank=3, ranks=list(range(1, 8)), n_rolls=3)

    first = evaluate_null_rolls(gain, source_rows, target_rows, columns, rng=np.random.default_rng(4), **kwargs)
    repeated = evaluate_null_rolls(gain, source_rows, target_rows, columns, rng=np.random.default_rng(4), **kwargs)
    different = evaluate_null_rolls(gain, source_rows, target_rows, columns, rng=np.random.default_rng(5), **kwargs)

    np.testing.assert_array_equal(first.mse_test_curve, repeated.mse_test_curve)
    assert first.r2_test == repeated.r2_test
    assert first.r2_test != different.r2_test

    skipped = evaluate_null_rolls(gain, source_rows, target_rows, columns, **{**kwargs, "n_rolls": 0}, rng=np.random.default_rng(4))
    assert np.isnan(skipped.r2_test) and np.isnan(skipped.rank)
    assert skipped.mse_val_curve.shape == (7,) and np.all(np.isnan(skipped.mse_val_curve))


def test_stack_curves_pads_missing_and_short_slots_with_nan():
    curves = {
        0: {key: np.arange(4, dtype=float) for key in CURVE_KEYS},
        2: {key: np.arange(2, dtype=float) for key in CURVE_KEYS},
    }
    stacked = _stack_curves(curves)

    for key in CURVE_KEYS:
        assert stacked[key].shape == (MAX_ENV_SLOTS, 4)
        np.testing.assert_array_equal(stacked[key][0], np.arange(4))
        np.testing.assert_array_equal(stacked[key][2][:2], np.arange(2))
        assert np.all(np.isnan(stacked[key][2][2:]))
        assert np.all(np.isnan(stacked[key][1]))

    empty = _stack_curves({})
    assert all(np.all(np.isnan(empty[key])) for key in CURVE_KEYS)


def _environment_inputs(n_rois: int, n_trials: int, seed: int, silent_rois: tuple[int, ...] = ()) -> tuple[SimpleNamespace, dict]:
    """A session stub and loaded-data dict for one synthetic environment."""
    rng = np.random.default_rng(seed)
    place_fields = _place_fields(n_rois, rng)
    place_fields[list(silent_rois)] = 0.0
    gain = rng.uniform(0.6, 1.6, (n_rois, n_trials))
    trial_maps = _trial_maps(place_fields, gain)
    full_trials = np.arange(n_trials)
    data = {
        "env_maps": SimpleNamespace(environments=[1], spkmap=[trial_maps]),
        "positions": POSITIONS,
        "full_trials": full_trials,
        "trials_by_environment": {1: full_trials},
        "roi_lookup": np.arange(n_rois) * 2,  # non-trivial mapping back to session ROI indices
    }
    return SimpleNamespace(session_uid="test-session"), data


def test_environment_gain_applies_the_configured_transform():
    session, data = _environment_inputs(n_rois=12, n_trials=18, seed=4)
    folds = [np.arange(0, 12), np.arange(12, 15), np.arange(15, 18)]

    raw = GainRegressionConfig(gain_transform="raw")._environment_gain(session, data, env=1, folds=folds)
    root = GainRegressionConfig(gain_transform="sqrt")._environment_gain(session, data, env=1, folds=folds)

    # The screen and the source/target split run off the same seeds, so only the values differ.
    np.testing.assert_array_equal(raw.roi_indices, root.roi_indices)
    np.testing.assert_array_equal(raw.source_rows, root.source_rows)
    np.testing.assert_allclose(root.gain, np.sqrt(raw.gain))


def test_environment_gain_partitions_neurons_and_trials():
    config = GainRegressionConfig()
    session, data = _environment_inputs(n_rois=12, n_trials=18, seed=4)
    folds = [np.arange(0, 12), np.arange(12, 15), np.arange(15, 18)]

    gain_data = config._environment_gain(session, data, env=1, folds=folds)

    assert gain_data is not None
    np.testing.assert_array_equal(gain_data.columns["train"], np.arange(0, 12))
    np.testing.assert_array_equal(gain_data.columns["val"], np.arange(12, 15))
    np.testing.assert_array_equal(gain_data.columns["test"], np.arange(15, 18))
    assert gain_data.n_neurons_env == 12
    assert gain_data.gain.shape == (gain_data.n_neurons_kept, 18)

    # The two halves partition the surviving rows exactly once each.
    combined = np.concatenate((gain_data.source_rows, gain_data.target_rows))
    np.testing.assert_array_equal(np.sort(combined), np.arange(gain_data.gain.shape[0]))
    assert abs(gain_data.source_rows.size - gain_data.target_rows.size) <= 1
    # ROI indices are reported in session coordinates, not gain-row coordinates.
    assert np.all(np.isin(gain_data.roi_indices, data["roi_lookup"]))
    assert gain_data.roi_indices.size == gain_data.gain.shape[0]


def test_environment_gain_screens_out_silent_rois():
    config = GainRegressionConfig()
    session, data = _environment_inputs(n_rois=12, n_trials=18, seed=4, silent_rois=(0, 5))
    folds = [np.arange(0, 12), np.arange(12, 15), np.arange(15, 18)]

    gain_data = config._environment_gain(session, data, env=1, folds=folds)

    # A silent ROI has no trial-to-trial structure, so reliability and fraction-active reject it
    # before it can reach the regression as an uninformative constant row.
    assert gain_data is not None
    assert gain_data.n_neurons_env == 12
    assert gain_data.n_neurons_kept == 10
    assert not np.any(np.isin(data["roi_lookup"][[0, 5]], gain_data.roi_indices))


def test_environment_gain_declines_a_split_with_too_few_trials():
    config = GainRegressionConfig()
    session, data = _environment_inputs(n_rois=12, n_trials=18, seed=5)
    # Two trials in the test fold is below MIN_TRIALS_PER_SPLIT.
    folds = [np.arange(0, 13), np.arange(13, 16), np.arange(16, 18)]

    assert config._environment_gain(session, data, env=1, folds=folds) is None


def test_environment_gain_declines_too_few_neurons():
    config = GainRegressionConfig()
    session, data = _environment_inputs(n_rois=3, n_trials=18, seed=6)
    folds = [np.arange(0, 12), np.arange(12, 15), np.arange(15, 18)]

    assert config._environment_gain(session, data, env=1, folds=folds) is None


def test_process_assembles_results_on_the_experience_order_axis(monkeypatch):
    rng = np.random.default_rng(8)
    n_rois = 14
    trial_counts = {1: 24, 3: 20}
    trial_environment = np.concatenate([np.full(count, env) for env, count in trial_counts.items()])
    full_trials = np.arange(trial_environment.size)

    spkmap = []
    for env, count in trial_counts.items():
        gain = rng.uniform(0.6, 1.6, (n_rois, count))
        spkmap.append(_trial_maps(_place_fields(n_rois, rng), gain))

    data = {
        "env_maps": SimpleNamespace(environments=list(trial_counts), spkmap=spkmap),
        "positions": POSITIONS,
        "full_trials": full_trials,
        "trials_by_environment": {env: full_trials[trial_environment == env] for env in trial_counts},
        "roi_lookup": np.arange(n_rois),
    }
    session = SimpleNamespace(session_uid="uid", mouse_name="mouse", trial_environment=trial_environment)

    config = GainRegressionConfig(n_null_rolls=3)
    monkeypatch.setattr(GainRegressionConfig, "_load_session", lambda self, _session: data)
    # This mouse met environment 3 first, so env 3 owns slot 0 and env 1 owns slot 1. Results must
    # follow experience order, not environment-index order.
    monkeypatch.setattr("dimensionality_manuscript.configs.gain_regression.load_env_order", lambda: {"mouse": [3, 1]})

    result = config.process(session, registry=None)

    np.testing.assert_array_equal(result["env_slot_ids"][:2], [3, 1])
    assert np.isnan(result["env_slot_ids"][2])
    np.testing.assert_array_equal(result["n_trials_env"][:2], [20, 24])
    np.testing.assert_array_equal(result["n_trials_train"][:2], [13, 16])

    scalar_keys = ("rrr_alpha", "rrr_rank", "rrr_rank_gss", "max_rank", "r2_test", "mse_test", "r2_val", "mse_val")
    null_keys = ("rrr_rank_null", "r2_test_null", "mse_test_null", "r2_val_null", "mse_val_null")
    for key in scalar_keys + null_keys:
        assert np.all(np.isfinite(result[key][:2])), key
        assert np.isnan(result[key][2]), key  # the unvisited third slot stays NaN

    assert np.all(result["rrr_rank"][:2] <= result["max_rank"][:2])
    assert result["reliability"].shape == (MAX_ENV_SLOTS, n_rois)

    width = int(np.nanmax(result["max_rank"]))
    for key in CURVE_KEYS:
        assert result[key].shape == (MAX_ENV_SLOTS, width), key
        assert np.all(np.isnan(result[key][2]))

    assert [matrix is None for matrix in result["gain_matrices"]] == [False, False, True]
    assert result["gain_matrices"][0].shape[1] == 20
    np.testing.assert_array_equal(result["trial_columns"][0]["test"].size, result["n_trials_test"][0])
    assert set(GainRegressionConfig._result_handling) <= set(result)


def test_config_validation_and_variations():
    assert GainRegressionConfig.display_name == "gain_regression"

    variations = GainRegressionConfig.generate_variations()
    assert {variation.placefield_split for variation in variations} == set(PLACEFIELD_SPLITS)
    assert {variation.gain_transform for variation in variations} == set(GAIN_TRANSFORMS)
    assert len(variations) == len(PLACEFIELD_SPLITS) * len(GAIN_TRANSFORMS)
    assert len({variation.key() for variation in variations}) == len(variations)
    assert len({variation.summary() for variation in variations}) == len(variations)

    with pytest.raises(ValueError):
        GainRegressionConfig(placefield_split="nonsense")
    with pytest.raises(ValueError):
        GainRegressionConfig(gain_transform="log")
    with pytest.raises(ValueError):
        GainRegressionConfig(trial_fractions=(1.0, 0.25))
    with pytest.raises(ValueError):
        GainRegressionConfig(trial_fractions=(1.0, 0.0, 0.25))
    with pytest.raises(ValueError):
        GainRegressionConfig(spks_type="not_a_spks_type")
    with pytest.raises(ValueError):
        GainRegressionConfig(n_null_rolls=-1)
