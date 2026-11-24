"""Trial RMS after fitting a fold-specific PF gain, rather than temporal variance."""

import numpy as np
import pytest

from dimensionality_manuscript.configs.pfpred_quality import (
    PF_VARIANCE_COMPARISONS, _trial_residual_rms_summaries, residual_rms_summary_keys,
)


def test_fold_residual_statistics_match_explicit_trial_fits():
    maps = np.array([[[1., 2., 4.], [3., 2., 1.], [1., 4., 3.],
                      [2., np.nan, 5.], [4., 1., 2.]]])
    result = _trial_residual_rms_summaries(maps)
    assert set(result) == set(residual_rms_summary_keys())
    rows = {"all": np.arange(5), "f1": np.array([0, 2, 4]), "f2": np.array([1, 3])}
    for fold in PF_VARIANCE_COMPARISONS[:5]:
        evaluation, training = fold.split("_from_")
        template = np.nanmean(maps[0, rows[training]], axis=0)
        rms = []
        for trial in rows[evaluation]:
            activity = maps[0, trial]
            valid = np.isfinite(activity) & np.isfinite(template)
            gain = np.linalg.lstsq(template[valid, None], activity[valid], rcond=None)[0][0]
            rms.append(np.sqrt(np.mean((activity[valid] - gain * template[valid])**2)))
        expected = {"mean": np.mean(rms), "variance": np.var(rms), "cv": np.std(rms) / np.mean(rms)}
        for statistic, value in expected.items():
            np.testing.assert_allclose(result[f"trial_residual_rms_{statistic}_{fold}"], [value])
    for average, directions in (("cv", ("f2_from_f1", "f1_from_f2")),
                                ("within_cv", ("f1_from_f1", "f2_from_f2"))):
        for statistic in ("mean", "variance", "cv"):
            prefix = f"trial_residual_rms_{statistic}"
            np.testing.assert_allclose(result[f"{prefix}_{average}"],
                                       (result[f"{prefix}_{directions[0]}"] + result[f"{prefix}_{directions[1]}"]) / 2)


@pytest.mark.parametrize("ntrials", [0, 1, 4])
def test_zero_energy_and_missing_trials_are_undefined(ntrials):
    maps = np.zeros((2, ntrials, 3))
    maps[1] = np.nan
    result = _trial_residual_rms_summaries(maps)
    assert all(np.isnan(values).all() for values in result.values())


def test_pure_multiplicative_activity_has_no_residual_and_no_defined_cv():
    maps = np.array([[[0., 1., 2.], [0., 2., 4.], [0., 3., 6.], [0., 4., 8.]]])
    result = _trial_residual_rms_summaries(maps)
    for fold in PF_VARIANCE_COMPARISONS:
        np.testing.assert_allclose(result[f"trial_residual_rms_mean_{fold}"], 0, atol=1e-14)
        np.testing.assert_allclose(result[f"trial_residual_rms_variance_{fold}"], 0, atol=1e-28)
        assert np.isnan(result[f"trial_residual_rms_cv_{fold}"]).all()
