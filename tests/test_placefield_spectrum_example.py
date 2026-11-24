"""Session matching, fitted parameters, and signed-spectrum display checks."""

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from dimensionality_manuscript.figure_scripts.figure4 import (
    AdaptiveAlphaConfig, PlacefieldSpectrumExampleViewer,
)
from dimensionality_manuscript.figure_scripts.figure4._param_axes import FIT_KEY_PARAM_KEYS
from dimensionality_manuscript.figure_scripts.figure4._spectrum_math import _decay_fit_per_session


class Aggregator:
    param_axes = {}
    config_class = SimpleNamespace

    def __init__(self, arrays, ids):
        self.arrays = arrays
        self.session_ids = ids
        self.mouse_names = np.array(["mouse"] * len(ids))

    def sel(self, *, keys, **kwargs):
        return {key: self.arrays[key] for key in keys}

    def sel_objects(self, *, keys, **kwargs):
        return {"dist_centers": [np.arange(100)] * len(self.session_ids)}


def make_viewer(**kwargs):
    n = np.arange(1, 81.)
    power = np.array([n ** -2., n ** -3.])
    gaussian = np.array([np.exp(-n**2 / (2 * m**2)) for m in (9., 12.)])
    arrays = {"eig_control": gaussian}
    for key in ("eig_tilbury", "eig_shrinkage", "eig_better"):
        arrays[key] = power
    for key in {p for ps in FIT_KEY_PARAM_KEYS.values() for p in ps}:
        arrays[key] = np.ones((2, 4, 3))
    fits = Aggregator(arrays, ["s1", "s2"])
    cv = power.copy()
    cv[:, 60:] = -1e-7
    # Data backend is deliberately in the opposite order.
    stim = Aggregator({"ss_cvpca": cv[::-1], "ss_cv": power[::-1]}, ["s2", "s1"])
    defaults = dict(source_cfg=AdaptiveAlphaConfig("none", 0., 1, 2, 5),
                    fit_zone="fixed", fixed_range=(5, 40))
    return PlacefieldSpectrumExampleViewer(fits, stim, **(defaults | kwargs))


def test_parameter_recovery_and_alignment():
    viewer = make_viewer(session_id="s2")
    viewer.plot(viewer.state).canvas.draw()
    values = viewer._example_fits
    assert values["data"]["parameter"] == pytest.approx(3.)
    assert values["eig_shrinkage"]["parameter"] == pytest.approx(3.)
    assert values["eig_control"]["parameter"] == pytest.approx(12.)
    for key, spectra in viewer._spectra.items():
        model = values[key]["model"]
        _, params = _decay_fit_per_session(spectra[0], spectra[1], model, "fixed", (5, 40), 2)
        index = list(spectra[3]).index("s2")
        assert values[key]["parameter"] == pytest.approx(params[index])


def test_signed_markers_and_log_limits():
    viewer = make_viewer(ylim_log10=(-4, 1), xlim=(1, 70))
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    ax = fig.axes[0]
    assert ax.get_xscale() == ax.get_yscale() == "log"
    np.testing.assert_allclose(ax.get_ylim(), [1e-4, 10])
    negative = next(line for line in ax.lines if line.get_label() == "Data: -ve values")
    assert negative.get_marker() == "x"
    np.testing.assert_array_equal(negative.get_xdata(), np.arange(61, 71))
    np.testing.assert_allclose(negative.get_ydata(), 1e-4)
    updated = viewer.plot(dict(viewer.state, ylim_log10=(-8, 0)))
    negative = next(line for line in updated.axes[0].lines if line.get_label() == "Data: -ve values")
    np.testing.assert_allclose(negative.get_ydata(), 1e-8)


def test_adaptive_fallback_and_smoothing_keep_raw_negatives():
    viewer = make_viewer(fit_zone="adaptive", source_cfg=AdaptiveAlphaConfig("gaussian", 3., 1, 2, 5))
    fig = viewer.plot(dict(viewer.state, alpha_method="median_fpd", show_unfiltered=False))
    values = viewer._example_fits
    assert values["eig_control"]["window"] == values["data"]["window"]
    assert values["eig_shrinkage"]["window"] == values["data"]["window"]
    assert values["data"]["parameter"] == pytest.approx(2., abs=.1)
    assert any(line.get_marker() == "x" for line in fig.axes[0].lines)


def test_fit_selection_missing_sessions_and_static_state_refresh():
    viewer = make_viewer()
    state = dict(viewer.state, generalized_fit="both", include_better=True, normalize=True)
    viewer.plot(state)
    assert set(viewer._example_fits) == {"data", "eig_control", "eig_shrinkage", "eig_tilbury", "eig_better"}
    for raw, _, _, _ in viewer._spectra.values():
        np.testing.assert_allclose(raw.sum(axis=1), 1.)
    # Missing session in the data backend must never silently select another row.
    viewer.results_spectra.session_ids = ["s2", "missing"]
    viewer.refresh_data(state)
    fig = viewer.plot(state)
    assert "data" not in viewer._example_fits
    assert "Unavailable: Data" in [text.get_text() for text in fig.axes[0].texts]


def test_insufficient_fit_window_is_reported():
    viewer = make_viewer(fixed_range=(1, 3))
    fig = viewer.plot(viewer.state)
    assert all(np.isnan(value["parameter"]) for value in viewer._example_fits.values())
    assert any("fit unavailable" in label for label in fig.axes[0].get_legend_handles_labels()[1])


def _fit_bundle():
    gaussian = np.array([[0., 2., 2., 1.], [1., 1., 5., 2.], [0., 3., 7., 1.5]])
    generalized = np.column_stack([gaussian[:, :3], gaussian[:, 3], gaussian[:, 3], [2., 3., 4.]])
    return dict(theta=np.arange(10.), params=generalized, params_control=gaussian,
                params_shrinkage=generalized.copy(), fluorescence_scale=np.array([2., 5., .5]),
                r2_val=np.array([.8, .1, .9]), r2_val_control=np.array([.2, .7, .4]))


def test_activity_conversion_scales_neurons_before_covariance():
    from dimensionality_manuscript.configs.tilbury_fit import _eval_gaussian, _eval_tilbury
    from dimensionality_manuscript.figure_scripts.figure4.placefield_spectrum_example import _rescaled_model_spectra

    bundle = _fit_bundle()
    spectra = _rescaled_model_spectra(bundle)
    gaussian = np.array([_eval_gaussian(bundle["theta"], p) for p in bundle["params_control"]])
    generalized = np.array([_eval_tilbury(bundle["theta"], p) for p in bundle["params"]])
    better = np.where((bundle["r2_val"] > bundle["r2_val_control"])[:, None], generalized, gaussian)
    for key, curves in (("eig_control", gaussian), ("eig_tilbury", generalized),
                        ("eig_shrinkage", generalized), ("eig_better", better)):
        expected = np.linalg.eigvalsh(np.cov(curves * bundle["fluorescence_scale"][:, None]))[::-1]
        np.testing.assert_allclose(spectra[key], expected)
    # A uniform activity conversion has a squared effect on every eigenvalue.
    bundle["fluorescence_scale"][:] = 3.
    converted = _rescaled_model_spectra(bundle)
    bundle["fluorescence_scale"][:] = 1.
    original = _rescaled_model_spectra(bundle)
    for key in original:
        np.testing.assert_allclose(converted[key], 9. * original[key])


def test_source_activity_preset_and_session_drive_conversion(monkeypatch):
    from dimensionality_manuscript.figure_scripts.figure4 import placefield_fits
    from dimensionality_manuscript.figure_scripts.figure4.placefield_spectrum_example import _rescaled_model_spectra

    viewer = make_viewer()
    viewer.results_fit.config_class = SimpleNamespace(activity_parameters_name="raw")
    viewer.results_spectra.config_class = SimpleNamespace(activity_parameters_name="std")
    viewer.registry = object()
    calls = []
    bundle = _fit_bundle()

    def load(results, registry, sid, fit_params, **kwargs):
        calls.append((sid, kwargs["target_activity_parameters_name"], kwargs["target_full_session"]))
        return bundle

    monkeypatch.setattr(placefield_fits, "_load_fit_curves", load)
    viewer.refresh_data(viewer.state)
    assert calls == [("s1", "std", True)]
    expected = _rescaled_model_spectra(bundle, covariance_axis="neurons")
    np.testing.assert_allclose(viewer._spectra["eig_control"][0][0], expected["eig_control"])
    # Replotting and changing smoothing reuse the converted model spectra.
    viewer.plot(viewer.state)
    viewer.refresh_data(dict(viewer.state, source_smooth_width=1.))
    assert len(calls) == 1
    viewer.plot(dict(viewer.state, session_id="s2"))
    assert calls[-1] == ("s2", "std", True)


def test_model_matches_actual_ss_cvpca_estimator_for_identical_repeats():
    import torch
    from dimensionality_manuscript.configs.stimspace import _cvpca_svd
    from dimensionality_manuscript.configs.tilbury_fit import _eval_gaussian
    from dimensionality_manuscript.figure_scripts.figure4.placefield_spectrum_example import _rescaled_model_spectra

    fit = _fit_bundle()
    curves = np.array([_eval_gaussian(fit["theta"], p) for p in fit["params_control"]])
    curves *= fit["fluorescence_scale"][:, None]
    tensor = torch.tensor(curves, dtype=torch.float64)
    expected = _cvpca_svd(tensor, tensor, tensor).numpy()
    actual = _rescaled_model_spectra(fit, covariance_axis="neurons")["eig_control"]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    old = _rescaled_model_spectra(fit)["eig_control"]
    assert not np.allclose(old, expected)


def test_force_scale_matches_data_inside_window_without_changing_decay():
    viewer = make_viewer()
    # s1 is the second data-backend row. A large change outside the fitting
    # interval must not influence the multiplicative match.
    viewer.results_spectra.arrays["ss_cvpca"][1, :60] *= 4.
    viewer.results_spectra.arrays["ss_cvpca"][1, 0] *= 100.
    viewer.refresh_data(viewer.state)
    original = viewer.plot(viewer.state)
    original_fits = {key: dict(value) for key, value in viewer._example_fits.items()}
    original_data = next(line.get_ydata().copy() for line in original.axes[0].lines if line.get_label().startswith("Data:"))
    fig = viewer.plot(dict(viewer.state, force_scale=True))
    assert viewer._example_fits["eig_shrinkage"]["scale_factor"] == pytest.approx(4. * 99. / 100.)
    for key in ("eig_control", "eig_shrinkage"):
        fit = viewer._example_fits[key]
        assert fit["parameter"] == original_fits[key]["parameter"]
        assert fit["window"] == original_fits[key]["window"]
        np.testing.assert_allclose(fit["prediction"], original_fits[key]["prediction"] * fit["scale_factor"])
        start, end = fit["window"]
        raw, smooth = viewer._session_rows(viewer._spectra[key], "s1")
        data = viewer._session_rows(viewer._spectra["data"], "s1")[1]
        residual = np.log(data[start:end]) - np.log(smooth[start:end] * fit["scale_factor"])
        assert residual.mean() == pytest.approx(0., abs=1e-12)
        label = "Gaussian:" if key == "eig_control" else "Generalized:"
        dots = next(line for line in fig.axes[0].lines if line.get_label().startswith(label))
        np.testing.assert_allclose(dots.get_ydata(), raw * fit["scale_factor"])
    data = next(line.get_ydata() for line in fig.axes[0].lines if line.get_label().startswith("Data:"))
    np.testing.assert_allclose(data, original_data)
    assert viewer._example_fits["data"]["scale_factor"] == 1.


def test_force_scale_without_decay_overlay_and_without_positive_overlap():
    viewer = make_viewer(force_scale=True, generalized_decay="none")
    viewer.plot(viewer.state)
    assert viewer._example_fits["eig_shrinkage"]["scale_factor"] == pytest.approx(.99)
    # Negatives remain display markers, never observations for scale estimation.
    viewer.results_spectra.arrays["ss_cvpca"][:] = -1.
    viewer.refresh_data(viewer.state)
    viewer.plot(viewer.state)
    assert all(fit["scale_factor"] == 1. for fit in viewer._example_fits.values())
