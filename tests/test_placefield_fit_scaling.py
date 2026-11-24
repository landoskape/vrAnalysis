"""Rebuilt test curves must use the activity units of the selected stored fits."""
from types import SimpleNamespace

import numpy as np
import pytest

from dimensionality_manuscript.configs.tilbury_fit import TilburyFitConfig, _SPLITS, _eval_gaussian
from dimensionality_manuscript.figure_scripts.figure4 import placefield_fits as module
from dimensionality_manuscript.figure_scripts.figure4.placefield_fits import _load_fit_curves

import matplotlib.pyplot as plt


@pytest.mark.parametrize("base_preset", ["raw", "default"])
@pytest.mark.parametrize("selected_preset", ["raw", "default", "std"])
@pytest.mark.parametrize("target_preset", ["std", "raw", "default"])
def test_rebuilt_curves_follow_selected_activity_preset(monkeypatch, base_preset, selected_preset, target_preset):
    config = TilburyFitConfig(activity_parameters_name=base_preset, num_bins=3)
    raw = np.array([[0.0, 20.0, 100.0]])
    population = SimpleNamespace(data=raw, idx_neurons=np.array([0]), idx_samples=np.arange(3))
    seen_scales = []

    def apply_split(data, *, time_idx, prefiltered, scale, scale_type, pre_split):
        seen_scales.append((scale, scale_type, pre_split))
        if not scale:
            return data
        divisor = data.std(axis=1, ddof=1, keepdims=True) if scale_type == "std" else data.max(axis=1, keepdims=True)
        return data / divisor

    population.apply_split = apply_split
    population.get_split_times = lambda *args, **kwargs: np.arange(3)
    behavior = SimpleNamespace(filter=lambda indices: None)
    registry = SimpleNamespace(get_population=lambda *args: (population, behavior),
                               time_split={split: i for i, split in enumerate(_SPLITS)})
    monkeypatch.setattr(TilburyFitConfig, "_avg_placefield",
                        lambda self, spks, *args: (spks.T, np.ones(3)))
    session = SimpleNamespace(trial_environment=np.array([0]), environments=[0], env_length=[200])
    stored = {key: [np.zeros((1, 6))] for key in ("params", "params_control", "params_shrinkage")}
    stored.update(r2_test=[np.ones(1)], r2_test_control=[np.ones(1)], idx_keep=[np.ones(1, dtype=bool)])
    results = SimpleNamespace(config_class=config, sessions=[session], _session_index={"example": 0},
                              sel=lambda **kwargs: stored)

    fit = _load_fit_curves(results, registry, "example", {"activity_parameters_name": selected_preset},
                           target_activity_parameters_name=target_preset)

    std = raw.std(axis=1, ddof=1, keepdims=True)
    expected = {"raw": raw, "default": raw / 100, "std": raw / std}[selected_preset]
    np.testing.assert_allclose(fit["test_curve"], expected)
    np.testing.assert_allclose(fit["train_curve"], expected)
    np.testing.assert_allclose(fit["validation_curve"], expected)
    assert fit["best_env"] == 0
    expected_scale = {"default": (True, "max", True), "raw": (False, "none", True), "std": (True, "std", True)}[selected_preset]
    calls = [expected_scale] * len(_SPLITS)
    if selected_preset != target_preset:
        calls += [{"default": (True, "max", True), "raw": (False, "none", True), "std": (True, "std", True)}[target_preset]] * len(_SPLITS)
    assert seen_scales == calls
    target = {"raw": raw, "default": raw / 100, "std": raw / std}[target_preset]
    np.testing.assert_allclose(fit["test_curve"] * fit["fluorescence_scale"][:, None], target)
    assert config.activity_parameters_name == base_preset


@pytest.mark.parametrize("text_only", [False, True])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("sharey", [False, True])
def test_figure_units_and_y_axes(monkeypatch, independent, sharey, text_only):
    theta = np.array([25., 75., 125., 175.])
    data = np.array([[0., 3., 4., 0.], [0., 15., 20., 0.]])
    gaussian = np.array([[0., 2., 80., 20.], [0., 10., 80., 20.]])
    generalized = np.array([[0., 3., 80., 20., 20., 2.], [0., 15., 80., 20., 20., 2.]])
    bundle = dict(theta=theta, test_curve=data, params=generalized,
                  params_control=gaussian, params_shrinkage=generalized,
                  idx_neurons=np.array([4, 8]), idx_keep=np.ones(2, dtype=bool),
                  fluorescence_scale=np.array([10., 10.]))
    monkeypatch.setattr(module, "_load_fit_curves", lambda *args, **kwargs: bundle)
    results = SimpleNamespace(config_class=TilburyFitConfig(), param_axes={})
    viewer = module.PlacefieldFitFigureViewer(
        results, None, ["session", "session"], [4, 8], n_rows=1, n_cols=2,
        normalize_independent=independent, sharey=sharey, fontsize=13.,
        data_color="#c44e52", legend_text_only=text_only,
    )
    fig = viewer.plot(viewer.state)
    try:
        axes = fig.axes
        for i, ax in enumerate(axes):
            expected_data = data[i] / np.std(data[i]) if independent else data[i] * 10
            prediction = _eval_gaussian(theta, gaussian[i])
            expected_fit = prediction / np.std(prediction) if independent else prediction * 10
            assert ax.lines[0].get_color() == "#c44e52"
            np.testing.assert_allclose(ax.lines[0].get_ydata(), expected_data)
            np.testing.assert_allclose(ax.lines[1].get_ydata(), expected_fit)
        assert axes[0].get_ylabel() == ("Activity (a.u.)" if independent else "Norm. Fluor.")
        assert axes[1].get_ylabel() == ""
        assert axes[0].get_shared_y_axes().joined(*axes) == sharey
        assert axes[1].spines["left"].get_visible() == (not sharey)
        assert bool(axes[1].get_yticklabels()) == (not sharey)
        legend = axes[0].get_legend()
        assert all(text.get_fontsize() == 13 for text in legend.get_texts())
        assert all(handle.get_visible() == (not text_only) for handle in legend.legend_handles)
        if text_only:
            from matplotlib.colors import to_rgba
            for text, line in zip(legend.get_texts(), axes[0].lines):
                np.testing.assert_allclose(to_rgba(text.get_color()), to_rgba(line.get_color()))
            assert legend.handlelength == legend.handletextpad == 0
        if sharey:
            np.testing.assert_allclose(axes[0].get_ylim(), axes[1].get_ylim())
        elif not independent:
            assert axes[1].get_ylim()[1] > axes[0].get_ylim()[1]
        for ax in axes:
            labels = ax.get_yticklabels()
            if labels:
                np.testing.assert_allclose([float(label.get_text()) for label in labels], ax.get_yticks())
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_stimspace_conversion_uses_full_recording_scaling_statistics(monkeypatch):
    raw = np.array([[0., 20., 100., 1000.]])
    vr = raw[:, :3]
    population = SimpleNamespace(data=raw, idx_neurons=np.array([0]), idx_samples=np.arange(3))
    calls = []

    def apply_split(data, *, time_idx, prefiltered, scale, scale_type, pre_split):
        calls.append(prefiltered)
        scaled = data / data.std(axis=1, ddof=1, keepdims=True)
        return scaled if prefiltered else scaled[:, :3]

    population.apply_split = apply_split
    population.get_split_times = lambda *args, **kwargs: np.arange(3)
    registry = SimpleNamespace(get_population=lambda *args: (population, SimpleNamespace(filter=lambda _: None)),
                               time_split={s: i for i, s in enumerate(_SPLITS)})
    session = SimpleNamespace(trial_environment=np.array([0]), environments=[0], env_length=[200])
    stored = {key: [np.zeros((1, 6))] for key in ("params", "params_control", "params_shrinkage")}
    stored.update(r2_test=[np.ones(1)], r2_test_control=[np.ones(1)], idx_keep=[np.ones(1, dtype=bool)])
    results = SimpleNamespace(config_class=TilburyFitConfig(activity_parameters_name="std", num_bins=3, activity_scaling_scope="retained_vr"),
                              sessions=[session], _session_index={"example": 0}, sel=lambda **kwargs: stored)
    monkeypatch.setattr(TilburyFitConfig, "_avg_placefield", lambda self, spks, *args: (spks.T, np.ones(3)))
    fit = _load_fit_curves(results, registry, "example", {}, target_full_session=True)
    np.testing.assert_allclose(fit["test_curve"], vr / vr.std(axis=1, ddof=1, keepdims=True))
    np.testing.assert_allclose(fit["test_curve"] * fit["fluorescence_scale"][:, None],
                               vr / raw.std(axis=1, ddof=1, keepdims=True))
    assert calls == [True] * len(_SPLITS) + [False] * len(_SPLITS)
