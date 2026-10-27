"""Aggregation and layout checks for the supplemental estimator comparison."""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from dimensionality_manuscript import average_by_mouse
from dimensionality_manuscript.figure_scripts.figure4 import SpectrumComparisonViewer
from dimensionality_manuscript.figure_scripts.figure4._alpha_config import AdaptiveAlphaConfig
from dimensionality_manuscript.figure_scripts.figure4._spectrum_math import _signed_participation_ratio


class Aggregator:
    def __init__(self, arrays, names, ids):
        self.arrays = arrays
        self.mouse_names = np.asarray(names)
        self.session_ids = ids
        self.param_axes = {}

    def sel(self, *, keys, avg_by_mouse=False, **kwargs):
        return {key: average_by_mouse(self.arrays[key], self.mouse_names) if avg_by_mouse else self.arrays[key] for key in keys}


def make_viewer(**kwargs):
    ranks = np.arange(1, 81.)
    spectra = np.array([ranks ** -exponent for exponent in (1.5, 2., 2.5)])
    cv = spectra.copy()
    cv[:, 65:] = -1e-6
    env = lambda x: np.stack([x, 2 * x], axis=1)
    stim = Aggregator(
        dict(sf_cv_env_full1=env(spectra), ss_cv_env=env(spectra), ss_cvpca_env=env(cv),
             sf_cv=spectra, ss_cv=spectra, ss_cvpca=cv, ff=spectra),
        ["a", "a", "b"], ["s1", "s2", "s3"],
    )
    # Deliberately different backend session order: fallback windows must align by id.
    svca = Aggregator(dict(ff=cv[::-1]), ["b", "a", "a"], ["s3", "s2", "s1"])
    cfg = AdaptiveAlphaConfig("none", 0., 1, 2, 5)
    return SpectrumComparisonViewer(stim, svca, source_cfg=cfg, full_cfg=cfg, **kwargs)


def test_comparison_has_requested_methods_and_mouse_level_measurements():
    viewer = make_viewer(normalize=False)
    pf, full = viewer._comparison_rows
    assert list(pf) == ["sf_cv", "ss_cvpca", "ss_cv"]
    assert list(full) == ["ff", "SVCA"]
    raw = viewer.results.arrays["sf_cv"] * 1.5  # mean of the two environments
    np.testing.assert_allclose(pf["sf_cv"]["dimensionality"], _signed_participation_ratio(average_by_mouse(raw, ["a", "a", "b"])))
    for methods in (pf, full):
        for data in methods.values():
            assert data["dimensionality"].shape == (2,)
            assert data["alpha"].shape == (2,)
        np.testing.assert_allclose(methods[next(iter(methods))]["alpha"], [1.75, 2.5], atol=.01)
    # Backend mouse order is b, a, preserving identity within each source.
    np.testing.assert_allclose(full["SVCA"]["alpha"], [2.5, 1.75], atol=.01)


def test_plot_layout_and_refresh_after_data_controls_change(tmp_path):
    viewer = make_viewer()
    state = dict(viewer.state, source_mode="all", normalize=False)
    viewer.refresh_data(state)
    np.testing.assert_allclose(viewer._comparison_rows[0]["sf_cv"]["spectrum"], average_by_mouse(viewer.results.arrays["sf_cv"], ["a", "a", "b"]))
    fig = viewer.plot(state)
    assert len(fig.axes) == 6
    assert [tick.get_text() for tick in fig.axes[1].get_xticklabels()] == ["PF x CA1", "r-cvPCA", "PF x PF"]
    assert fig.axes[0].get_xscale() == fig.axes[0].get_yscale() == "log"
    assert fig.axes[1].get_yscale() == "linear"
    assert fig.axes[4].get_yscale() == "log"
    for row, methods in enumerate(viewer._comparison_rows):
        for col, metric in ((1, "dimensionality"), (2, "alpha")):
            values = np.concatenate([data[metric] for data in methods.values()])
            low = 1 if row == 1 and col == 1 else 0
            np.testing.assert_allclose(fig.axes[row * 3 + col].get_ylim(), [low, np.ceil(np.nanmax(values))])
    for ax in fig.axes:
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale == "log":
                ticks = axis.get_majorticklocs()
                np.testing.assert_allclose(np.log10(ticks), np.round(np.log10(ticks)))
                assert len(axis.get_minorticklocs()) == 0
    linear = viewer.plot(dict(state, dimensionality_log=False))
    assert linear.axes[4].get_yscale() == "linear"
    assert linear.axes[4].get_ylim()[0] == 0
    fig.savefig(tmp_path / "comparison.png", dpi=120)


def test_unfiltered_display_is_independent_of_smoothing_and_preserves_statistics():
    viewer = make_viewer(pr_pre_smooth=False, show_unfiltered=True)
    for prefix in ("source", "full"):
        viewer.update_selection(f"{prefix}_smooth_method", value="gaussian")
        viewer.update_float(f"{prefix}_smooth_width", value=6.)
    state = dict(viewer.state)
    viewer.refresh_data(state)
    before = [[(data["alpha"].copy(), data["dimensionality"].copy()) for data in row.values()] for row in viewer._comparison_rows]
    fig = viewer.plot(state)
    for index, row in enumerate(viewer._comparison_rows):
        first = next(iter(row.values()))
        np.testing.assert_allclose(fig.axes[index * 3].lines[0].get_ydata(), first["raw_spectrum"][0])
        assert not np.allclose(first["raw_spectrum"], first["spectrum"], equal_nan=True)
    fig = viewer.plot(dict(state, show_unfiltered=False))
    for index, row in enumerate(viewer._comparison_rows):
        first = next(iter(row.values()))
        np.testing.assert_allclose(fig.axes[index * 3].lines[0].get_ydata(), first["spectrum"][0])
        for data, (alpha, pr) in zip(row.values(), before[index]):
            np.testing.assert_allclose(data["alpha"], alpha)
            np.testing.assert_allclose(data["dimensionality"], pr)


def test_optional_reg_covariances_uses_cvpca_backend_and_whole_session_spectra():
    ranks = np.arange(1, 81.)
    spectra = np.array([ranks ** -2.5, ranks ** -1.5, ranks ** -2.])
    cvpca = Aggregator(dict(reg_covariances_fixed=spectra), ["b", "a", "a"], ["s3", "s1", "s2"])
    viewer = make_viewer(results_cvpca=cvpca, include_reg_covariances_fixed=True)
    assert list(viewer._comparison_rows[0]) == ["sf_cv", "ss_cvpca", "ss_cv", "reg_covariances_fixed"]
    data = viewer._comparison_rows[0]["reg_covariances_fixed"]
    expected = average_by_mouse(spectra, cvpca.mouse_names)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(data["spectrum"], expected)
    np.testing.assert_allclose(data["alpha"], [2.5, 1.75], atol=.01)
    viewer.refresh_data(dict(viewer.state, include_reg_covariances_fixed=False))
    assert "reg_covariances_fixed" not in viewer._comparison_rows[0]
    with pytest.raises(ValueError, match="requires results_cvpca"):
        make_viewer(include_reg_covariances_fixed=True)
