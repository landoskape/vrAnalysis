"""Numerical and cache checks for the single-reference PF projection diagnostic."""

from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import torch

from dimilibi import gaussian_filter
from dimilibi.cvpca import CVPCA
from dimilibi.helpers import vector_correlation
from dimensionality_manuscript.configs.cvpca import CVPCAConfig
from dimensionality_manuscript.configs.stimspace import (
    StimSpaceSpectraConfig, StimspaceSVCAConfig, _cvpca_svd, _cvsvd, _to_g, _svca_placefield_folds,
)
from vrAnalysis.processors.placefields import Placefield
from dimensionality_manuscript.figure_scripts.figure4 import PlacefieldProjectionViewer
from dimensionality_manuscript.figure_scripts.figure4 import placefield_projections as module


def _folds():
    rng = torch.Generator().manual_seed(71)
    signal = torch.randn(35, 9, generator=rng, dtype=torch.float64)
    return [signal + torch.randn(35, 9, generator=rng, dtype=torch.float64) for _ in range(4)]


def test_ss_projections_reproduce_original_covariance_for_each_reference():
    folds = _folds()
    prepared = {"train": folds, "test": folds}
    for reference in range(4):
        fit = module._fit_projections("ss_cvpca", StimSpaceSpectraConfig(), prepared, reference)
        assert reference not in fit.projections
        first, second, _ = sorted(fit.projections)
        # Independent transcription of the original estimator, including its centering axis.
        train = folds[reference] - folds[reference].mean(dim=0, keepdim=True)
        _, _, vt = torch.linalg.svd(train, full_matrices=False)
        p1 = (folds[first] - folds[first].mean(dim=0, keepdim=True)) @ vt.T
        p2 = (folds[second] - folds[second].mean(dim=0, keepdim=True)) @ vt.T
        expected = vector_correlation(p1, p2, covariance=True, dim=0, center=True)
        actual = vector_correlation(
            torch.from_numpy(fit.projections[first]), torch.from_numpy(fit.projections[second]),
            covariance=True, dim=0, center=True,
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(_cvpca_svd(folds[reference], folds[first], folds[second]), expected)
        np.testing.assert_allclose(fit.projections[first], p1.numpy())


@pytest.mark.parametrize("center", [True, False])
def test_fixed_projections_match_original_fit_and_score_with_missing_bin(center):
    config = CVPCAConfig(num_bins=10, center=center)
    folds = _folds()[:3]
    full_centers = np.arange(10, dtype=float) * 10 + 5
    centers = np.delete(full_centers, 4)
    prepared = {"torch_pfs": folds, "dist_centers": centers, "full_dist_centers": full_centers}
    for reference in range(3):
        fit = module._fit_projections("reg_covariances_fixed", config, prepared, reference)
        smooth = gaussian_filter(
            folds[reference], config.fixed_smooth_width, axis=1,
            stimulus_positions=centers, full_stimulus_positions=full_centers,
        )
        original = CVPCA(center=center, on_stimuli=True).fit(smooth)
        np.testing.assert_allclose(fit.basis, original.pca.get_components().numpy())
        first, second = sorted(fit.projections)
        for fold, projected in fit.projections.items():
            np.testing.assert_allclose(projected, original.pca.transform(folds[fold].T).T.numpy())
        actual = vector_correlation(
            torch.from_numpy(fit.projections[first]), torch.from_numpy(fit.projections[second]),
            covariance=True, dim=0, center=center,
        )
        torch.testing.assert_close(actual, original.score(folds[first], folds[second]))


@pytest.fixture
def viewer_setup(monkeypatch):
    folds = _folds()
    prep_ss = Mock(return_value={"train": folds, "test": folds})
    prep_fixed = Mock(return_value={
        "torch_pfs": folds[:3], "dist_centers": np.arange(9, dtype=float),
        "full_dist_centers": np.arange(9, dtype=float),
    })
    monkeypatch.setattr(StimSpaceSpectraConfig, "process", prep_ss)
    monkeypatch.setattr(CVPCAConfig, "process", prep_fixed)
    monkeypatch.setattr(module, "PopulationRegistry", Mock())
    original_fit = module._fit_projections
    fit_spy = Mock(wraps=original_fit)
    monkeypatch.setattr(module, "_fit_projections", fit_spy)
    sessions = [
        SimpleNamespace(mouse_name="mouse-a", session_uid="a-1", environments=[2, 7]),
        SimpleNamespace(mouse_name="mouse-a", session_uid="a-2", environments=[2, 7]),
        SimpleNamespace(mouse_name="mouse-b", session_uid="b-1", environments=[9]),
    ]
    viewer = PlacefieldProjectionViewer(sessions, projection_dim=4)
    return viewer, prep_ss, prep_fixed, fit_spy


def test_viewer_caches_and_draws_each_heldout_pair_separately(viewer_setup):
    viewer, prep_ss, prep_fixed, fit_spy = viewer_setup
    first_fit = viewer._get_fit(viewer.state)
    fig = viewer.plot(viewer.state)
    scatters = fig.axes[0].collections
    assert len(scatters) == 2
    for scatter, fold in zip(scatters, [2, 3]):
        np.testing.assert_allclose(scatter.get_offsets()[:, 0], first_fit.projections[1][:, 4])
        np.testing.assert_allclose(scatter.get_offsets()[:, 1], first_fit.projections[fold][:, 4])
    assert not np.array_equal(scatters[0].get_facecolors(), scatters[1].get_facecolors())
    viewer.set_parameter_value("projection_dim", 5)
    viewer.plot(viewer.state)
    assert fit_spy.call_count == prep_ss.call_count == 1
    viewer.set_parameter_value("reference_fold", 3)
    assert fit_spy.call_count == 2
    assert prep_ss.call_count == 1
    viewer.set_parameter_value("reference_fold", 0)
    assert viewer._get_fit(viewer.state) is first_fit
    assert fit_spy.call_count == 2
    viewer.set_parameter_value("dim method", "reg_covariances_fixed")
    assert prep_fixed.call_count == 1
    assert len(viewer.plot(viewer.state).axes[0].collections) == 1
    viewer.set_parameter_value("dim method", "ss_cvpca")
    assert viewer._get_fit(viewer.state) is first_fit
    viewer.set_parameter_value("example_session", "a-2")
    assert prep_ss.call_count == 2
    viewer.set_parameter_value("mouse", "mouse-b")
    assert viewer.state["example_session"] == "b-1"
    assert prep_ss.call_count == 3
    viewer.clear_cache()
    assert not viewer.projection_cache
    viewer.plot(viewer.state)
    assert prep_ss.call_count == 4


def test_reference_and_dimension_bounds_follow_method(viewer_setup):
    viewer, _, _, _ = viewer_setup
    assert viewer.parameters["projection_dim"].max == 8
    viewer.set_parameter_value("reference_fold", 3)
    viewer.set_parameter_value("dim method", "reg_covariances_fixed")
    assert viewer.state["reference_fold"] == 2
    assert set(viewer._get_fit(viewer.state).projections) == {0, 1}


def test_zero_variance_and_tiny_correlations():
    assert np.isnan(module._pearson_r(np.ones(3), np.arange(3)))
    assert np.isnan(module._pearson_r(np.array([np.nan, 1]), np.array([2, 3])))
    x = np.array([-1., -1., 1., 1.])
    y = np.array([-1., 1., -1., 1.]) + 1e-4 * x
    assert module._pearson_r(x, y) == pytest.approx(1e-4, abs=1e-11)


def test_viewer_rejects_neural_basis():
    sessions = [SimpleNamespace(mouse_name="m", session_uid="s")]
    with pytest.raises(ValueError, match="spatial"):
        PlacefieldProjectionViewer(sessions, cvpca_config=CVPCAConfig(use_spatial_eigenvectors=False))


def test_ss_cv_uv_scatter_reproduces_original_score_for_every_fold_assignment():
    folds = _folds()
    # Deliberately different fitting/test smoothing to catch use of the wrong fold version.
    prepared = {"train": [pf * 0.8 + 0.3 for pf in folds], "test": folds}
    for reference in range(4):
        for partner in range(4):
            if reference == partner:
                continue
            fit = module._fit_projections("ss_cv", StimSpaceSpectraConfig(), prepared, reference, partner)
            expected_folds = sorted(set(range(4)) - {reference, partner})
            assert sorted(fit.projections) == expected_folds
            train = _to_g(prepared["train"][reference])
            v_data = _to_g(prepared["test"][partner])
            u, _, vt = np.linalg.svd((train.T @ v_data).numpy(), full_matrices=False)
            np.testing.assert_allclose(fit.basis, u)
            np.testing.assert_allclose(fit.right_basis, vt.T)
            for fold, pair in zip(expected_folds, fit.pairs):
                u_data = _to_g(prepared["test"][fold])
                np.testing.assert_allclose(pair.x, u_data.numpy() @ u)
                np.testing.assert_allclose(pair.y, v_data.numpy() @ vt.T)
                # ss_cv is an uncentered cross-product across neurons, not a sample covariance.
                np.testing.assert_allclose(
                    np.sum(pair.x * pair.y, axis=0), _cvsvd(train, u_data, v_data).numpy(), atol=1e-14,
                )
                assert "used in fit" in pair.ylabel


def _svca_folds():
    rng = torch.Generator().manual_seed(93)
    train, test = [], []
    for fold in range(2):
        # Unequal neuron groups; train/test can have different counts of valid spatial bins.
        train.append(tuple(torch.randn(n, 12 - fold, generator=rng, dtype=torch.float64) for n in (18, 23)))
        test.append(tuple(torch.randn(n, 10 - fold, generator=rng, dtype=torch.float64) + 8 for n in (18, 23)))
    return {"train": train, "test": test, "num_components": 7}


@pytest.mark.parametrize("centered", [True, False])
def test_svca_uv_scatter_matches_original_score_and_transform(monkeypatch, centered):
    prepared = _svca_folds()
    config = StimspaceSVCAConfig(centered=centered)
    for reference in (0, 1):
        model = module._fit_svca(*prepared["train"][reference], centered, prepared["num_components"])
        fit_spy = Mock(return_value=model)
        with monkeypatch.context() as patcher:
            patcher.setattr(module, "_fit_svca", fit_spy)
            fit = module._fit_projections("SVCA", config, prepared, reference)
        assert fit_spy.call_count == 1
        assert sorted(fit.projections) == [1 - reference]
        assert len(fit.pairs) == 1
        source, target = prepared["test"][1 - reference]
        sc = source - source.mean(dim=1, keepdim=True) if centered else source
        tc = target - target.mean(dim=1, keepdim=True) if centered else target
        x = (model.U.T @ sc).numpy().T
        y = (model.V.T @ tc).numpy().T
        pair = fit.pairs[0]
        np.testing.assert_allclose(pair.x, x)
        np.testing.assert_allclose(pair.y, y)
        np.testing.assert_allclose(fit.right_basis, model.V.numpy())
        expected_score = np.mean(x * y, axis=0)
        np.testing.assert_allclose(expected_score, model.score(source, target)[0].numpy())
        assert "Source" in pair.xlabel and "Target" in pair.ylabel
        assert "spatial bins" in fit.description


def test_ss_cv_shares_preparation_and_caches_partner_fits(viewer_setup):
    viewer, prep_ss, _, fit_spy = viewer_setup
    viewer.set_parameter_value("dim method", "ss_cv")
    fit = viewer._get_fit(viewer.state)
    assert prep_ss.call_count == 1  # same original PF preparation as ss_cvpca
    assert fit_spy.call_count == 2
    assert len(viewer.plot(viewer.state).axes[0].collections) == 2
    viewer.set_parameter_value("projection_dim", 3)
    viewer.plot(viewer.state)
    assert fit_spy.call_count == 2
    viewer.set_parameter_value("partner_fold", 3)
    assert fit_spy.call_count == 3
    assert prep_ss.call_count == 1
    viewer.set_parameter_value("partner_fold", 2)
    assert viewer._get_fit(viewer.state) is fit
    viewer.set_parameter_value("reference_fold", 2)
    assert viewer.state["partner_fold"] != viewer.state["reference_fold"]
    assert len(viewer._get_fit(viewer.state).pairs) == 2


def test_svca_viewer_fold_bounds_and_scatter(viewer_setup, monkeypatch):
    viewer, _, _, fit_spy = viewer_setup
    prepared = _svca_folds()
    prepare = Mock(return_value=prepared)
    monkeypatch.setattr(StimspaceSVCAConfig, "process", prepare)
    viewer.set_parameter_value("reference_fold", 3)
    viewer.set_parameter_value("projection_dim", 8)
    viewer.set_parameter_value("dim method", "SVCA")
    assert viewer.state["reference_fold"] == 1
    assert viewer.parameters["reference_fold"].max == 1
    assert viewer.state["projection_dim"] == 6
    fit = viewer._get_fit(viewer.state)
    scatter = viewer.plot(viewer.state).axes[0].collections[0]
    np.testing.assert_allclose(scatter.get_offsets(), np.column_stack((fit.pairs[0].x[:, 6], fit.pairs[0].y[:, 6])))
    assert len(scatter.get_offsets()) == prepared["test"][0][0].shape[1]
    calls = fit_spy.call_count
    viewer.set_parameter_value("partner_fold", 0)
    viewer.plot(viewer.state)
    assert fit_spy.call_count == calls  # partner is only meaningful for ss_cv
    assert prepare.call_count == 1
    viewer.set_parameter_value("reference_fold", 0)
    viewer.set_parameter_value("reference_fold", 1)
    assert viewer._get_fit(viewer.state) is fit
    assert prepare.call_count == 1


@pytest.mark.parametrize("method", ["ss_cvpca", "ss_cv", "reg_covariances_fixed"])
def test_roi_projection_position_panel_explains_missing_spatial_axis(viewer_setup, method):
    viewer, _, _, _ = viewer_setup
    viewer.set_parameter_value("dim method", method)
    figure = viewer.plot(viewer.state)
    assert len(figure.axes) == 2
    position_ax = figure.axes[1]
    assert "one value per ROI" in position_ax.texts[0].get_text()
    assert not position_ax.lines


def test_svca_position_panel_retains_environments_and_missing_bin_gaps(viewer_setup, monkeypatch):
    viewer, _, _, fit_spy = viewer_setup
    rng = np.random.default_rng(5)

    def placefield(nroi):
        return Placefield(
            placefield=rng.normal(size=(2, 5, nroi)), dist_edges=np.arange(6) * 10.,
            environment=np.array([2, 7]), count=np.ones((2, 5)),
        )

    source, target = placefield(3), placefield(4)
    source.placefield[0, 1, 0] = np.nan
    target.placefield[1, 2, 0] = np.nan
    prepared = _svca_placefield_folds([source] * 2, [target] * 2, [source] * 2, [target] * 2, 3, 4, 3)
    meta = prepared["test_positions"][1]
    np.testing.assert_array_equal(meta["position"], [5, 25, 35, 45, 5, 15, 35, 45])
    np.testing.assert_array_equal(meta["environment"], [2, 2, 2, 2, 7, 7, 7, 7])
    np.testing.assert_array_equal(meta["bin_index"], [0, 2, 3, 4, 0, 1, 3, 4])
    monkeypatch.setattr(StimspaceSVCAConfig, "process", Mock(return_value=prepared))
    viewer.set_parameter_value("dim method", "SVCA")
    viewer.set_parameter_value("projection_dim", 1)
    fit = viewer._get_fit(viewer.state)
    figure = viewer.plot(viewer.state)
    ax = figure.axes[1]
    assert ax.get_xlabel() == "Position (cm)"
    curves = [line for line in ax.lines if "Fold" in line.get_label()]
    assert len(curves) == 4  # U and V separately in each environment
    for line, env, scores in zip(curves, [2, 2, 7, 7], [fit.pairs[0].x, fit.pairs[0].y] * 2):
        x, y = line.get_xdata(), line.get_ydata()
        valid = np.isfinite(x)
        selected = meta["environment"] == env
        np.testing.assert_array_equal(x[valid], meta["position"][selected])
        np.testing.assert_allclose(y[valid], scores[selected, 1])
        assert np.isnan(x).sum() == 1
        assert f"env {env}" in line.get_label()
    calls = fit_spy.call_count
    viewer.set_parameter_value("projection_dim", 0)
    viewer.plot(viewer.state)
    assert fit_spy.call_count == calls


def test_environment_selection_refits_and_reuses_each_environment_cache(viewer_setup):
    viewer, prep_ss, prep_fixed, fit_spy = viewer_setup
    original = viewer._get_fit(viewer.state)
    assert viewer.state["environment"] == 2
    assert prep_ss.call_args.kwargs["environment"] == 2
    viewer.set_parameter_value("environment", 7)
    assert prep_ss.call_args.kwargs["environment"] == 7
    assert prep_ss.call_count == fit_spy.call_count == 2
    assert viewer._get_fit(viewer.state) is not original
    viewer.set_parameter_value("environment", 2)
    assert viewer._get_fit(viewer.state) is original
    assert prep_ss.call_count == 2
    viewer.set_parameter_value("dim method", "reg_covariances_fixed")
    assert prep_fixed.call_args.kwargs["environment"] == 2
    viewer.set_parameter_value("environment", 7)
    assert prep_fixed.call_args.kwargs["environment"] == 7
    viewer.set_parameter_value("mouse", "mouse-b")
    assert viewer.state["environment"] == 9
    assert list(viewer.parameters["environment"].options) == [9]
    assert prep_fixed.call_args.kwargs["environment"] == 9
