"""Read-only audit of the example viewer's first stored session."""
import numpy as np
import torch
from vrAnalysis.database import get_database
from dimensionality_manuscript import ResultsAggregator, ResultsStore, StimSpaceSpectraConfig, TilburyFitConfig, get_data_config
from dimensionality_manuscript.registry import PopulationRegistry
from dimensionality_manuscript.figure_scripts.figure4 import PlacefieldSpectrumExampleViewer
from dimensionality_manuscript.figure_scripts.figure4.placefield_fits import _load_fit_curves
from dimensionality_manuscript.figure_scripts.figure4.placefield_spectrum_example import _rescaled_model_spectra
from dimensionality_manuscript.configs.stimspace import _cvpca_svd, _cvpca_spatial_basis, _cvpca_project
from dimensionality_manuscript.configs.tilbury_fit import _eval_tilbury

print("Loading stored result metadata", flush=True)
sessions = list(get_database("vrSessions").iter_sessions(imaging=True, experimentType="Blender VR"))
store = ResultsStore()
fits = ResultsAggregator(TilburyFitConfig(), store, sessions)
spectra = ResultsAggregator(StimSpaceSpectraConfig(), store, sessions)
registry = PopulationRegistry(registry_params=get_data_config("even").to_registry_params(), autosave=False)
print("First fit session:", fits.session_ids[0], flush=True)
viewer = PlacefieldSpectrumExampleViewer(fits, results_spectra=spectra, registry=registry,
    smooth_widths=(5., None), reliability_fraction_active_thresholds=(.3, .1))
sid = viewer._session_ids[viewer.state["session_id"]]
bundle = _load_fit_curves(fits, registry, sid, viewer._fit_sel_params(viewer.state),
                          extra_keys=("r2_test_shrinkage",), target_full_session=True)
valid = np.logical_and.reduce([np.isfinite(bundle[key]).all(axis=1) for key in ("params", "params_control", "params_shrinkage")])
scale = bundle["fluorescence_scale"][valid, None]
curves = {s: torch.tensor(bundle[s + "_curve"][valid] * scale, dtype=torch.float64) for s in ("train", "validation", "test")}
basis = _cvpca_spatial_basis(curves["train"])
model = np.array([_eval_tilbury(bundle["theta"], p) for p in bundle["params_shrinkage"][valid]]) * scale
projection = _cvpca_project(torch.tensor(model), basis).numpy()
scores = (projection - projection.mean(axis=0)) ** 2
model_in_data_basis = scores.sum(axis=0) / (scores.shape[0] - 1)
aligned_cv = _cvpca_svd(curves["train"], curves["validation"], curves["test"]).numpy()
model_own = _rescaled_model_spectra(bundle, "neurons")["eig_shrinkage"]
data = viewer._session_rows(viewer._spectra["data"], sid)[0]
print("N fitted:", valid.sum(), "P:", len(bundle["theta"]), "best env:", bundle["best_env"], flush=True)
for label, row in (("stored data", data), ("same-neuron/env cv data", aligned_cv), ("model own basis", model_own), ("model data basis", model_in_data_basis)):
    print(label, "first 8:", np.round(row[:8], 5), "signed sum:", np.nansum(row), flush=True)
for split, matrix in curves.items():
    centered = matrix.numpy() - matrix.numpy().mean(axis=0, keepdims=True)
    eig = np.linalg.svd(centered, compute_uv=False) ** 2 / (matrix.shape[0] - 1)
    print(split, "raw covariance first8:", np.round(eig[:8], 5), "trace:", eig.sum(), flush=True)
from dimensionality_manuscript.configs.tilbury_fit import _r2
raw_model = np.array([_eval_tilbury(bundle["theta"], p) for p in bundle["params"][valid]])
rebuilt_r2 = np.array([_r2(m, y) for m, y in zip(raw_model, bundle["test_curve"][valid])])
saved_r2 = bundle["r2_test"][valid]
print("Rebuilt vs stored R2 max diff:", np.nanmax(np.abs(rebuilt_r2-saved_r2)), "medians:", np.nanmedian(rebuilt_r2), np.nanmedian(saved_r2), flush=True)
test_std = curves['test'].numpy().std(axis=1)
print("median model/test std ratio:", np.median(model.std(axis=1)[test_std > 0] / test_std[test_std > 0]), flush=True)
shrink_r2 = np.array([_r2(m, y) for m, y in zip(model, curves['test'].numpy())])
print("Shrinkage median test R2:", np.nanmedian(shrink_r2),
      "stored R2 max difference:", np.nanmax(np.abs(shrink_r2 - bundle['r2_test_shrinkage'][valid])), flush=True)
print("scale quantiles:", np.quantile(scale, [0,.25,.5,.75,1]), flush=True)
