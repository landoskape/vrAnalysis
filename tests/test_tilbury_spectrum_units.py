"""Units, support, metadata, and direct stored-spectrum checks for Tilbury v11."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from dimensionality_manuscript.registry import TimeSplit
from dimilibi import Population
from dimilibi.pca import PCA
from dimensionality_manuscript.configs.tilbury_fit import TilburyFitConfig, _spatial_curve_spectrum, _curve_spectrum
from dimensionality_manuscript.configs.stimspace import _cvpca_svd
from dimensionality_manuscript.figure_scripts.figure4 import PlacefieldSpectrumExampleViewer, AdaptiveAlphaConfig


@pytest.mark.parametrize("scope", ["full_recording", "retained_vr"])
def test_scaling_scope_uses_recorded_frame_population(scope):
    raw = torch.tensor([[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,100.],
                        [2.,4.,6.,8.,10.,12.,14.,16.,18.,20.,22.,24.,0.]])
    pop = Population(raw, idx_neurons=np.arange(2), idx_samples=np.arange(12), generate_splits=False)
    pop.time_split_indices = [torch.arange(i*3,(i+1)*3) for i in range(4)]
    registry = SimpleNamespace(get_population=lambda *args: (pop, SimpleNamespace(filter=lambda x: x)),
                               time_split=TimeSplit())
    cfg = TilburyFitConfig(activity_parameters_name="std", activity_scaling_scope=scope)
    split, _ = cfg._get_split_data(None, registry)
    divisor = (raw if scope == "full_recording" else raw[:,:12]).std(dim=1,keepdim=True)
    for name in ("train", "validation", "test"):
        indices = registry.time_split[name]
        times = pop.get_split_times(indices, within_idx_samples=False)
        np.testing.assert_allclose(split[name], (raw[:,times]/divisor).T, rtol=1e-6)
    assert cfg.activity_scaling_scope == scope


def test_noise_free_spatial_spectrum_matches_actual_estimator_and_covariance():
    rng = np.random.default_rng(73)
    matrix = rng.normal(size=(17,7)) * np.arange(1,18)[:,None]
    tensor = torch.tensor(matrix)
    np.testing.assert_allclose(_spatial_curve_spectrum(matrix), _cvpca_svd(tensor,tensor,tensor), atol=1e-12)
    expected = np.linalg.eigvalsh(np.cov(matrix, rowvar=False))[::-1]
    np.testing.assert_allclose(_spatial_curve_spectrum(matrix), expected, atol=1e-12)
    # Preserve the native historical denominator P rather than quietly changing eig_*.
    np.testing.assert_allclose(_curve_spectrum(matrix), PCA(center=True).fit(tensor).get_eigenvalues(),rtol=1e-5,atol=1e-10)
    assert not np.allclose(_curve_spectrum(matrix), expected)


def test_matched_spectra_intersect_fold_support_and_original_rois(monkeypatch):
    cfg = TilburyFitConfig(num_bins=4)
    matrix = np.array([[0.,1.,2.,0.], [1.,0.,3.,1.], [2.,2.,0.,0.]])
    population = SimpleNamespace(idx_neurons=np.array([4,8,12]), time_split_indices=[0,1,2,3],
                                 data=np.zeros((13,50)), idx_samples=np.arange(40))
    registry = SimpleNamespace(get_population=lambda *args:(population,None),
                               time_split={"train":[0,1],"validation":[2],"test":[3]})
    monkeypatch.setattr(TilburyFitConfig,"_get_split_data",
                        lambda *args,**kwargs: ({i:matrix.T for i in range(4)},{i:i for i in range(4)}))
    widths=[]
    def average(self, activity, behavior, *args, smooth_width=None):
        widths.append(smooth_width)
        count=np.ones(4)
        if behavior==1:
            count[1]=0
        return activity.T,count
    monkeypatch.setattr(TilburyFitConfig,"_avg_placefield",average)
    keep=np.array([True,False,True])
    chosen=matrix[keep]
    theta=np.array([25.,75.,125.,175.])
    curves={s:chosen for s in ("train","validation","test")}
    models={f"eig_{name}":chosen for name in ("control","tilbury","shrinkage","better")}
    result=cfg.matched_spectra(SimpleNamespace(env_length=[200]),registry,theta,curves,keep,models,1)
    expected=_spatial_curve_spectrum(chosen[:,[0,2,3]])
    np.testing.assert_allclose(result["ss_cvpca_matched"],expected,atol=1e-12)
    for name in models:
        np.testing.assert_allclose(result[name.replace("eig_","eig_spatial_")],expected)
    np.testing.assert_array_equal(result["spectrum_roi_indices"],[4,12])
    np.testing.assert_array_equal(result["spectrum_dist_centers"],theta[[0,2,3]])
    meta=result["spectrum_metadata"]
    assert meta["native"]["denominator"]==4
    assert meta["spatial"]["denominator"]==1
    assert meta["spatial"]["n_positions"]==3
    assert meta["activity_scaling_scope"]=="full_recording"
    assert meta["scaling_n_frames"]==50
    assert meta["matched_data"]["assignments"]==[[0,1,2],[0,1,3],[0,2,3],[1,2,3]]
    assert widths==[None,5.]*4
    json.dumps(meta)


class Stored:
    param_axes={}
    config_class=TilburyFitConfig()
    session_ids=["example"]
    mouse_names=np.array(["mouse"])
    def __init__(self):
        spectrum=np.arange(1,21,dtype=float)**-2
        self.arrays={"ss_cvpca_matched":spectrum[None],
                     "eig_spatial_raw_train":2*spectrum[None],
                     "eig_spatial_raw_test":3*spectrum[None]}
        self.arrays.update({f"eig_spatial_{key}":spectrum[None]
                           for key in ("control","tilbury","shrinkage","better")})
    def sel(self,*,keys,**kwargs):
        return {key:self.arrays[key] for key in keys}
    def sel_objects(self,**kwargs):
        return {"spectrum_metadata":[{"activity_scaling_scope":"full_recording"}]}


@pytest.mark.parametrize("source", [None,"matched_pca_train","matched_pca_test"])
def test_v11_viewer_reads_stored_comparison_without_activity_reload(monkeypatch,source):
    from dimensionality_manuscript.figure_scripts.figure4 import placefield_fits
    monkeypatch.setattr(placefield_fits,"_load_fit_curves",lambda *a,**k:pytest.fail("Should use stored spectra"))
    stored=Stored()
    viewer=PlacefieldSpectrumExampleViewer(stored,source_key=source,
        source_cfg=AdaptiveAlphaConfig("none",0.,1,2,3),fit_zone="fixed",fixed_range=(1,10))
    assert viewer.state["source_key"]==(source or "matched_ss_cvpca")
    key={None:"ss_cvpca_matched","matched_pca_train":"eig_spatial_raw_train",
         "matched_pca_test":"eig_spatial_raw_test"}[source]
    np.testing.assert_array_equal(viewer._spectra["data"][0],stored.arrays[key])
    np.testing.assert_array_equal(viewer._spectra["eig_shrinkage"][0],stored.arrays["eig_spatial_shrinkage"])
    assert viewer.comparison_metadata=={"activity_scaling_scope":"full_recording"}
    fig=viewer.plot(viewer.state)
    assert any(label.startswith("Matched data:") for label in fig.axes[0].get_legend_handles_labels()[1])
    import matplotlib.pyplot as plt
    plt.close(fig)
