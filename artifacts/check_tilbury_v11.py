"""Short real-session integration check; never writes scientific results to the real store."""
from pathlib import Path
import json
import tempfile
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

from vrAnalysis.database import get_database
from dimensionality_manuscript import TilburyFitConfig, ResultsStore, ResultsAggregator, get_data_config
from dimensionality_manuscript.registry import PopulationRegistry
from dimensionality_manuscript.configs.tilbury_fit import _spatial_curve_spectrum
from dimensionality_manuscript.configs.stimspace import _cvpca_svd
from dimensionality_manuscript.figure_scripts.figure4 import PlacefieldSpectrumExampleViewer, AdaptiveAlphaConfig


def main():
    torch.set_num_threads(4)
    threadpool_limits(limits=4)
    sid="ATL057.2024-07-08.701"
    session=next(s for s in get_database("vrSessions").iter_sessions(imaging=True,experimentType="Blender VR")
                 if s.session_uid==sid)
    cfg=TilburyFitConfig(activity_parameters_name="std")
    registry=PopulationRegistry(registry_params=get_data_config("even").to_registry_params(),autosave=False)
    print("Running TWO-STEP integration fit, not a scientific result:",sid,flush=True)
    result=cfg.process(session,registry,device="cpu",gd_num_steps=2,gd_learning_rate=.001)
    meta=result["spectrum_metadata"]
    assert meta["activity_scaling_scope"]=="full_recording"
    assert meta["n_neurons"]==len(result["spectrum_roi_indices"])
    assert meta["spatial"]["denominator"]==meta["n_neurons"]-1
    assert result["ss_cvpca_matched"].shape==result["eig_spatial_control"].shape
    assert np.isfinite(result["ss_cvpca_matched"]).all()
    with tempfile.TemporaryDirectory(prefix="tilbury-v11-check-") as tmp:
        store=ResultsStore(Path(tmp)/"check.sqlite")
        store.put(sid,cfg,result)
        agg=ResultsAggregator(cfg,store,[session])
        viewer=PlacefieldSpectrumExampleViewer(agg,fit_zone="fixed",fixed_range=(5,35),
                     source_cfg=AdaptiveAlphaConfig("none",0.,1,2,5))
        assert viewer.state["activity_parameters_name"]=="std"
        assert viewer.state["source_key"]=="matched_ss_cvpca"
        np.testing.assert_allclose(viewer._spectra["data"][0][0],result["ss_cvpca_matched"])
        np.testing.assert_allclose(viewer._spectra["eig_shrinkage"][0][0],result["eig_spatial_shrinkage"])
        assert viewer.comparison_metadata==meta
        fig=viewer.plot(viewer.state)
        fig.suptitle("Integration check only: two optimizer steps",fontsize=8)
        fig.canvas.draw()
        fig.savefig("artifacts/tilbury_v11_validation_preview.png",dpi=160)
        plt.close(fig)
    prepared=Path("artifacts/s1_spectrum_validation/prepared.npz")
    if prepared.exists():
        matrix=np.load(prepared)["signal"]
        tensor=torch.tensor(matrix)
        oracle=_cvpca_svd(tensor,tensor,tensor).numpy()
        calculated=_spatial_curve_spectrum(matrix)
        np.testing.assert_allclose(calculated,oracle,rtol=1e-8,atol=1e-12)
        print("S1 prepared-session no-noise scale check passed",flush=True)
    summary={"session":sid,"schema":cfg.schema_version,"optimizer_steps":2,"scientific_result":False,
             "temporary_store_roundtrip":"passed","stored_viewer_roundtrip":"passed","metadata":meta}
    Path("artifacts/tilbury_v11_validation.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
