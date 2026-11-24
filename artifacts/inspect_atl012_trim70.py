"""Recompute ATL012 March 1 reliability after removing chronological trials 199-268.

Uses the existing SpkmapProcessor and reliability_loo implementations, with all
processor cache reads/writes disabled. Session files and result stores are read-only.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
import json

import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vrAnalysis.sessions import B2Session
from vrAnalysis.processors.spkmaps import SpkmapProcessor, SpkmapParams
from vrAnalysis.processors.placefields import get_frame_behavior, get_placefield
from vrAnalysis.processors.support import median_zscore
from vrAnalysis.helpers import reliability_loo
from vrAnalysis.metrics import FractionActive
from dimensionality_manuscript import PFPredQualityConfig, ResultsAggregator
from inspect_atl012_reliability import ReadOnlyStore


class ROIChunkSession(B2Session):
    """In-memory view exposing only a chunk of original ROI activity."""
    def __init__(self, source, activity):
        self.__dict__.update(source.__dict__)
        self.values = SimpleNamespace(**vars(source.values))
        self.values.numROIs = activity.shape[1]
        self.one_cache = {}
        self._chunk_activity = activity

    @property
    def spks(self):
        return self._chunk_activity


class UncachedProcessor(SpkmapProcessor):
    def load_from_cache(self, data_type):
        return None, False

    def save_cache(self, data_type, data):
        raise AssertionError("Diagnostic must never write processor caches")


def run():
    output = Path(__file__).parent / "atl012_trim70_inspection"
    output.mkdir(exist_ok=True)
    session = B2Session.create("ATL012", "2023-03-01", "701", params={"spks_type": PFPredQualityConfig().spks_type})
    selected = session.idx_rois.copy()
    original_indices = np.flatnonzero(selected)
    classes = session.roicat_classifier["class_predictions"][selected]
    planes = session.get_plane_idx()[selected]
    trial_env = session.trial_environment.copy()
    cutoff = session.num_trials - 70
    assert cutoff == 198 and len(trial_env) == 268
    print("TRIAL COUNTS", {condition:{int(e):int(np.sum(env==e)) for e in np.unique(env)}
                           for condition,env in (("full",trial_env),("trim70",trial_env[:cutoff]),("removed",trial_env[cutoff:]))},flush=True)
    behavior = get_frame_behavior(session)
    stored = ResultsAggregator(PFPredQualityConfig, ReadOnlyStore(), [session],
                               keys=["reliability_slot", "fraction_active_slot"]).sel(squeeze_ones=False)
    cached_rel = stored["reliability_slot"][0,:,:len(original_indices)]
    cached_fa = stored["fraction_active_slot"][0,:,:len(original_indices)]
    assert cached_rel.shape == (3,len(original_indices))
    session.clear_cache()
    activity = sparse.load_npz(session.one_path / "mpci.roiSignificantFluorescenceRebase.npz").tocsc()
    assert activity.shape == (session.values.numFrames, session.values.numROIs)
    values = {condition:{metric:np.full((3,len(original_indices)),np.nan) for metric in ("reliability","fraction_active")}
              for condition in ("full","trim70")}
    reuse = "--reuse-computed" in sys.argv
    if reuse:
        with np.load(output/"recomputed_metrics.npz") as saved:
            assert np.array_equal(saved["original_roi"],original_indices)
            for condition in values:
                for metric in values[condition]:
                    values[condition][metric] = saved[f"{condition}_{metric}"].copy()
    coverage = []
    for start in ([] if reuse else range(0,len(original_indices),256)):
        stop = min(start+256,len(original_indices))
        chunk_spks = activity[:,original_indices[start:stop]].toarray()
        chunk = ROIChunkSession(session,chunk_spks)
        processor = UncachedProcessor(chunk,SpkmapParams(autosave=False))
        maps = processor.get_processed_maps(force_recompute=True)
        required = processor._idx_required_position_bins()
        full_trial = np.all(~np.isnan(maps.occmap[:,required]),axis=1)
        for condition in values:
            keep = full_trial & ((np.arange(session.num_trials)<cutoff) if condition=="trim70" else True)
            idx_by_env = [np.flatnonzero(keep & (trial_env==env)) for env in (1,3,4)]
            valid_positions = ~np.any(np.stack([np.any(np.isnan(maps.occmap[idx]),axis=0) for idx in idx_by_env]),axis=0)
            for slot,(env,idx) in enumerate(zip((1,3,4),idx_by_env)):
                spkmap = maps.spkmap[:,idx][:,:,valid_positions]
                assert np.isfinite(spkmap).all()
                values[condition]["reliability"][slot,start:stop] = reliability_loo(spkmap)
                if start == 0:
                    coverage.append(dict(condition=condition,environment=env,raw_trials=int(np.sum(trial_env[:cutoff] == env) if condition=="trim70" else np.sum(trial_env==env)),
                                         retained_full_trials=len(idx),position_bins=int(valid_positions.sum())))
        if start == 0:
            official = processor.get_reliability(use_session_filters=False,force_recompute=True).values
            print("OFFICIAL CHECK",np.max(np.abs(official-values["full"]["reliability"][:,:stop])),flush=True)
            print("BASELINE SAMPLE",cached_rel[:,:8],"RECOMPUTED",values["full"]["reliability"][:,:8],flush=True)
        del maps
        # Fraction-active uses the pipeline's imaging-frame trial maps; these are
        # intentionally distinct from reliability's full-trial-filtered maps.
        standardized = median_zscore(chunk_spks,median_subtract=not session.zero_baseline_spks)
        trial_maps = get_placefield(standardized,behavior,processor.dist_edges,processor.params.speed_threshold,
                                   average=False,smooth_width=processor.params.smooth_width)
        for slot,env in enumerate((1,3,4)):
            env_maps = trial_maps.filter_by_environment(env)
            for condition in values:
                keep = (env_maps.trials < cutoff) if condition=="trim70" else np.ones(len(env_maps.trials),dtype=bool)
                spkmap = np.transpose(env_maps.placefield[keep],(2,0,1))
                values[condition]["fraction_active"][slot,start:stop] = FractionActive.compute(
                    spkmap,activity_axis=2,fraction_axis=1,activity_method="rms",fraction_method="participation")
        print("COMPUTED",stop,"/",len(original_indices),flush=True)
        if start == 0:
            print("FIRST CHUNK ERRORS",{key:float(np.nanmax(np.abs(values["full"][key][:,:stop]-expected[:,:stop]))) for key,expected in (("reliability",cached_rel),("fraction_active",cached_fa))},flush=True)
    np.savez_compressed(output/"recomputed_metrics.npz",original_roi=original_indices,classes=classes,planes=planes,
                        full_reliability=values["full"]["reliability"],trim70_reliability=values["trim70"]["reliability"],
                        full_fraction_active=values["full"]["fraction_active"],trim70_fraction_active=values["trim70"]["fraction_active"])
    if reuse:
        # Occupancy and trial eligibility are independent of neural activity.
        processor = UncachedProcessor(ROIChunkSession(session,np.zeros((session.values.numFrames,1),dtype=np.float32)),SpkmapParams(autosave=False))
        maps = processor.get_processed_maps(force_recompute=True)
        required = processor._idx_required_position_bins()
        full_trial = np.all(~np.isnan(maps.occmap[:,required]),axis=1)
        for condition in values:
            keep = full_trial & ((np.arange(session.num_trials)<cutoff) if condition=="trim70" else True)
            idx_by_env = [np.flatnonzero(keep & (trial_env==env)) for env in (1,3,4)]
            valid_positions = ~np.any(np.stack([np.any(np.isnan(maps.occmap[idx]),axis=0) for idx in idx_by_env]),axis=0)
            for env,idx in zip((1,3,4),idx_by_env):
                coverage.append(dict(condition=condition,environment=env,raw_trials=int(np.sum(trial_env[:cutoff] == env) if condition=="trim70" else np.sum(trial_env==env)),
                                     retained_full_trials=len(idx),position_bins=int(valid_positions.sum())))
        del maps
    validation = {}
    for metric,expected in (("reliability",cached_rel),("fraction_active",cached_fa)):
        actual=values["full"][metric]
        validation[metric+"_max_abs_error"] = float(np.nanmax(np.abs(actual-expected)))
        validation[metric+"_nan_masks_match"] = bool(np.array_equal(np.isnan(actual),np.isnan(expected)))
        # Low-activity reliability can be numerically unstable when a spatial
        # profile is almost constant. Validate every activity-eligible cell,
        # not only cells that also passed the reliability cutoff.
        eligible = cached_fa > .1
        validation[metric+"_eligible_max_abs_error"] = float(np.nanmax(np.abs(actual[eligible]-expected[eligible])))
        validation[metric+"_low_activity_discrepancy_count"] = int(np.sum((np.abs(actual-expected)>2e-5)&~eligible))
        compare = eligible if metric=="reliability" else np.ones(actual.shape,dtype=bool)
        assert np.allclose(actual[compare],expected[compare],atol=2e-5,rtol=2e-5,equal_nan=True), (metric,validation)
    expected_quality = (cached_rel>.3)&(cached_fa>.1)
    actual_quality = (values["full"]["reliability"]>.3)&(values["full"]["fraction_active"]>.1)
    assert np.array_equal(actual_quality,expected_quality)
    validation["quality_membership_matches"] = True
    print("BASELINE VALIDATION",validation,flush=True)
    rows,cell_rows = [],[]
    for slot,env in enumerate((1,3,4)):
        full_quality = (values["full"]["reliability"][slot]>.3) & (values["full"]["fraction_active"][slot]>.1)
        for condition in values:
            rel=values[condition]["reliability"][slot]
            fa=values[condition]["fraction_active"][slot]
            new_quality = (rel>.3) & (fa>.1)
            groups={"all_selected":np.ones(len(rel),dtype=bool),"somatic":classes==0,"dendritic":classes==3}
            groups.update({f"plane_{p}":planes==p for p in np.unique(planes)})
            for group,group_mask in groups.items():
                for subset,mask in (("all",group_mask),("quality_reselected",group_mask & new_quality),
                                    ("full_session_quality_fixed",group_mask & full_quality)):
                    r=rel[mask]
                    rows.append(dict(condition=condition,environment=env,group=group,subset=subset,n_rois=int(mask.sum()),
                                     n_finite=int(np.isfinite(r).sum()),mean_reliability=float(np.nanmean(r)),
                                     median_reliability=float(np.nanmedian(r))))
            for i in range(len(rel)):
                cell_rows.append(dict(condition=condition,environment=env,original_roi=int(original_indices[i]),
                                      cell_class=session.roicat_classifier["id_to_label"][int(classes[i])],plane=int(planes[i]),
                                      reliability=float(rel[i]),fraction_active=float(fa[i]),quality=bool(new_quality[i]),
                                      full_session_quality=bool(full_quality[i])))
    summary=pd.DataFrame(rows)
    summary.to_csv(output/"reliability_summary.csv",index=False)
    pd.DataFrame(cell_rows).to_csv(output/"per_roi_metrics.csv",index=False)
    pd.DataFrame(coverage).to_csv(output/"trial_coverage.csv",index=False)
    np.savez_compressed(output/"recomputed_metrics.npz",original_roi=original_indices,classes=classes,planes=planes,
                        full_reliability=values["full"]["reliability"],trim70_reliability=values["trim70"]["reliability"],
                        full_fraction_active=values["full"]["fraction_active"],trim70_fraction_active=values["trim70"]["fraction_active"])
    with open(output/"baseline_validation.json","w") as file:
        json.dump(validation,file,indent=2)
    fig,axes=plt.subplots(1,3,figsize=(10,3.5),layout="constrained",sharey=True)
    for ax,env in zip(axes,(1,3,4)):
        for subset,label,color in (("quality_reselected","Quality reselected","C0"),("full_session_quality_fixed","Original quality cohort","C1")):
            table=summary[(summary.environment==env)&(summary.group=="all_selected")&(summary.subset==subset)].set_index("condition").loc[["full","trim70"]]
            ax.plot([0,1],table.mean_reliability,marker="o",color=color,label=label)
        ax.set_xticks([0,1],["All 268 trials","First 198 trials"])
        ax.set_title(f"Environment {env}")
        ax.spines[["right","top"]].set_visible(False)
    axes[0].set_ylabel("Mean spatial reliability")
    axes[-1].legend(frameon=False,fontsize=8)
    fig.savefig(output/"trim70_comparison.png",dpi=150)
    print("COVERAGE\n",pd.DataFrame(coverage).to_string(index=False),flush=True)
    print("SUMMARY\n",summary[(summary.group=="all_selected")].to_string(index=False),flush=True)
    print("SAVED",output,flush=True)


if __name__ == "__main__":
    run()
