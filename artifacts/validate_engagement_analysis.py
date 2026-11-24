"""Validate source baselines and disjoint, count-matched evaluation trials."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vrAnalysis.sessions import B2Session
from vrAnalysis.processors.spkmaps import SpkmapParams
from dimensionality_manuscript import PFPredQualityConfig,ResultsAggregator
from inspect_atl012_reliability import ReadOnlyStore
from inspect_atl012_trim70 import ROIChunkSession,UncachedProcessor
from compare_engagement_neural import OUTPUT


def run():
    manifest=pd.read_csv(OUTPUT/"neural_manifest.csv")
    rows=[]
    for entry in manifest.to_dict("records"):
        path=OUTPUT/"neural"/entry["session_key"]
        info=json.loads((path/"complete.json").read_text())
        if info.get("no_eligible_environments"):continue
        mouse,date,sid=entry["session_key"].rsplit("_",2)
        s=B2Session.create(mouse,date,sid,params={"spks_type":"sigrebase"})
        stored=ResultsAggregator(PFPredQualityConfig,ReadOnlyStore(),[s],keys=["reliability_slot","fraction_active_slot","env_slot_ids"]).sel(squeeze_ones=False)
        full=pd.read_csv(path/"full_trim_per_roi.csv")
        proc=UncachedProcessor(ROIChunkSession(s,np.zeros((s.values.numFrames,1),np.float32)),SpkmapParams(autosave=False))
        maps=proc.get_processed_maps(force_recompute=True)
        valid=np.all(~np.isnan(maps.occmap[:,proc._idx_required_position_bins()]),axis=1)
        plans={}
        for env,g in full.groupby("environment"):
            g=g.sort_values("original_roi")
            slot=int(np.flatnonzero(stored["env_slot_ids"][0]==env)[0])
            expected=stored["reliability_slot"][0,slot,:len(g)]
            fa=stored["fraction_active_slot"][0,slot,:len(g)]
            keep=(expected>.3)&(fa>.1)
            assert np.array_equal(keep,g.original_quality.to_numpy())
            error=float(np.max(np.abs(g.full_reliability.to_numpy()[keep]-expected[keep])))
            absolute=np.abs(g.full_reliability.to_numpy()[keep]-expected[keep])
            mean_error=float(abs(np.mean(g.full_reliability.to_numpy()[keep])-np.mean(expected[keep])))
            # The scientific comparisons report cohort means. Check these and
            # record the full per-cell discrepancy instead of hiding outliers.
            assert mean_error<1e-4 and float(np.mean(absolute))<1e-4,(entry["session_key"],env,error,mean_error)
            idx=np.flatnonzero(valid&(s.trial_environment==env));pre=idx[idx<entry["cut"]];post=idx[idx>=entry["cut"]]
            cmax=max(5,min(10,len(pre)//3));m=min(20,len(post),len(pre)-cmax)
            before=pre[-m:];after=post[:m];cal=pre[:-m]
            assert len(before)==len(after) and len(before)>=8 and len(cal)>=5
            assert not np.intersect1d(cal,before).size and not np.intersect1d(cal,after).size
            assert max(cal)<min(before) and max(before)<min(after)
            plans[str(env)]=dict(calibration_zero_based=cal.tolist(),before_zero_based=before.tolist(),after_zero_based=after.tolist())
            rows.append(dict(session_key=entry["session_key"],environment=int(env),quality_baseline_max_abs_error=error,
                             quality_baseline_mean_abs_error=float(np.mean(absolute)),quality_baseline_cohort_mean_error=mean_error,
                             n_calibration=len(cal),matched_evaluation_trials=m,calibration_disjoint=True))
        (path/"validated_trial_indices.json").write_text(json.dumps(plans,indent=2))
        print("VALIDATED",entry["session_key"],flush=True)
        s.clear_cache()
    pd.DataFrame(rows).to_csv(OUTPUT/"analysis_validation.csv",index=False)


if __name__=="__main__":run()
