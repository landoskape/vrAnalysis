"""Check the same sigrebase-selected somata using independent Oasis extraction."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vrAnalysis.sessions import B2Session
from vrAnalysis.processors.spkmaps import SpkmapParams
from vrAnalysis.helpers import reliability_loo
from inspect_atl012_trim70 import ROIChunkSession,UncachedProcessor
from compare_engagement_neural import pearson,OUTPUT


def run():
    rows=[];fluor=[];timing=[]
    for key in ("ATL012_2023-03-01_701","ATL012_2023-02-28_701"):
        path=OUTPUT/"neural"/key
        cells=pd.read_csv(path/"per_roi_comparison.csv")
        cells=cells[(cells["mode"]=="matched_count")&(cells.cell_class=="c")&(cells.calibration_reliability>.3)&(cells.calibration_fraction_active>.1)]
        manifest=pd.read_csv(OUTPUT/"neural_manifest.csv");cut=int(manifest.loc[manifest.session_key==key,"cut"].iloc[0])
        mouse,date,sid=key.rsplit("_",2)
        s=B2Session.create(mouse,date,sid,params={"spks_type":"oasis"})
        original=np.sort(cells.original_roi.unique())
        planes=s.get_plane_idx()[original];offsets=np.r_[0,np.cumsum(s.values.roiPerPlane)]
        full_activity=np.empty((s.values.numFrames,len(original)),np.float32)
        bt=s.loadone("positionTracking.times");starts=s.loadone("trials.positionTracking");boundary=bt[starts[cut]]
        times=s.timestamps;before=times<boundary;after=~before
        for plane in np.unique(planes):
            local=original[planes==plane]-offsets[plane]
            spks=np.load(s.s2p_path/f"plane{plane}"/"spks.npy",mmap_mode="r")
            full_activity[:,planes==plane]=spks[local,:s.values.numFrames].T
            F=np.load(s.s2p_path/f"plane{plane}"/"F.npy",mmap_mode="r")[local,:s.values.numFrames]
            N=np.load(s.s2p_path/f"plane{plane}"/"Fneu.npy",mmap_mode="r")[local,:s.values.numFrames]
            for name,trace in (("raw_F",F),("neuropil",N),("corrected_F",F-.7*N),("oasis",spks[local,:s.values.numFrames])):
                fluor.append(dict(session_key=key,plane=int(plane),signal=name,n_rois=len(local),
                                  mean_before=float(np.mean(trace[:,before])),mean_after=float(np.mean(trace[:,after])),
                                  median_cell_std_before=float(np.median(np.std(trace[:,before],axis=1))),
                                  median_cell_std_after=float(np.median(np.std(trace[:,after],axis=1)))))
        processor=UncachedProcessor(ROIChunkSession(s,full_activity),SpkmapParams(autosave=False))
        maps=processor.get_processed_maps(force_recompute=True)
        required=processor._idx_required_position_bins();fulltrial=np.all(~np.isnan(maps.occmap[:,required]),axis=1)
        env=s.trial_environment
        allidx=[np.flatnonzero(fulltrial&(env==e)) for e in s.environments]
        valid=~np.any(np.stack([np.any(np.isnan(maps.occmap[t]),axis=0) for t in allidx]),axis=0)
        smap=maps.spkmap[:,:,valid].astype(float)
        for e,g in cells.groupby("environment"):
            idx=np.flatnonzero(fulltrial&(env==e));pre=idx[idx<cut];post=idx[idx>=cut]
            m=int(g.n_evaluation.iloc[0]);cal=pre[:-m];pre=pre[-m:];post=post[:m]
            take=np.searchsorted(original,g.original_roi.to_numpy())
            rpre=reliability_loo(smap[take][:,pre]);rpost=reliability_loo(smap[take][:,post])
            template=np.mean(smap[take][:,cal],axis=1)
            correlation_pre=pearson(template,np.mean(smap[take][:,pre],axis=1));correlation_post=pearson(template,np.mean(smap[take][:,post],axis=1))
            rows.append(dict(session_key=key,environment=int(e),n_cells=len(take),n_trials=m,
                             oasis_before=float(np.mean(rpre)),oasis_after=float(np.mean(rpost)),
                             oasis_template_before=float(np.nanmean(correlation_pre)),oasis_template_after=float(np.nanmean(correlation_post)),
                             sigrebase_before=float(g.reliability_before.mean()),sigrebase_after=float(g.reliability_after.mean()),
                             oasis_activity_before=float(np.mean(smap[take][:,pre])),oasis_activity_after=float(np.mean(smap[take][:,post]))))
        # Descriptive independent neural change point, never used to choose trims.
        t=pd.read_csv(path/"trial_template_overlap.csv")
        t=t[~t.calibration].dropna(subset=["mean_calibration_template_correlation"])
        y=t.mean_calibration_template_correlation.to_numpy();env=t.environment.to_numpy();tr=t.trial.to_numpy()-1
        residual=y.copy()
        for e in np.unique(env):residual[env==e]-=np.mean(y[env==e])
        cuts=np.arange(20,len(s.trial_environment)-20)
        z=(tr[None,:]>=cuts[:,None]).astype(float)
        for e in np.unique(env):z[:,env==e]-=np.mean(z[:,env==e],axis=1,keepdims=True)
        norm=np.sum(z*z,axis=1);cov=z@residual
        support=(np.sum(tr[None,:]<cuts[:,None],axis=1)>=10)&(np.sum(tr[None,:]>=cuts[:,None],axis=1)>=10)&(norm>1)
        gain=np.where(support,np.minimum(cov,0)**2/np.maximum(norm,1),-np.inf)
        neural_cut=int(cuts[np.argmax(gain)])
        timing.append(dict(session_key=key,behavior_cut=cut,behavior_minutes=float((bt[starts[cut]]-bt[0])/60),
                           descriptive_neural_cut=neural_cut,descriptive_neural_minutes=float((bt[starts[neural_cut]]-bt[0])/60)))
        print("ALTERNATIVE SIGNAL",key,[r for r in rows if r["session_key"]==key],flush=True)
        s.clear_cache()
    pd.DataFrame(rows).to_csv(OUTPUT/"alternative_signal_comparison.csv",index=False)
    pd.DataFrame(fluor).to_csv(OUTPUT/"fluorescence_stability.csv",index=False)
    pd.DataFrame(timing).to_csv(OUTPUT/"descriptive_neural_timing.csv",index=False)
    print("TIMING",timing,flush=True)


if __name__=="__main__":run()
