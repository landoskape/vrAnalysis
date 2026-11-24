"""Same-cell neural comparisons at behavior-selected cutoffs, with sham controls."""
from pathlib import Path
import sys
import argparse
import json
import gc
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vrAnalysis.sessions import B2Session
from vrAnalysis.processors.spkmaps import SpkmapParams
from vrAnalysis.helpers import reliability_loo
from vrAnalysis.metrics import FractionActive
from dimensionality_manuscript import PFPredQualityConfig, ResultsAggregator
from inspect_atl012_reliability import ReadOnlyStore
from inspect_atl012_trim70 import ROIChunkSession, UncachedProcessor

OUTPUT=Path(__file__).parent/"behavioral_engagement_inspection"
NEURAL=OUTPUT/"neural"


def pearson(a,b,axis=-1):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    a=a-np.mean(a,axis=axis,keepdims=True);b=b-np.mean(b,axis=axis,keepdims=True)
    den=np.sqrt(np.sum(a*a,axis=axis)*np.sum(b*b,axis=axis))
    return np.divide(np.sum(a*b,axis=axis),den,out=np.full_like(den,np.nan,dtype=float),where=den>1e-14)


def manifest():
    screen=pd.read_csv(OUTPUT/"behavior_screen.csv")
    rows=[]
    for row in screen[screen.candidate].to_dict("records"):
        rows.append(dict(session_key=row["session_key"],role="screen_positive",cut=int(row["cut_after_trial"]),reference=row["session_key"]))
    borderline=["ATL027_2023-08-02_701","ATL020_2023-04-11_701","ATL022_2023-04-24_701","CR_Hippocannula6_2022-09-06_701"]
    for key in borderline:
        row=screen[screen.session_key==key].iloc[0]
        rows.append(dict(session_key=key,role="borderline_exploratory",cut=int(row.cut_after_trial),reference=key))
    controls=[("ATL012_2023-02-28_701","ATL012_2023-03-01_701"),
              ("ATL012_2023-03-03_701","ATL012_2023-03-01_701"),
              ("ATL045_2024-02-02_702","ATL045_2024-02-01_701"),
              ("ATL049_2024-05-13_701","ATL049_2024-06-05_701"),
              ("ATL027_2023-08-07_701","ATL027_2023-08-02_701"),
              ("ATL020_2023-04-04_701","ATL020_2023-04-11_701"),
              ("ATL022_2023-04-26_701","ATL022_2023-04-24_701"),
              ("CR_Hippocannula6_2022-08-30_701","CR_Hippocannula6_2022-09-06_701")]
    for key,reference in controls:
        row=screen[screen.session_key==key].iloc[0];ref=screen[screen.session_key==reference].iloc[0]
        rows.append(dict(session_key=key,role="sham_control",cut=int(round(row.n_trials*ref.onset_fraction)),reference=reference))
    table=pd.DataFrame(rows)
    path=OUTPUT/"neural_manifest.csv"
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path),table,check_dtype=False)
    else:table.to_csv(path,index=False)
    return table


def compute_session(entry):
    key=entry["session_key"];cut=int(entry["cut"])
    mouse,date,sid=key.rsplit("_",2)
    output=NEURAL/key;output.mkdir(parents=True,exist_ok=True)
    if (output/"complete.json").exists():
        print("REUSE",key,flush=True);return
    session=B2Session.create(mouse,date,sid,params={"spks_type":PFPredQualityConfig().spks_type})
    selected=session.idx_rois.copy();ids=np.flatnonzero(selected)
    classes=np.asarray(session.roicat_classifier["class_predictions"])[selected]
    planes=session.get_plane_idx()[selected]
    envs=np.asarray(session.trial_environment);ntrials=len(envs)
    behavior=pd.read_csv(OUTPUT/"behavior_maps"/f"{key}.csv")
    stored=ResultsAggregator(PFPredQualityConfig,ReadOnlyStore(),[session],keys=["reliability_slot","fraction_active_slot","env_slot_ids"]).sel(squeeze_ones=False)
    slot_ids=stored["env_slot_ids"][0]
    stored_rel={int(env):stored["reliability_slot"][0,slot,:len(ids)] for slot,env in enumerate(slot_ids) if np.isfinite(env)}
    stored_fa={int(env):stored["fraction_active_slot"][0,slot,:len(ids)] for slot,env in enumerate(slot_ids) if np.isfinite(env)}
    session.clear_cache()
    source=sparse.load_npz(session.one_path/"mpci.roiSignificantFluorescenceRebase.npz").tocsc()
    # Trial eligibility is behavior-only. One zero-valued ROI avoids loading neural maps.
    processor=UncachedProcessor(ROIChunkSession(session,np.zeros((session.values.numFrames,1),np.float32)),SpkmapParams(autosave=False))
    maps=processor.get_processed_maps(force_recompute=True)
    required=processor._idx_required_position_bins()
    full_trial=np.all(~np.isnan(maps.occmap[:,required]),axis=1)
    usable=[np.flatnonzero(full_trial&(envs==env)) for env in session.environments if env>=0]
    validpos=~np.any(np.stack([np.any(np.isnan(maps.occmap[indices]),axis=0) for indices in usable]),axis=0)
    positions=processor.dist_centers[validpos]
    specs={};coverage=[]
    for env in session.environments:
        if env<0:continue
        env=int(env)
        allidx=np.flatnonzero(full_trial&(envs==env));pre=allidx[allidx<cut];post=allidx[allidx>=cut]
        # Keep evaluation trials nearest the boundary; calibration is earlier and disjoint.
        calibration_min=max(5,min(10,len(pre)//3))
        m=min(20,len(post),len(pre)-calibration_min)
        record=dict(session_key=key,environment=env,n_before=len(pre),n_after=len(post),matched_trials=max(0,m),role=entry["role"],cut=cut)
        if m<8 or len(pre)-m<5:
            record["status"]="insufficient_disjoint_calibration_or_evaluation_trials";coverage.append(record);continue
        before=pre[-m:];after=post[:m];cal=pre[:-m]
        # Trial-mean speed matching is a sensitivity analysis, not full position-wise matching.
        prepool=pre[-min(len(pre)-5,40):]
        postpool=post[:min(len(post),40)]
        sa=behavior.mean_speed.to_numpy()[prepool];sb=behavior.mean_speed.to_numpy()[postpool]
        distance=np.abs(np.log(np.maximum(sa,1)[:,None]/np.maximum(sb,1)[None,:]))
        pi,pj=linear_sum_assignment(distance)
        keep=np.argsort(distance[pi,pj])[:20]
        keep=keep[distance[pi[keep],pj[keep]]<.25]
        smbefore=prepool[pi[keep]];smafter=postpool[pj[keep]]
        smcal=pre[~np.isin(pre,smbefore)]
        smcal=smcal[smcal<min(smbefore)] if len(smbefore) else smcal[:0]
        if len(smcal)<5 or len(smbefore)<8:
            smbefore=np.array([],int);smafter=np.array([],int);smcal=np.array([],int)
        specs[env]=dict(all=allidx,pre_all=pre,post_all=post,before=before,after=after,calibration=cal,
                        speed_before=smbefore,speed_after=smafter,speed_calibration=smcal)
        record.update(status="ok",calibration_trials=len(cal),speed_matched_trials=len(smbefore),
                      pre_mean_speed=float(behavior.mean_speed.to_numpy()[before].mean()),post_mean_speed=float(behavior.mean_speed.to_numpy()[after].mean()))
        coverage.append(record)
    pd.DataFrame(coverage).to_csv(output/"trial_coverage.csv",index=False)
    del maps
    if not specs:
        (output/"complete.json").write_text(json.dumps(dict(**entry,n_rois=len(ids),no_eligible_environments=True)))
        print("NO ELIGIBLE ENVIRONMENTS",key,flush=True);return
    percell=[];summary=[];trialacc={}
    for start in range(0,len(ids),256):
        stop=min(start+256,len(ids))
        chunk=ROIChunkSession(session,source[:,ids[start:stop]].toarray())
        proc=UncachedProcessor(chunk,SpkmapParams(autosave=False))
        maps=proc.get_processed_maps(force_recompute=True)
        smap=np.asarray(maps.spkmap[:,:,validpos],dtype=np.float64)
        for env,spec in specs.items():
            for mode in ("matched_count","matched_trial_speed"):
                pre=spec["before"] if mode=="matched_count" else spec["speed_before"]
                post=spec["after"] if mode=="matched_count" else spec["speed_after"]
                cal=spec["calibration"] if mode=="matched_count" else spec["speed_calibration"]
                if len(pre)<8:continue
                rcal=reliability_loo(smap[:,cal]);fcal=np.atleast_1d(FractionActive.compute(smap[:,cal],activity_axis=2,fraction_axis=1,activity_method="rms",fraction_method="participation"))
                rpre=reliability_loo(smap[:,pre]);rpost=reliability_loo(smap[:,post])
                tcal=np.mean(smap[:,cal],axis=1);tpre=np.mean(smap[:,pre],axis=1);tpost=np.mean(smap[:,post],axis=1)
                overlap=pearson(tpre,tpost)
                cal_pre=pearson(tcal,tpre);cal_post=pearson(tcal,tpost)
                split_pre=pearson(np.mean(smap[:,pre[::2]],axis=1),np.mean(smap[:,pre[1::2]],axis=1))
                split_post=pearson(np.mean(smap[:,post[::2]],axis=1),np.mean(smap[:,post[1::2]],axis=1))
                reward_start=float(behavior.loc[behavior.environment==env,"reward_start"].iloc[0])
                nonreward=(positions<reward_start-10)|(positions>reward_start+30)
                nrpre=reliability_loo(smap[:,pre][:,:,nonreward]);nrpost=reliability_loo(smap[:,post][:,:,nonreward])
                for i in range(stop-start):
                    percell.append(dict(session_key=key,role=entry["role"],environment=env,mode=mode,original_roi=int(ids[start+i]),
                                        cell_class=session.roicat_classifier["id_to_label"][int(classes[start+i])],plane=int(planes[start+i]),
                                        n_calibration=len(cal),n_evaluation=len(pre),calibration_reliability=float(rcal[i]),calibration_fraction_active=float(fcal[i]),
                                        reliability_before=float(rpre[i]),reliability_after=float(rpost[i]),
                                        template_overlap=float(overlap[i]),calibration_template_before=float(cal_pre[i]),calibration_template_after=float(cal_post[i]),
                                        split_half_before=float(split_pre[i]),split_half_after=float(split_post[i]),
                                        nonreward_reliability_before=float(nrpre[i]),nonreward_reliability_after=float(nrpost[i]),
                                        mean_activity_before=float(np.mean(smap[i,pre])),mean_activity_after=float(np.mean(smap[i,post]))))
                if mode=="matched_count":
                    # Held-out trial/template agreement over the whole session.
                    eligible=(fcal>.1)&(rcal>.3)&(classes[start:stop]==0)
                    if eligible.any():
                        trialcorr=pearson(smap[eligible][:,spec["all"]],tcal[eligible,None,:])
                        valid=np.isfinite(trialcorr)
                        acc=trialacc.setdefault(env,dict(sum=np.zeros(len(spec["all"])),count=np.zeros(len(spec["all"]))))
                        acc["sum"]+=np.nansum(trialcorr,axis=0);acc["count"]+=valid.sum(axis=0)
            # Whole-session versus trim comparison uses identical original full-quality ROIs.
            fullrel=reliability_loo(smap[:,spec["all"]]);trimrel=reliability_loo(smap[:,spec["pre_all"]]);laterel=reliability_loo(smap[:,spec["post_all"]])
            fullq=(stored_rel[env][start:stop]>.3)&(stored_fa[env][start:stop]>.1)
            for i in range(stop-start):
                summary.append(dict(environment=env,original_roi=int(ids[start+i]),cell_class=session.roicat_classifier["id_to_label"][int(classes[start+i])],
                                    original_quality=bool(fullq[i]),full_reliability=float(fullrel[i]),trim_reliability=float(trimrel[i]),late_reliability=float(laterel[i])))
        del maps,smap,chunk,proc
        print("NEURAL",key,stop,"/",len(ids),flush=True)
    cells=pd.DataFrame(percell);cells.to_csv(output/"per_roi_comparison.csv",index=False)
    pd.DataFrame(summary).to_csv(output/"full_trim_per_roi.csv",index=False)
    trialrows=[]
    for env,acc in trialacc.items():
        values=np.divide(acc["sum"],acc["count"],out=np.full_like(acc["sum"],np.nan),where=acc["count"]>0)
        for trial,value,count in zip(specs[env]["all"],values,acc["count"]):
            trialrows.append(dict(trial=int(trial+1),environment=env,mean_calibration_template_correlation=float(value),n_cells=int(count),
                                  calibration=bool(trial in specs[env]["calibration"])))
    pd.DataFrame(trialrows).to_csv(output/"trial_template_overlap.csv",index=False)
    # Existing registration summaries around the behavior cutoff, without loading movies.
    times=session.timestamps
    threshold_time=session.loadone("positionTracking.times")[session.loadone("trials.positionTracking")[cut]]
    motion=[]
    for p in session.values.planeIDs:
        ops=np.load(session.s2p_path/f"plane{p}"/"ops.npy",allow_pickle=True).item()
        for condition,mask in (("before",times<threshold_time),("after",times>=threshold_time)):
            row=dict(plane=int(p),condition=condition)
            for name in ("corrXY","xoff","yoff","badframes"):
                if name in ops:
                    a=np.asarray(ops[name])[:len(times)][mask]
                    row[name+"_mean"]=float(np.mean(a));row[name+"_median"]=float(np.median(a))
            motion.append(row)
    pd.DataFrame(motion).to_csv(output/"motion_summary.csv",index=False)
    (output/"complete.json").write_text(json.dumps(dict(**entry,n_rois=len(ids),n_trials=ntrials,position_bins=int(validpos.sum())),indent=2))
    session.clear_cache();gc.collect()
    print("COMPLETE",key,flush=True)


def run():
    parser=argparse.ArgumentParser();parser.add_argument("--session");parser.add_argument("--all",action="store_true");args=parser.parse_args()
    NEURAL.mkdir(exist_ok=True)
    entries=manifest()
    if args.session:entries=entries[entries.session_key==args.session]
    elif not args.all:entries=entries[entries.session_key=="ATL012_2023-03-01_701"]
    errors=[]
    for entry in entries.to_dict("records"):
        try:compute_session(entry)
        except Exception as error:
            import traceback
            traceback.print_exc();errors.append(dict(**entry,error=repr(error)))
    if errors:
        pd.DataFrame(errors).to_csv(OUTPUT/"neural_errors.csv",index=False)
        raise RuntimeError(errors)
    if args.all:
        pd.DataFrame(columns=["session_key","role","cut","reference","error"]).to_csv(OUTPUT/"neural_errors.csv",index=False)


if __name__=="__main__":run()
