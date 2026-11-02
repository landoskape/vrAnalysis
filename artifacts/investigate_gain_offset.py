from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import json
from dimensionality_manuscript.pipeline import ResultsStore
from dimensionality_manuscript.configs import GainRegressionConfig
from dimensionality_manuscript.scripts.run import collect_sessions
from dimensionality_manuscript.configs.gain_regression import TRIAL_SHIFTS, roll_gain_rows
from dimensionality_manuscript.figure_scripts.figure2.placefield_gain import _finite_mean
OUT=Path(__file__).resolve().parent/"gain_offset_dive"
OUT.mkdir(exist_ok=True)
store=ResultsStore(blob_cache_maxsize=0);sessions=collect_sessions();records=[]
for i,ses in enumerate(sessions):
    for pf in ("all","train"):
        cfg=GainRegressionConfig(placefield_split=pf,gain_transform="sqrt",gain_estimator="gaussian")
        b=store.get(ses.session_uid,cfg)
        if b is None:continue
        others={lag:store.get(ses.session_uid,replace(cfg,trial_shift=lag)) for lag in TRIAL_SHIFTS}
        for slot,gain in enumerate(b["gain_matrices"]):
            if gain is None or b["n_trials_env"][slot]<10:continue
            columns=b["trial_columns"][slot];targets=b["target_rows"][slot]
            train=gain[targets][:,columns["train"]];test=gain[targets][:,columns["test"]]
            base=((test-train.mean(axis=1,keepdims=True))**2).mean()
            grand=((test-train.mean())**2).mean()
            var=((test-test.mean())**2).mean()
            within=((test-test.mean(axis=1,keepdims=True))**2).mean()
            row=dict(session=ses.session_uid,mouse=ses.mouse_name,pf=pf,slot=slot,env=b["env_slot_ids"][slot],
                     n_train=train.shape[1],n_test=test.shape[1],n_targets=len(targets),
                     mean_only_r2=1-base/var,grand_mean_r2=1-grand/var,
                     oracle_neuron_mean_fraction=1-within/var,train_mean=float(train.mean()),test_mean=float(test.mean()))
            for lag in TRIAL_SHIFTS:
                if others[lag] is None:continue
                mse=float(others[lag]["mse_test_sqrt"][slot]);r2=float(others[lag]["r2_test_sqrt"][slot])
                row[f"lag{lag}"]=r2;row[f"vs_mean_lag{lag}"]=1-mse/base
            row["roll_r2"]=float(b["r2_test_sqrt_null"][slot])
            # No model fitting: does the mean-only baseline itself drop under full-row rolls?
            rng=np.random.default_rng(5);baselines=[]
            for _ in range(20):
                rolled=roll_gain_rows(gain[targets],rng)
                tr=rolled[:,columns["train"]];te=rolled[:,columns["test"]]
                baselines.append(1-((te-tr.mean(axis=1,keepdims=True))**2).mean()/((te-te.mean())**2).mean())
            row["rolled_mean_only_r2"]=float(np.mean(baselines))
            records.append(row)
    if i%30==0:print("sessions",i,flush=True)
df=pd.DataFrame(records)
# Match PF settings by session/environment, for every usable lag.
keys=["session","slot"]
a=df[df.pf=="all"].set_index(keys);t=df[df.pf=="train"].set_index(keys)
common=a.index.intersection(t.index)
df=df.set_index(keys).loc[common].reset_index()
df.to_csv(OUT/"cohort.csv",index=False)
cols=["mean_only_r2","rolled_mean_only_r2","grand_mean_r2","oracle_neuron_mean_fraction","lag0","lag3","lag-3","roll_r2","vs_mean_lag0","vs_mean_lag3","train_mean","test_mean"]
summary=[]
for pf,group in df.groupby("pf"):
    sessionmean=group.groupby(["mouse","session"])[cols].mean()
    mm=sessionmean.groupby("mouse").mean()
    summary.append(dict(pf=pf,**mm.mean().to_dict()))
    mm.to_csv(OUT/f"mouse_{pf}.csv")
print(pd.DataFrame(summary).to_string(index=False),flush=True)
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
# Choose historical example and a different mouse near median off-zero performance.
train=df[(df.pf=="train")&(df.n_test>=6)]
median=float(train.lag3.median())
candidate=train[~train.mouse.isin(["ATL027"])].copy()
candidate["distance"]=abs(candidate.lag3-median)
print("EXAMPLES",train[train.session=="ATL027.2023-08-04.702"][["session","slot","env","lag0","lag3","mean_only_r2"]].to_string(index=False),flush=True)
print(candidate.nsmallest(2,"distance")[["session","slot","env","lag0","lag3","mean_only_r2"]].to_string(index=False),flush=True)
