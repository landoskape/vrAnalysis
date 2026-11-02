from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from dimensionality_manuscript.configs import GainRegressionConfig
import dimensionality_manuscript.configs.gain_regression as gr
from dimensionality_manuscript.scripts.run import collect_sessions
from vrAnalysis.helpers import reliability_loo,stable_hash
from vrAnalysis.metrics import FractionActive
from dimilibi.regression import ReducedRankRegression
OUT=Path(__file__).resolve().parent/"gain_offset_isolation";OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
class FastRRR(ReducedRankRegression):
 def __init__(self,*args,**kwargs):kwargs["use_svd"]=True;super().__init__(*args,**kwargs)
gr.ReducedRankRegression=FastRRR
sessions={s.session_uid:s for s in collect_sessions()};rows=[];details=[]
for uid,env in [("ATL027.2023-08-04.702",1),("ATL049.2024-05-01.702",4)]:
 ses=sessions[uid];cfg=GainRegressionConfig(gain_transform="sqrt",trial_shift=3)
 print("LOAD",uid,flush=True);data=cfg._load_session(ses)
 folds=gr.split_trials(np.asarray(ses.trial_environment)[data["full_trials"]],cfg.trial_fractions,gr._seed(uid,cfg.split_seed,"trials"))
 maps=data["env_maps"].spkmap[list(data["env_maps"].environments).index(env)]
 trials=data["trials_by_environment"][env]
 columns={name:np.flatnonzero(np.isin(trials,data["full_trials"][fold])) for name,fold in zip(gr.SPLIT_NAMES,folds)}
 pred_trials=columns["train"]
 pair={name:c[(c>=3)&(c<maps.shape[1]-3)] for name,c in columns.items()}
 screens={}
 for screen in ("all","train"):
  x=maps if screen=="all" else maps[:,pred_trials]
  rel=reliability_loo(x);fa=FractionActive.compute(x,activity_axis=2,fraction_axis=1,activity_method="rms",fraction_method="participation")
  screens[screen]=np.isfinite(rel)&np.isfinite(fa)&(rel>=cfg.reliability_threshold)&(fa>=cfg.fraction_active_threshold)
 union=screens["all"]|screens["train"];indices=np.flatnonzero(union)
 gains={}
 for ref in ("all","train"):
  print("GAUSSIAN",uid,ref,len(indices),flush=True)
  g,_,_=gr.gaussian_gain_matrix(maps[union],data["positions"],None if ref=="all" else pred_trials)
  gains[ref]=gr.apply_gain_transform(g,"sqrt")
 finite=np.logical_and.reduce([np.isfinite(g).all(axis=1) for g in gains.values()])
 # Fixed neuron assignment across all four conditions, independent of the selected screen.
 source=np.array([int(stable_hash(uid,env,int(data["roi_lookup"][idx]),"fixed-screen-cross"),16)%2==0 for idx in indices])
 for screen in ("all","train"):
  selected=screens[screen][indices]&finite
  sr=np.flatnonzero(selected&source);tr=np.flatnonzero(selected&~source)
  for ref in ("all","train"):
   g=gains[ref];ts=gr.split_tensors(g,sr,tr,pair,trial_shift=3)
   fit=gr.optimize_rrr(*ts["train"],*ts["val"],alpha_bounds=cfg.ALPHA_BOUNDS,max_iterations=cfg.GSS_MAX_ITERATIONS)
   p=fit.model.predict(ts["test"][0],rank=fit.rank,nonnegative=True).numpy();y=ts["test"][1].numpy()
   def decompose(p,y):
    sst=np.sum((y-y.mean())**2);within=np.sum((y-y.mean(axis=0))**2)
    mp=1-np.sum((y-p.mean(axis=0))**2)/sst
    tp=(within-np.sum(((y-y.mean(axis=0))-(p-p.mean(axis=0)))**2))/sst
    return mp+tp,mp,tp
   real,mp,tp=decompose(p,y);nroll=20;rng=np.random.default_rng(814)
   roll=[];withinroll=[];trainraw=[]
   sub=g[np.r_[sr,tr]];ss=np.arange(len(sr));tt=np.arange(len(sr),len(sr)+len(tr))
   for rep in range(nroll):
    for typ in ("full","within"):
     shuffled=gr.roll_gain_rows(sub,rng) if typ=="full" else sub.copy()
     if typ=="within":
      for cc in pair.values():
       for neuron in range(len(sub)):shuffled[neuron,cc]=sub[neuron,rng.permutation(cc)]
     ns=gr.split_tensors(shuffled,ss,tt,pair,trial_shift=3)
     model=FastRRR(alpha=fit.alpha,fit_intercept=True).fit(*ns["train"])
     pp=model.predict(ns["test"][0],rank=fit.rank,nonnegative=True).numpy();yy=ns["test"][1].numpy()
     (roll if typ=="full" else withinroll).append(decompose(pp,yy))
   rec=dict(session=uid,screen=screen,reference=ref,n_source=len(sr),n_target=len(tr),alpha=fit.alpha,rank=fit.rank,
       real_r2=real,mean_part=mp,trial_part=tp,roll_r2=float(np.mean(roll,axis=0)[0]),roll_mean_part=float(np.mean(roll,axis=0)[1]),roll_trial_part=float(np.mean(roll,axis=0)[2]),
       # Within-fold rolling with lagged source indices does not preserve source fold marginals;
       # retained only as exploratory, not as the primary matched-fold control.
       exploratory_within_shuffle_r2=float(np.mean(withinroll,axis=0)[0]))
   rows.append(rec);pd.DataFrame(rows).to_csv(OUT/"crossed_examples.csv",index=False)
   print("CROSS",rec,flush=True)
print("DONE",flush=True)
