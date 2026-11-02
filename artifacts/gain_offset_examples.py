from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from dimensionality_manuscript.pipeline import ResultsStore
from dimensionality_manuscript.configs import GainRegressionConfig
from dimensionality_manuscript.configs.gain_regression import split_tensors, roll_gain_rows, _seed
from dimilibi.regression import ReducedRankRegression
OUT=Path(__file__).resolve().parent/"gain_offset_dive";OUT.mkdir(exist_ok=True)
torch.set_num_threads(2);store=ResultsStore(blob_cache_maxsize=0)
records=[];details=[]
examples=[("ATL027.2023-08-04.702",1),("ATL049.2024-05-01.702",1)]
def fitpredict(g,src,tgt,cols,alpha,rank,lag=0):
    ts=split_tensors(g,src,tgt,cols,trial_shift=lag)
    model=ReducedRankRegression(alpha=alpha,fit_intercept=True,use_svd=True).fit(*ts["train"])
    pred=model.predict(ts["test"][0],rank=rank,nonnegative=True).numpy()
    return pred,ts["test"][1].numpy()
def metric(p,y):return 1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)
fig,axs=plt.subplots(2,3,figsize=(12,7),layout="constrained")
for ei,(uid,slot) in enumerate(examples):
 for pf in ("all","train"):
    cfg=GainRegressionConfig(placefield_split=pf,gain_transform="sqrt")
    b=store.get(uid,cfg);g=b["gain_matrices"][slot];src=b["source_rows"][slot];tgt=b["target_rows"][slot];cols=b["trial_columns"][slot]
    if g is None:continue
    alpha=float(b["rrr_alpha"][slot]);rank=int(b["rrr_rank"][slot]);p,y=fitpredict(g,src,tgt,cols,alpha,rank)
    np.testing.assert_allclose(metric(p,y),b["r2_test_sqrt"][slot],atol=1e-7,rtol=1e-6)
    means=g[tgt][:,cols["train"]].mean(axis=1)
    permean=np.broadcast_to(means,y.shape)
    condition_values={"RRR lag 0":[metric(p,y)],"Training neuron means":[metric(permean,y)]}
    # Renormalize the SAME neurons to mean raw gain one using all trials. This is a
    # post-hoc counterfactual, not an independent training-only pipeline.
    normalized=g/np.sqrt(np.mean(g*g,axis=1,keepdims=True))
    for lag in (0,3):
        lb=store.get(uid,replace(cfg,trial_shift=lag));a=float(lb["rrr_alpha"][slot]);r=int(lb["rrr_rank"][slot])
        pp,yy=fitpredict(g,src,tgt,cols,a,r,lag)
        np.testing.assert_allclose(metric(pp,yy),lb["r2_test_sqrt"][slot],atol=1e-7,rtol=1e-6)
        condition_values[f"RRR lag {lag}"]=[metric(pp,yy)]
        pn,yn=fitpredict(normalized,src,tgt,cols,a,r,lag)
        condition_values[f"All-normalized lag {lag}"]=[metric(pn,yn)]
    rng=np.random.default_rng(20261101)
    for label in ("Source shuffled within fold","Both shuffled within fold","Source full-row roll","Original full-row roll"):
        vals=[];base=[]
        for rep in range(20):
            if "full-row" in label:
                shuffled=roll_gain_rows(g,rng)
                if label.startswith("Source"):shuffled[tgt]=g[tgt]
            else:
                shuffled=g.copy();rows=src if label.startswith("Source") else np.arange(len(g))
                for cc in cols.values():
                    for neuron in rows:
                        shuffled[neuron,cc]=g[neuron,rng.permutation(cc)]
            pp,yy=fitpredict(shuffled,src,tgt,cols,alpha,rank)
            vals.append(metric(pp,yy))
            tr=shuffled[tgt][:,cols["train"]].mean(axis=1)
            base.append(metric(np.broadcast_to(tr,yy.shape),yy))
        condition_values[label]=vals
        details.append(dict(session=uid,slot=slot,pf=pf,condition=label,values=vals,mean_baseline=base))
    for label,vals in condition_values.items():
        records.append(dict(session=uid,pf=pf,condition=label,r2=float(np.mean(vals)),sd=float(np.std(vals))))
    print(uid,pf,{k:round(float(np.mean(v)),4) for k,v in condition_values.items()},flush=True)
    if pf=="train":
        ax=axs[ei,0]
        ax.scatter(means,y.mean(axis=0),s=5,alpha=.25)
        ax.plot([0,1.5],[0,1.5],c="k",lw=.7)
        ax.set(xlabel="Neuron mean in training trials",ylabel="Neuron mean in held-out trials",title=uid)
        ax=axs[ei,1]
        # Population-wide fold shift: compare every retained trial with its split assignment.
        ax.plot(g[src].mean(axis=0),color="0.65",lw=1,label="Source population")
        ax.scatter(cols["train"],g[src][:,cols["train"]].mean(axis=0),c="#2566aa",s=12,label="Train")
        ax.scatter(cols["test"],g[src][:,cols["test"]].mean(axis=0),c="#c25a18",s=18,label="Test")
        ax.set(xlabel="Retained environment trial",ylabel="Mean sqrt gain",title="PF=train: fold-dependent activity level")
        ax.legend(fontsize=7,frameon=False)
        ax=axs[ei,2]
        labels=["Training neuron means","RRR lag 0","RRR lag 3","Source shuffled within fold","Original full-row roll"]
        values=[np.mean(condition_values[k]) for k in labels]
        ax.bar(range(len(labels)),values,color=[".6","#2566aa","#c25a18","#38815a","#b44343"])
        ax.set_xticks(range(len(labels)),["Mean only","Lag 0","Lag 3","Source shuffle\nwithin fold","Full roll"],rotation=35,ha="right",fontsize=8)
        ax.set_ylabel("Held-out pooled R2")
for ax in axs.flat:ax.spines[["top","right"]].set_visible(False)
fig.savefig(OUT/"examples.png",dpi=160);plt.close(fig)
pd.DataFrame(records).to_csv(OUT/"example_results.csv",index=False)
(OUT/"example_rolls.json").write_text(json.dumps(details,indent=2))
