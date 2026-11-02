from pathlib import Path
import numpy as np
import pandas as pd
from dimensionality_manuscript.pipeline import ResultsStore
from dimensionality_manuscript.configs import GainRegressionConfig
OUT=Path(__file__).resolve().parent/"gain_offset_dive"
d=pd.read_csv(OUT/"cohort.csv");store=ResultsStore(blob_cache_maxsize=0);records=[]
for (uid,pf),rows in d.groupby(["session","pf"]):
 b=store.get(uid,GainRegressionConfig(placefield_split=pf,gain_transform="sqrt"))
 for row in rows.itertuples():
    slot=int(row.slot);g=b["gain_matrices"][slot];cols=b["trial_columns"][slot];src=b["source_rows"][slot];tgt=b["target_rows"][slot]
    y={k:g[tgt][:,v].T for k,v in cols.items()};mu=y["train"].mean(axis=0)
    time=np.arange(g.shape[1]);m=time[cols["train"]].mean();sd=time[cols["train"]].std()
    x={k:(time[v]-m)/sd for k,v in cols.items()}
    best=None
    for alpha in np.logspace(-4,4,25):
        beta=x["train"] @ (y["train"]-mu)/(np.sum(x["train"]**2)+alpha)
        pred=np.maximum(0,mu+x["val"][:,None]*beta)
        loss=np.mean((pred-y["val"])**2)
        if best is None or loss<best[0]:best=(loss,beta,alpha)
    pred=np.maximum(0,mu+x["test"][:,None]*best[1]);truth=y["test"];sst=np.sum((truth-truth.mean())**2)
    centered_sse=np.sum(((truth-truth.mean(axis=0))-(pred-pred.mean(axis=0)))**2)
    within=np.sum((truth-truth.mean(axis=0))**2)
    population_source=g[src].mean(axis=0);population_target=g[tgt].mean(axis=0)
    records.append(dict(session=uid,mouse=row.mouse,pf=pf,slot=slot,time_only_r2=1-np.sum((pred-truth)**2)/sst,
        time_only_trial_part=(within-centered_sse)/sst,alpha=best[2],
        source_target_pop_corr_lag3=np.corrcoef(population_source[:-3],population_target[3:])[0,1],
        source_time_corr=np.corrcoef(time,population_source)[0,1]))
f=pd.DataFrame(records);f.to_csv(OUT/"time_baseline.csv",index=False)
summary=f.groupby(["pf","mouse","session"]).mean(numeric_only=True).groupby(["pf","mouse"]).mean().groupby("pf").mean()
summary.to_csv(OUT/"time_baseline_summary.csv");print(summary.to_string())
print(f[f.session.isin(["ATL027.2023-08-04.702","ATL049.2024-05-01.702"])&(f.slot==1)].to_string(index=False))
