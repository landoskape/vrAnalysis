from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dimensionality_manuscript.pipeline import ResultsStore
from dimensionality_manuscript.configs import GainRegressionConfig
from dimensionality_manuscript.configs.gain_regression import split_tensors
from dimilibi.regression import ReducedRankRegression
OUT=Path(__file__).resolve().parent/"gain_offset_dive"
df=pd.read_csv(OUT/"cohort.csv");store=ResultsStore(blob_cache_maxsize=0);torch.set_num_threads(2)
records=[]
for i,((uid,pf),rows) in enumerate(df.groupby(["session","pf"])):
 for lag in (0,3):
    cfg=GainRegressionConfig(placefield_split=pf,gain_transform="sqrt",trial_shift=lag)
    b=store.get(uid,cfg)
    for row in rows.itertuples():
        slot=int(row.slot);gain=b["gain_matrices"][slot]
        ts=split_tensors(gain,b["source_rows"][slot],b["target_rows"][slot],b["trial_columns"][slot],trial_shift=lag)
        model=ReducedRankRegression(alpha=float(b["rrr_alpha"][slot]),fit_intercept=True,use_svd=True).fit(*ts["train"])
        p=model.predict(ts["test"][0],rank=int(b["rrr_rank"][slot]),nonnegative=True).numpy();y=ts["test"][1].numpy()
        sst=np.sum((y-y.mean())**2);within=np.sum((y-y.mean(axis=0))**2)
        constant_sse=np.sum((y-p.mean(axis=0))**2)
        centered_sse=np.sum(((y-y.mean(axis=0))-(p-p.mean(axis=0)))**2)
        mean_part=1-constant_sse/sst;temporal=(within-centered_sse)/sst
        r2=1-np.sum((y-p)**2)/sst
        np.testing.assert_allclose(r2,b["r2_test_sqrt"][slot],atol=1e-7,rtol=1e-6)
        np.testing.assert_allclose(r2,mean_part+temporal,atol=1e-10)
        records.append(dict(session=uid,mouse=row.mouse,pf=pf,slot=slot,lag=lag,r2=r2,
            prediction_mean_part=mean_part,trial_varying_part=temporal,
            centered_temporal_r2=1-centered_sse/within,
            predicted_mean=float(p.mean()),true_test_mean=float(y.mean()),target_training_mean=float(ts["train"][1].mean())))
 if i%40==0:print("session/settings",i,flush=True)
f=pd.DataFrame(records);f.to_csv(OUT/"prediction_decomposition.csv",index=False)
summary=f.groupby(["pf","lag","mouse","session"]).mean(numeric_only=True).groupby(["pf","lag","mouse"]).mean().groupby(["pf","lag"]).mean()
summary.to_csv(OUT/"prediction_decomposition_summary.csv")
print(summary.to_string(),flush=True)
