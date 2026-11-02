"""Compare PF=train/all shift curves for the sqrt gain model on sqrt scale."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from dimensionality_manuscript.pipeline import ResultsStore, ResultsAggregator
from dimensionality_manuscript.configs import GainRegressionConfig
from dimensionality_manuscript.configs.gain_regression import TRIAL_SHIFTS
from dimensionality_manuscript.scripts.run import collect_sessions
from dimensionality_manuscript.figure_scripts.figure2.placefield_gain import _finite_mean

OUTPUT=Path(__file__).resolve().parent / "sqrt_pf_comparison"
OUTPUT.mkdir(exist_ok=True)
results=ResultsAggregator(GainRegressionConfig(),ResultsStore(blob_cache_maxsize=0),collect_sessions())
shifts=np.asarray(TRIAL_SHIFTS)
curves={};nulls={};masks=[]
for pf in ("all","train"):
    values=[]
    for lag in shifts:
        selected=results.sel(keys=["r2_test_sqrt","n_trials_env","r2_test_sqrt_null"],
            squeeze_ones=False,placefield_split=pf,gain_estimator="gaussian",gain_transform="sqrt",trial_shift=int(lag))
        score=np.asarray(selected["r2_test_sqrt"],float).copy()
        score[np.asarray(selected["n_trials_env"])<10]=np.nan
        values.append(score)
        if lag==0:
            nulls[pf]=np.asarray(selected["r2_test_sqrt_null"],float).copy()
    curves[pf]=np.stack(values,axis=-1)
    masks.append(np.isfinite(curves[pf]).all(axis=-1)&np.isfinite(nulls[pf]))
matched=np.logical_and.reduce(masks)
mouse_names=np.asarray(results.mouse_names)
mice=list(dict.fromkeys(mouse_names.tolist()))
def by_mouse(values):
    session_values=_finite_mean(values,axis=1)
    return np.asarray([_finite_mean(session_values[mouse_names==mouse],axis=0) for mouse in mice])
fig,ax=plt.subplots(figsize=(8,4.8),layout="constrained")
rows=[];summary={}
for pf,color in (("all","#2566aa"),("train","#c25a18")):
    subject=by_mouse(np.where(matched[...,None],curves[pf],np.nan))
    null_subject=by_mouse(np.where(matched,nulls[pf],np.nan))
    means=np.nanmean(subject,axis=0)
    sem=np.nanstd(subject,axis=0,ddof=1)/np.sqrt(np.isfinite(subject).sum(axis=0))
    baseline=float(np.nanmean(null_subject))
    ax.plot(shifts,means,color=color,marker="o",markersize=5,lw=2,label=f"PF={pf}")
    ax.fill_between(shifts,means-sem,means+sem,color=color,alpha=.13,linewidth=0)
    ax.axhline(baseline,color=color,ls=":",lw=2,label=f"PF={pf} random roll ({baseline:.3f})")
    for lag,mean,error in zip(shifts,means,sem):
        rows.append(dict(placefield_split=pf,lag=int(lag),mean_r2=float(mean),sem=float(error),random_roll=baseline))
    summary[pf]=dict(mean_curve=means.tolist(),sem=sem.tolist(),random_roll=baseline,mouse_names=mice,mouse_curves=subject.tolist())
    print(pf,"curve",np.round(means,5).tolist(),"roll",baseline,flush=True)
ax.set_xticks(shifts)
ax.set_xlabel("Trial shift (target minus source; positive predicts future trials)")
ax.set_ylabel(r"Held-out $R^2$ (sqrt gain)")
ax.set_title("Gaussian gain: sqrt model, sqrt scoring scale")
ax.spines[["top","right"]].set_visible(False)
ax.legend(frameon=False,fontsize=9,loc="upper right")
fig.suptitle("PF=train versus PF=all | mouse mean ± SEM",fontsize=12)
fig.savefig(OUTPUT / "shift_curves_sqrt.png",dpi=180)
fig.savefig(OUTPUT / "shift_curves_sqrt.svg")
plt.close(fig)
pd.DataFrame(rows).to_csv(OUTPUT / "shift_curves_sqrt.csv",index=False)
summary["coverage"]=dict(mice=len(mice),sessions=int(matched.any(axis=1).sum()),session_environments=int(matched.sum()),lags=shifts.tolist())
(OUTPUT / "shift_curves_sqrt.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print("coverage",summary["coverage"],flush=True)
print("output",OUTPUT / "shift_curves_sqrt.png",flush=True)
