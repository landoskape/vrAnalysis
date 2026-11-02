"""Render completed v6 sqrt-model results for PF=all versus PF=train."""
from pathlib import Path
import json
import numpy as np
import torch
from matplotlib import pyplot as plt
from dimensionality_manuscript.pipeline import ResultsStore, ResultsAggregator
from dimensionality_manuscript.configs import GainRegressionConfig
from dimensionality_manuscript.scripts.run import collect_sessions
from dimensionality_manuscript.figure_scripts.figure2.placefield_gain import (
    PlacefieldGainViewer, PlacefieldGainPerformanceViewer,
)
from copy import copy

OUTPUT=Path(__file__).resolve().parent / "sqrt_pf_comparison"
OUTPUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
store=ResultsStore(blob_cache_maxsize=0)
sessions=collect_sessions()
results=ResultsAggregator(GainRegressionConfig(),store,sessions)
summary={}
fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True,layout="constrained")
for estimator in ("gaussian","least_squares"):
    viewers={pf:PlacefieldGainPerformanceViewer(results,gain_estimator=estimator,placefield_split=pf,
                  gain_transform="sqrt",score_scale="sqrt",fontsize=10) for pf in ("all","train")}
    matched=np.logical_and.reduce([np.isfinite(v.comparison_scores).all(axis=(-2,-1)) for v in viewers.values()])
    for pf,v in viewers.items():
        v.comparison_scores[~matched]=np.nan
        names,scores=v.subject_scores(v.state)
        summary[f"{estimator}_{pf}"]=dict(mice=int(np.isfinite(scores).all(axis=1).sum()),
            sessions=int(matched.any(axis=1).sum()),session_environments=int(matched.sum()),
            means=np.nanmean(scores,axis=0).tolist(),mouse_names=names,mouse_scores=scores.tolist())
        if estimator=="gaussian":
            i=("all","train").index(pf)
            # Use the viewer's own drawing code in a shared-axis comparison figure.
            v.new_subplots=lambda *args, axis=axes[i], **kwargs:(fig,axis)
            v.plot(v.state)
            axes[i].set_title(f"PF={pf} | sqrt fit, sqrt scoring")
    print(estimator,{pf:summary[f"{estimator}_{pf}"]["means"] for pf in ("all","train")},flush=True)
fig.suptitle("Gaussian gain: paired observations; each dot is one mouse")
fig.savefig(OUTPUT / "performance_controls.png",dpi=170)
plt.close(fig)
# Use the original place-field gain viewer and the same example as the historical figure.
print("Loading the original viewer example session...",flush=True)
viewer=PlacefieldGainViewer(sessions,results,mouse="ATL027",session="ATL027/2023-08-04/702",environment=1,
    gain_estimator="gaussian",gain_transform="sqrt",score_scale="sqrt",placefield_split="all",
    include_covariance=False,include_familiarity=False,sort_method="rastermap_gain",
    gain_vmax=np.sqrt(2.),interpolation="none",beewidth=.15,figsize=(8,3.2),fontsize=10)
rows={}
states={}
for pf in ("all","train"):
    state=dict(viewer.state,placefield_split=pf)
    viewer.refresh_data(state)
    row=copy(viewer)
    rows[pf]=row;states[pf]=state
matched=np.logical_and.reduce([np.isfinite(v.summary_scores)&np.isfinite(v.summary_null_scores) for v in rows.values()])
for pf,row in rows.items():
    row.summary_scores=row.summary_scores.copy();row.summary_null_scores=row.summary_null_scores.copy()
    row.summary_scores[~matched]=np.nan;row.summary_null_scores[~matched]=np.nan
    row._group_summary(states[pf])
    fig=row.plot(states[pf])
    fig.suptitle(f"PF={pf} | sqrt model, sqrt scale | ATL027 / 2023-08-04 / 702, environment 1",fontsize=11)
    # Give the performance axes common limits for an honest visual comparison.
    fig.axes[-1].set_ylim(-.1,.3)
    fig.savefig(OUTPUT / f"placefield_gain_{pf}.png",dpi=170)
    print("SAVED",pf,"selected",row.selected_result,flush=True)
    plt.close(fig)
(OUTPUT / "summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print("OUTPUT",OUTPUT,flush=True)
