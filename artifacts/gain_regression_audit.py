"""Reproduce the stored gain-regression comparison and audit without changing results."""
from pathlib import Path
from dataclasses import replace
from collections import Counter
import json
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from threadpoolctl import threadpool_limits
from dimensionality_manuscript.pipeline import ResultsStore, ResultsAggregator
from dimensionality_manuscript.configs import GainRegressionConfig
from dimensionality_manuscript.scripts.run import collect_sessions
from dimensionality_manuscript.figure_scripts.figure2.placefield_gain import gain_performance_comparison, _finite_mean
from dimensionality_manuscript.configs.gain_regression import split_tensors, roll_gain_rows, _seed, score_rrr
from dimilibi.regression import ReducedRankRegression

OUTPUT = Path(__file__).resolve().parent / "gain_regression_audit"
OUTPUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
store = ResultsStore(blob_cache_maxsize=0)
sessions = collect_sessions()
metadata = store.summary_table(analysis_type="gain_regression")
results = ResultsAggregator(GainRegressionConfig(), store, sessions)
mouse_names = np.asarray(results.mouse_names)
mice = list(dict.fromkeys(mouse_names.tolist()))
summary = []
mouse_rows = []
coverage = {}
fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained", sharey="col")
for e, estimator in enumerate(("gaussian", "least_squares")):
    scale_arrays = {scale: gain_performance_comparison(results, estimator, 10, scale) for scale in ("raw", "sqrt")}
    matched = np.logical_and.reduce([np.isfinite(a).all(axis=(-2, -1)) for a in scale_arrays.values()])
    coverage[estimator] = {"session_environments": int(matched.sum()), "sessions": int(matched.any(axis=1).sum())}
    for k, (scale, array) in enumerate(scale_arrays.items()):
        array = np.where(matched[..., None, None], array, np.nan)
        session_values = _finite_mean(array, axis=1)
        mouse_values = np.asarray([_finite_mean(session_values[mouse_names == mouse], axis=0) for mouse in mice])
        for t, transform in enumerate(("raw", "sqrt")):
            for c, condition in enumerate(("same_trial", "lag_pm1", "random_roll")):
                values = mouse_values[:, t, c]
                for mouse, value in zip(mice, values):
                    mouse_rows.append(dict(estimator=estimator, score_scale=scale, fit=transform, condition=condition, mouse=mouse, r2=float(value)))
                summary.append(dict(estimator=estimator, score_scale=scale, fit=transform, condition=condition, mice=int(np.isfinite(values).sum()), mean_r2=float(_finite_mean(values)), min_mouse=float(np.nanmin(values)), max_mouse=float(np.nanmax(values))))
                x = c * 2 + t
                offsets = np.linspace(-.13, .13, len(values))
                axs[e,k].scatter(x + offsets, values, color=("black", "#2566aa")[t], s=15, alpha=.7)
                axs[e,k].plot([x-.22,x+.22], [_finite_mean(values)]*2, color=("black", "#2566aa")[t], lw=2)
        axs[e,k].axhline(0,color=".6",lw=.6,ls=":")
        axs[e,k].set_title(f"{estimator.replace('_',' ').title()} estimator | {scale} scoring scale")
        axs[e,k].set_xticks(range(6), [f"{fit.title()}\n{condition}" for condition in ("Same trial", "Lag +/-1", "Random roll") for fit in ("raw", "sqrt")], rotation=35, ha="right")
        axs[e,k].set_ylabel("Held-out R2")
        axs[e,k].spines[["top","right"]].set_visible(False)
fig.suptitle("Gain regression v6: each dot is one mouse; bars are mouse means",fontsize=13)
fig.savefig(OUTPUT / "comparison.png",dpi=170)
plt.close(fig)
pd.DataFrame(summary).to_csv(OUTPUT / "summary.csv",index=False)
pd.DataFrame(mouse_rows).to_csv(OUTPUT / "mouse_scores.csv",index=False)
print("COVERAGE",json.dumps(coverage),flush=True)
print(pd.DataFrame(summary).pivot(index=["estimator","score_scale","fit"],columns="condition",values="mean_r2").round(5).to_string(),flush=True)
# Verify the implied pooled target variance is identical across models and lags
# for every finite held-out observation, rather than only selected spot checks.
variance_checks=[]
for estimator in ("gaussian","least_squares"):
    for scale in ("raw","sqrt"):
        reference=None
        for transform in ("raw","sqrt"):
            for lag in (-1,0,1):
                selected=results.sel(keys=[f"r2_test_{scale}",f"mse_test_{scale}"],squeeze_ones=False,
                    gain_estimator=estimator,gain_transform=transform,trial_shift=lag)
                variance=selected[f"mse_test_{scale}"] / (1-selected[f"r2_test_{scale}"])
                if reference is None: reference=variance
                finite=np.isfinite(variance)&np.isfinite(reference)
                np.testing.assert_allclose(variance[finite],reference[finite],rtol=1e-9,atol=1e-10)
                variance_checks.append(dict(estimator=estimator,scale=scale,fit=transform,lag=lag,
                    observations=int(finite.sum()),max_relative_error=float(np.max(np.abs(variance[finite]-reference[finite])/np.maximum(reference[finite],1e-15)))))
print("TARGET_VARIANCE_CHECKS",len(variance_checks),"max_error",max(r["max_relative_error"] for r in variance_checks),flush=True)
pd.DataFrame(variance_checks).to_csv(OUTPUT / "target_variance_checks.csv",index=False)
control_differences=[]
mouse_frame=pd.DataFrame(mouse_rows)
for (estimator,scale,fit),group in mouse_frame.groupby(["estimator","score_scale","fit"]):
    table=group.pivot(index="mouse",columns="condition",values="r2")
    for condition in ("lag_pm1","random_roll"):
        delta=table["same_trial"]-table[condition]
        control_differences.append(dict(estimator=estimator,scale=scale,fit=fit,control=condition,
            mean_advantage=float(delta.mean()),mice_above=int((delta>0).sum()),mice=int(delta.notna().sum())))
pd.DataFrame(control_differences).to_csv(OUTPUT / "control_advantages.csv",index=False)
print("CONTROL_ADVANTAGES",pd.DataFrame(control_differences).round(5).to_string(index=False),flush=True)
# Historical native-scale comparison, retaining all-trial place fields for v3.
history=[]
for version in ("v3", "v5", "v6"):
    for estimator in ("gaussian", "least_squares"):
        for transform in ("raw", "sqrt"):
            candidates=[r for r in metadata if r["schema_version"]==version and f"estimator={estimator}_" in r["analysis_summary"] and f"gain={transform}_" in r["analysis_summary"] and ("pf=all_" in r["analysis_summary"] if version=="v3" else "shift=+0_" in r["analysis_summary"])]
            values={}
            ranks=[]; ntrials=[]
            for row in candidates:
                blob=store.get_by_uid(row["result_uid"])
                if blob is None: continue
                a=np.asarray(blob["r2_test"],float).copy()
                a[np.asarray(blob["n_trials_env"])<10]=np.nan
                if np.isfinite(a).any(): values[row["session_id"]]=float(_finite_mean(a))
                ranks.extend(np.asarray(blob["rrr_rank"])[np.isfinite(blob["rrr_rank"])].tolist())
                ntrials.extend(np.asarray(blob["n_trials_test"])[np.isfinite(blob["n_trials_test"])].tolist())
            per_mouse=[]
            for mouse in mice:
                vals=[values[s.session_uid] for s in sessions if s.mouse_name==mouse and s.session_uid in values]
                if vals: per_mouse.append(float(np.mean(vals)))
            history.append(dict(schema=version,estimator=estimator,fit=transform,native_r2_mean=float(np.mean(per_mouse)),sessions=len(values),median_rank=float(np.median(ranks)),median_test_trials=float(np.median(ntrials))))
pd.DataFrame(history).to_csv(OUTPUT / "history.csv",index=False)
print("HISTORY",pd.DataFrame(history).round(5).to_string(index=False),flush=True)
# Independently refit representative models at saved alpha/rank. No new optimization.
audit=[]
for estimator in ("gaussian","least_squares"):
    arrays=results.sel(keys=["r2_test_raw","n_neurons_target"],squeeze_ones=False,gain_estimator=estimator,gain_transform="raw",trial_shift=0)
    candidates=np.argwhere(np.isfinite(arrays["r2_test_raw"]) & (arrays["n_neurons_target"]<=350))
    if not len(candidates): candidates=np.argwhere(np.isfinite(arrays["r2_test_raw"]))
    si,slot=map(int,candidates[len(candidates)//2]); session=sessions[si]
    blobs={}
    for transform in ("raw","sqrt"):
        cfg=GainRegressionConfig(gain_estimator=estimator,gain_transform=transform)
        blob=store.get(session.session_uid,cfg)
        blobs[transform]=blob
        gain=blob["gain_matrices"][slot]
        tensors=split_tensors(gain,blob["source_rows"][slot],blob["target_rows"][slot],blob["trial_columns"][slot])
        model=ReducedRankRegression(alpha=float(blob["rrr_alpha"][slot]),fit_intercept=True).fit(*tensors["train"])
        rank=int(blob["rrr_rank"][slot])
        record=dict(estimator=estimator,fit=transform,session=session.session_uid,slot=slot,rank=rank,checks={})
        for scale in ("raw","sqrt"):
            for split in ("val","test"):
                x,y=tensors[split]; pred=model.predict(x,rank=rank,nonnegative=True).numpy(); truth=y.numpy()
                if transform!=scale:
                    convert=np.square if scale=="raw" else np.sqrt
                    pred,truth=convert(pred),convert(truth)
                sse=float(np.sum((truth-pred)**2)); sst=float(np.sum((truth-truth.mean())**2))
                manual=1-sse/sst
                stored=float(blob[f"r2_{split}_{scale}"][slot])
                np.testing.assert_allclose(manual,stored,rtol=1e-7,atol=1e-8)
                np.testing.assert_allclose(np.mean((truth-pred)**2),blob[f"mse_{split}_{scale}"][slot],rtol=1e-7,atol=1e-8)
                record["checks"][f"{split}_{scale}"]=dict(manual_r2=manual,stored_r2=stored,sst=sst,n=truth.size)
        env=int(blob["env_slot_ids"][slot])
        rng=np.random.default_rng(_seed(session.session_uid,env,cfg.split_seed,"null"))
        nulls={scale:[] for scale in ("raw","sqrt")}; denominators={scale:[] for scale in nulls}
        for _ in range(cfg.n_null_rolls):
            rolled=roll_gain_rows(gain,rng)
            ts=split_tensors(rolled,blob["source_rows"][slot],blob["target_rows"][slot],blob["trial_columns"][slot])
            m=ReducedRankRegression(alpha=float(blob["rrr_alpha"][slot]),fit_intercept=True).fit(*ts["train"])
            for scale in nulls:
                nulls[scale].append(score_rrr(m,rank,*ts["test"],gain_transform=transform,score_scale=scale)[1])
                target=ts["test"][1].numpy()
                if transform!=scale: target=(np.square if scale=="raw" else np.sqrt)(target)
                denominators[scale].append(float(np.sum((target-target.mean())**2)))
        for scale in nulls:
            np.testing.assert_allclose(np.mean(nulls[scale]),blob[f"r2_test_{scale}_null"][slot],rtol=1e-7,atol=1e-8)
            real_sst=record["checks"][f"test_{scale}"]["sst"]
            record["checks"][f"null_{scale}"]=dict(manual_r2=float(np.mean(nulls[scale])),sst_ratio_min=float(np.min(denominators[scale])/real_sst),sst_ratio_max=float(np.max(denominators[scale])/real_sst))
        # The two adjacent-lag fits must use exactly the same target test values.
        for lag in (-1,1):
            other=store.get(session.session_uid,replace(cfg,trial_shift=lag))
            np.testing.assert_array_equal(other["gain_matrices"][slot],gain)
            for name in ("train","val","test"):
                np.testing.assert_array_equal(other["trial_columns"][slot][name],blob["trial_columns"][slot][name])
            np.testing.assert_array_equal(other["target_rows"][slot],blob["target_rows"][slot])
        audit.append(record)
        print("AUDITED",estimator,transform,session.session_uid,slot,flush=True)
    raw,root=blobs["raw"],blobs["sqrt"]
    for key in ("source_rows","target_rows","gain_roi_indices"):
        np.testing.assert_array_equal(raw[key][slot],root[key][slot])
    np.testing.assert_allclose(np.square(root["gain_matrices"][slot]),raw["gain_matrices"][slot],rtol=1e-12,atol=1e-12)
(OUTPUT / "audit.json").write_text(json.dumps(dict(schema=GainRegressionConfig().schema_version,schema_rows=dict(Counter(r["schema_version"] for r in metadata)),coverage=coverage,audit=audit),indent=2),encoding="utf-8")
print("OUTPUT",OUTPUT,flush=True)
