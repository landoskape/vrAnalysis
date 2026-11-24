"""Summarize and plot behavior-selected same-cell neural comparisons."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

OUTPUT=Path(__file__).parent/"behavioral_engagement_inspection"


def run():
    screen=pd.read_csv(OUTPUT/"behavior_screen.csv")
    manifest=pd.read_csv(OUTPUT/"neural_manifest.csv")
    rows=[];trimrows=[];motionrows=[];coverage=[]
    for entry in manifest.to_dict("records"):
        path=OUTPUT/"neural"/entry["session_key"]
        if not (path/"complete.json").exists():continue
        info=json.loads((path/"complete.json").read_text())
        if (path/"trial_coverage.csv").exists():coverage.append(pd.read_csv(path/"trial_coverage.csv"))
        if info.get("no_eligible_environments"):continue
        cells=pd.read_csv(path/"per_roi_comparison.csv")
        for (env,mode),table in cells.groupby(["environment","mode"]):
            active=table.calibration_fraction_active>.1
            quality=active&(table.calibration_reliability>.3)
            subsets={"somatic_active":active&(table.cell_class=="c"),"somatic_quality":quality&(table.cell_class=="c"),
                     "dendritic_quality":quality&(table.cell_class=="d"),"all_quality":quality}
            for subset,mask in subsets.items():
                t=table[mask]
                if len(t)<10:continue
                row=dict(**entry,environment=env,mode=mode,subset=subset,n_cells=len(t),n_trials=int(t.n_evaluation.iloc[0]))
                metrics=["reliability_before","reliability_after","template_overlap","calibration_template_before","calibration_template_after",
                         "split_half_before","split_half_after","nonreward_reliability_before","nonreward_reliability_after",
                         "mean_activity_before","mean_activity_after"]
                for metric in metrics:
                    row[metric]=float(t[metric].mean())
                for prefix in ("reliability","calibration_template","split_half","nonreward_reliability","mean_activity"):
                    row[prefix+"_delta"]=float((t[prefix+"_after"]-t[prefix+"_before"]).mean())
                row["fraction_cells_reliability_decreased"]=float(np.mean(t.reliability_after<t.reliability_before))
                rows.append(row)
        full=pd.read_csv(path/"full_trim_per_roi.csv")
        for env,t in full[full.original_quality].groupby("environment"):
            trimrows.append(dict(**entry,environment=env,n_cells=len(t),full_reliability=float(t.full_reliability.mean()),
                                 trim_reliability=float(t.trim_reliability.mean()),delta=float((t.trim_reliability-t.full_reliability).mean())))
        motion=pd.read_csv(path/"motion_summary.csv")
        before=motion[motion.condition=="before"].set_index("plane");after=motion[motion.condition=="after"].set_index("plane")
        motionrows.append(dict(**entry,corrXY_before=float(before.corrXY_mean.mean()),corrXY_after=float(after.corrXY_mean.mean()),
                               max_plane_corrXY_change=float(np.max(np.abs(after.corrXY_mean-before.corrXY_mean))),
                               max_plane_x_shift=float(np.max(np.abs(after.xoff_mean-before.xoff_mean))),
                               max_plane_y_shift=float(np.max(np.abs(after.yoff_mean-before.yoff_mean))),
                               mean_badframes_before=float(before.badframes_mean.mean()),mean_badframes_after=float(after.badframes_mean.mean())))
    env_summary=pd.DataFrame(rows)
    env_summary.to_csv(OUTPUT/"neural_environment_summary.csv",index=False)
    trim=pd.DataFrame(trimrows);trim.to_csv(OUTPUT/"trim_summary.csv",index=False)
    pd.DataFrame(motionrows).to_csv(OUTPUT/"neural_motion_summary.csv",index=False)
    if coverage:pd.concat(coverage,ignore_index=True).to_csv(OUTPUT/"neural_trial_coverage.csv",index=False)
    metric_columns=[c for c in env_summary if c not in ("session_key","role","cut","reference","environment","mode","subset","n_cells","n_trials")]
    session_summary=env_summary.groupby(["session_key","role","mode","subset"],as_index=False)[metric_columns].mean()
    support=env_summary.groupby(["session_key","mode","subset"]).agg(n_environments=("environment","size"),mean_cells=("n_cells","mean"),minimum_trials=("n_trials","min")).reset_index()
    session_summary=session_summary.merge(support,on=["session_key","mode","subset"])
    session_summary=session_summary.merge(screen[["session_key","mouse","date","discovery","candidate","onset_fraction","drop","q_fdr"]],on="session_key")
    session_summary.to_csv(OUTPUT/"neural_session_summary.csv",index=False)
    primary=session_summary[(session_summary["mode"]=="matched_count")&(session_summary.subset=="somatic_quality")]
    print("PRIMARY SESSION SUMMARY\n",primary[["session_key","role","n_environments","minimum_trials","reliability_before","reliability_after","reliability_delta","calibration_template_delta","split_half_delta","nonreward_reliability_delta"]].to_string(index=False),flush=True)
    group_rows=[]
    for (role,mode,subset),group in session_summary.groupby(["role","mode","subset"]):
        row=dict(role=role,mode=mode,subset=subset,n_sessions=len(group),n_mice=group.mouse.nunique())
        for metric in ("reliability_before","reliability_after","reliability_delta","calibration_template_delta","split_half_delta","nonreward_reliability_delta"):
            row[metric+"_session_mean"]=float(group[metric].mean())
            row[metric+"_mouse_mean"]=float(group.groupby("mouse")[metric].mean().mean())
        row["n_sessions_reliability_drop"]=int(np.sum(group.reliability_delta<0))
        group_rows.append(row)
    pd.DataFrame(group_rows).to_csv(OUTPUT/"neural_group_summary.csv",index=False)
    print("GROUP SUMMARY\n",pd.DataFrame(group_rows).query("mode=='matched_count' and subset=='somatic_quality'").to_string(index=False),flush=True)

    fig,axes=plt.subplots(2,3,figsize=(14,8),layout="constrained")
    colors={"screen_positive":"#cc4755","sham_control":"#347faf","borderline_exploratory":"#c39535"}
    for role,g in primary.groupby("role"):
        for _,r in g.iterrows():
            axes[0,0].plot([0,1],[r.reliability_before,r.reliability_after],color=colors[role],alpha=.6,marker=".")
        axes[0,0].plot([],[],color=colors[role],label=role.replace("_"," "))
    axes[0,0].set_xticks([0,1],["Before","After"]);axes[0,0].set_ylabel("Same-cell spatial reliability")
    axes[0,0].legend(frameon=False,fontsize=8)
    order=["screen_positive","borderline_exploratory","sham_control"]
    rng=np.random.default_rng(22)
    for i,role in enumerate(order):
        g=primary[primary.role==role]
        axes[0,1].scatter(i+rng.uniform(-.12,.12,len(g)),g.reliability_delta,color=colors[role],s=30)
        axes[0,1].plot([i-.2,i+.2],[g.reliability_delta.mean()]*2,color="black",lw=2)
        axes[0,2].scatter(i+rng.uniform(-.12,.12,len(g)),g.calibration_template_delta,color=colors[role],s=30)
    for ax in axes[0,1:]:
        ax.axhline(0,color="0.7",ls=":");ax.set_xticks(range(3),["Screen +","Borderline","Sham controls"])
    axes[0,1].set_ylabel("Reliability change (after − before)")
    axes[0,2].set_ylabel("Change in independent PF-template correlation")
    for role,g in primary.groupby("role"):
        axes[1,0].scatter(g["drop"],g.reliability_delta,color=colors[role],label=role.replace("_"," "))
        axes[1,1].scatter(g.split_half_delta,g.reliability_delta,color=colors[role])
    axes[1,0].axhline(0,color="0.7",ls=":");axes[1,0].set_xlabel("Reward-slowing drop (behavior-only fit)");axes[1,0].set_ylabel("Same-cell reliability change")
    axes[1,1].axhline(0,color="0.7",ls=":");axes[1,1].axvline(0,color="0.7",ls=":")
    axes[1,1].set_xlabel("Within-period split-half correlation change");axes[1,1].set_ylabel("Reliability change")
    for role,g in trim.merge(screen[["session_key","mouse"]],on="session_key").groupby("role"):
        t=g.groupby("session_key").delta.mean()
        x=order.index(role)
        axes[1,2].scatter(x+rng.uniform(-.12,.12,len(t)),t,color=colors[role])
    axes[1,2].axhline(0,color="0.7",ls=":");axes[1,2].set_xticks(range(3),["Screen +","Borderline","Sham controls"])
    axes[1,2].set_ylabel("Trim − full reliability; original quality cells")
    for ax in axes.flat:ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUTPUT/"neural_summary.png",dpi=150);plt.close(fig)

    examples=["ATL012_2023-03-01_701","ATL045_2024-02-01_701","ATL049_2024-06-05_701","ATL012_2023-02-28_701"]
    fig,axes=plt.subplots(len(examples),3,figsize=(14,11),layout="constrained")
    for row,key in enumerate(examples):
        path=OUTPUT/"neural"/key
        if not (path/"complete.json").exists():continue
        behavior=pd.read_csv(OUTPUT/"behavior_maps"/f"{key}.csv")
        info=json.loads((path/"complete.json").read_text());cut=int(info["cut"])
        templates=pd.read_csv(path/"trial_template_overlap.csv")
        for env,g in behavior.groupby("environment"):
            axes[row,0].plot(g.trial,g.reward_slowdown.rolling(10,center=True,min_periods=4).mean(),color={1:"black",2:"blue",3:"blue",4:"green"}.get(env),label=f"Env {env}")
        axes[row,0].axvline(cut+1,color="red",ls=":");axes[row,0].axhline(0,color="0.7",lw=.6)
        axes[row,0].set_title(key);axes[row,0].set_ylabel("Reward-related slowing");axes[row,0].set_xlabel("Trial")
        axes[row,0].legend(fontsize=8,frameon=False)
        for env,g in templates[~templates.calibration].groupby("environment"):
            axes[row,1].plot(g.trial,g.mean_calibration_template_correlation.rolling(5,center=True,min_periods=3).mean(),color={1:"black",2:"blue",3:"blue",4:"green"}.get(env),label=f"Env {env}")
        axes[row,1].axvline(cut+1,color="red",ls=":");axes[row,1].set_ylabel("Held-out PF-template agreement");axes[row,1].set_xlabel("Trial")
        t=env_summary[(env_summary.session_key==key)&(env_summary["mode"]=="matched_count")&(env_summary.subset=="somatic_quality")]
        for _,r in t.iterrows():axes[row,2].plot([0,1],[r.reliability_before,r.reliability_after],marker="o",color={1:"black",2:"blue",3:"blue",4:"green"}.get(r.environment),label=f"Env {r.environment}; n={r.n_cells}")
        axes[row,2].set_xticks([0,1],["Before","After"]);axes[row,2].set_ylabel("Same-cell reliability; matched trial count")
        axes[row,2].legend(fontsize=8,frameon=False)
    for ax in axes.flat:ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUTPUT/"behavior_neural_examples.png",dpi=140);plt.close(fig)


if __name__=="__main__":run()
