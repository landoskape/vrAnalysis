"""Freeze and apply a behavior-only reward-approach change-point screen."""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

OUTPUT=Path(__file__).parent/"behavioral_engagement_inspection"
SPEC=dict(version=1,feature="(control_speed - prereward_speed)/(control_speed + prereward_speed)",
          prereward_window_track_fraction=[-.1,0],control_window_track_fraction=[-.225,-.125],
          minimum_trials=60,minimum_segment_trials=20,minimum_segment_fraction=.20,
          minimum_drop=.08,minimum_before_slowdown=.10,maximum_after_to_before_ratio=.5,
          null_block_trials=10,permutations=1999,seed=20261109,fdr=.05,
          discovery_sessions=["ATL012_2023-02-27_701","ATL012_2023-02-28_701","ATL012_2023-03-01_701","ATL012_2023-03-03_701"],
          notes="Behavior-only exploratory screen, not a validated engagement classifier. No neural outcomes used.")


def fit(table,permutations=1999):
    n=len(table)
    finite=np.isfinite(table.reward_slowdown.to_numpy())
    y=table.reward_slowdown.to_numpy()[finite]
    env=table.environment.to_numpy()[finite]
    original=np.flatnonzero(finite)
    if len(y)<SPEC["minimum_trials"]:
        return dict(status="insufficient_trials")
    minimum=max(SPEC["minimum_segment_trials"],int(np.ceil(SPEC["minimum_segment_fraction"]*n)))
    cuts=np.arange(minimum,n-minimum+1)
    z=(original[None,:]>=cuts[:,None]).astype(float)
    residual=y.copy()
    for e in np.unique(env):
        selected=env==e
        residual[selected]-=np.mean(y[selected])
        z[:,selected]-=np.mean(z[:,selected],axis=1,keepdims=True)
    norm=np.sum(z*z,axis=1)
    good=norm>1
    cuts,z,norm=cuts[good],z[good],norm[good]
    if not len(cuts):
        return dict(status="environment_time_confounded")
    covariance=z@residual
    improvement=np.minimum(covariance,0)**2/norm
    best=int(np.argmax(improvement))
    cut=int(cuts[best]); beta=float(covariance[best]/norm[best])
    # Block-order permutations preserve short within-session temporal dependence.
    rng=np.random.default_rng(SPEC["seed"])
    blocks=[np.arange(a,min(a+SPEC["null_block_trials"],len(y))) for a in range(0,len(y),SPEC["null_block_trials"])]
    null=np.empty((len(y),permutations))
    for b in range(permutations):
        index=np.concatenate([blocks[k] for k in rng.permutation(len(blocks))])
        null[:,b]=residual[index]
    null_cov=z@null
    null_max=np.max(np.minimum(null_cov,0)**2/norm[:,None],axis=0)
    p=float((1+np.sum(null_max>=improvement[best]))/(permutations+1))
    pre=table.iloc[:cut]; post=table.iloc[cut:]
    # Balance the descriptive before/after averages across environments.
    differences=[]; before=[]; after=[]
    for e in np.unique(env):
        a=pre.loc[pre.environment==e,"reward_slowdown"].dropna()
        b=post.loc[post.environment==e,"reward_slowdown"].dropna()
        if len(a)>=5 and len(b)>=5:
            before.append(float(a.mean()));after.append(float(b.mean()));differences.append(float(a.mean()-b.mean()))
    if not before:
        return dict(status="insufficient_within_environment_support")
    early=float(np.mean(before));late=float(np.mean(after))
    # Compare step and linear trend fits without using neural measurements.
    time=table.start_minutes.to_numpy()[finite].copy()
    for e in np.unique(env):
        time[env==e]-=np.mean(time[env==e])
    linear_gain=float(np.dot(time,residual)**2/np.dot(time,time))
    total=float(np.dot(residual,residual))
    return dict(status="fit",cut_after_trial=cut,onset_trial=cut+1,onset_minutes=float(table.iloc[cut].start_minutes),
                onset_fraction=cut/n,drop=-beta,before_slowdown=early,after_slowdown=late,
                environment_drops=json.dumps(differences),supported_environments=len(differences),
                p_block_permutation=p,variance_explained=float(improvement[best]/total) if total else 0,
                step_minus_linear_gain=float((improvement[best]-linear_gain)/total) if total else 0,
                effect_pass=bool(-beta>=SPEC["minimum_drop"] and early>=SPEC["minimum_before_slowdown"] and late<=early*SPEC["maximum_after_to_before_ratio"]),
                post_mean_speed=float(post.mean_speed.mean()),pre_mean_speed=float(pre.mean_speed.mean()),
                pre_lick_count=int(pre.preconsumption_licks.sum()),post_lick_count=int(post.preconsumption_licks.sum()))


def run():
    parser=argparse.ArgumentParser();parser.add_argument("--all",action="store_true");args=parser.parse_args()
    specfile=OUTPUT/"behavior_rule_v1.json"
    if specfile.exists():
        assert json.loads(specfile.read_text())==SPEC,"Rule changed after initial discovery"
    else:
        specfile.write_text(json.dumps(SPEC,indent=2))
    data=pd.read_csv(OUTPUT/("all_trial_behavior.csv" if args.all else "discovery_trial_behavior.csv"))
    rows=[]
    for (mouse,date,sid),table in data.groupby(["mouse","date","session_id"],sort=True):
        table=table.sort_values("trial").reset_index(drop=True)
        key=f"{mouse}_{date}_{sid}"
        result=fit(table,SPEC["permutations"])
        row=dict(mouse=mouse,date=date,session_id=str(sid),session_key=key,n_trials=len(table),discovery=key in SPEC["discovery_sessions"],**result)
        rows.append(row)
        print("SCREEN",key,result,flush=True)
    results=pd.DataFrame(rows)
    good=results.status=="fit"
    p=results.loc[good,"p_block_permutation"].to_numpy()
    order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    results.loc[good,"q_fdr"]=q
    results["candidate"]=good & results.effect_pass.fillna(False) & (results.q_fdr<=SPEC["fdr"])
    results.to_csv(OUTPUT/("behavior_screen.csv" if args.all else "discovery_screen.csv"),index=False)
    print("CANDIDATES\n",results[results.candidate].to_string(index=False),flush=True)
    if args.all:
        selected=results[results.candidate].sort_values(["q_fdr","drop"],ascending=[True,False]).head(8).session_key.tolist()
        selected=list(dict.fromkeys([SPEC["discovery_sessions"][2],*selected]))
    else:
        selected=SPEC["discovery_sessions"]
    fig,axes=plt.subplots(len(selected),3,figsize=(14,3*len(selected)),layout="constrained",squeeze=False)
    for row,key in enumerate(selected):
        table=pd.read_csv(OUTPUT/"behavior_maps"/f"{key}.csv")
        maps=np.load(OUTPUT/"behavior_maps"/f"{key}.npz")
        screen=results[results.session_key==key].iloc[0]
        ax=axes[row,0]
        for e,g in table.groupby("environment"):
            ax.plot(g.trial,g.reward_slowdown.rolling(10,center=True,min_periods=5).mean(),label=f"Env {e}")
        ax.set_title(key+f" | q={screen.q_fdr:.3g}")
        ax.set_ylabel("Pre-reward slowing index")
        ax.axhline(0,color="0.7",lw=.7)
        if screen.status=="fit": ax.axvline(screen.onset_trial,color="red",ls=":")
        ax.legend(fontsize=8,frameon=False)
        ax.set_xlabel("Chronological trial")
        ax=axes[row,1]
        ax.imshow(maps["speed"],aspect="auto",origin="lower",extent=[0,1,1,len(table)],vmin=0,vmax=60,cmap="viridis")
        reward_frac=maps["reward_start"]/(maps["position_centers"][-1]+np.diff(maps["position_centers"])[0]/2)
        ax.scatter(reward_frac,np.arange(1,len(table)+1),s=1,c="white")
        if screen.status=="fit": ax.axhline(screen.onset_trial,color="red",ls=":")
        ax.set_xlabel("Fraction of track; white = reward onset")
        ax.set_ylabel("Chronological trial")
        ax=axes[row,2]
        cut=int(screen.cut_after_trial) if screen.status=="fit" else len(table)//2
        for e,g in table.groupby("environment"):
            ids=g.index.to_numpy()
            for late,style in ((False,"-"),(True,"--")):
                keep=ids[(ids>=cut)==late]
                if len(keep):
                    speed=maps["speed"][keep]
                    normalized=speed/np.nanmean(speed,axis=1,keepdims=True)
                    ax.plot(maps["position_centers"]/(maps["position_centers"][-1]+np.diff(maps["position_centers"])[0]/2),np.nanmean(normalized,axis=0),ls=style,label=f"{e} {'after' if late else 'before'}")
        ax.set_ylabel("Speed / trial mean")
        ax.set_xlabel("Fraction of track")
        ax.legend(fontsize=7,ncol=2,frameon=False)
    fig.savefig(OUTPUT/("behavior_examples.png" if args.all else "discovery_examples.png"),dpi=130)
    plt.close(fig)


if __name__=="__main__":run()
