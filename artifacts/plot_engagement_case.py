"""Compact figure for the motivating ATL012 example."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

root=Path(__file__).parent/"behavioral_engagement_inspection"
key="ATL012_2023-03-01_701"
b=pd.read_csv(root/"behavior_maps"/f"{key}.csv")
t=pd.read_csv(root/"neural"/key/"trial_template_overlap.csv")
s=pd.read_csv(root/"neural_environment_summary.csv")
s=s[(s.session_key==key)&(s["mode"]=="matched_count")&(s.subset=="somatic_quality")]
a=pd.read_csv(root/"alternative_signal_comparison.csv");a=a[a.session_key==key]
colors={1:"black",3:"blue",4:"green"}
fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
for env,g in b.groupby("environment"):
    axes[0,0].plot(g.trial,g.reward_slowdown.rolling(10,center=True,min_periods=4).mean(),color=colors[env],label=f"Env {env}")
for env,g in t[~t.calibration].groupby("environment"):
    axes[0,1].plot(g.trial,g.mean_calibration_template_correlation.rolling(5,center=True,min_periods=3).mean(),color=colors[env])
for ax in axes[0]:
    ax.axvline(167,color="#c83c47",ls=":",label="Behavior split: trial 167")
    ax.axvline(196,color="0.5",ls="--",label="Descriptive neural split: trial 196")
    ax.set_xlabel("Chronological trial")
axes[0,0].axhline(0,color="0.7",lw=.7)
axes[0,0].set_ylabel("Pre-reward slowing index")
axes[0,1].set_ylabel("Agreement with earlier PF template")
axes[0,0].legend(fontsize=8,frameon=False)
for _,r in s.iterrows():
    axes[1,0].plot([0,1],[r.reliability_before,r.reliability_after],marker="o",color=colors[r.environment],label=f"Env {int(r.environment)}; {int(r.n_cells)} cells")
    axes[1,1].plot([0,1],[r.split_half_before,r.split_half_after],marker="o",color=colors[r.environment],label=f"Env {int(r.environment)}")
for ax in axes[1]:ax.set_xticks([0,1],["Before behavior split","After behavior split"])
axes[1,0].set_ylabel("Same-cell spatial reliability")
axes[1,1].set_ylabel("Within-period PF split-half correlation")
axes[1,0].legend(fontsize=8,frameon=False)
for ax in axes.flat:ax.spines[["right","top"]].set_visible(False)
fig.suptitle("ATL012 · 2023-03-01 · equal trial counts, independently selected somata",fontsize=12)
fig.savefig(root/"atl012_case_summary.png",dpi=160)
