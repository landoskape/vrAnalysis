"""Scatter cached per-environment external PF R2 against its held-out frame count."""
from pathlib import Path
import pickle
import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dimensionality_manuscript import RegressionConfig
from dimensionality_manuscript.registry import RegistryPaths

output = Path(__file__).parent
counts = pd.read_csv(output / "all_heldout_frames.csv")
visits = counts[counts.ran_environment].copy()
cfg = RegressionConfig(model_name="external_placefield_1d")
db_path = RegistryPaths.pipeline_v2_db_path
with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as conn:
    entries = dict(conn.execute("SELECT session_id, result_uid FROM results WHERE analysis_key=?", (cfg.key(),)).fetchall())
cache = {}
scores = []
for row in visits.itertuples():
    if row.session not in cache:
        uid = entries[row.session]
        cache[row.session] = pickle.loads((db_path.parent / "blobs" / f"{uid}.pkl").read_bytes())
    result = cache[row.session]
    slot = row.environment_slot - 1
    assert float(result["num_frames_env"][slot]) == row.heldout_frames_pf
    assert int(result["env_slot_ids"][slot]) == row.environment_id
    scores.append(float(result["r2_env"][slot]))
visits["external_pf_r2"] = scores
visits.to_csv(output / "external_pf_r2_vs_frames.csv", index=False)
finite = visits[np.isfinite(visits.external_pf_r2) & np.isfinite(visits.heldout_frames_pf)].copy()
assert not (finite.heldout_frames_pf <= 0).any()
print(f"Plotted {len(finite)} observations from {finite.mouse.nunique()} mice and {finite.session.nunique()} sessions.", flush=True)
print(f"Omitted {len(visits) - len(finite)} observations with nonfinite R2.", flush=True)
print(finite[["heldout_frames_pf", "external_pf_r2"]].describe().to_string(), flush=True)
print("Lowest R2:\n" + finite.nsmallest(5, "external_pf_r2").to_string(index=False), flush=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(9, 6.2), layout="constrained")
colors = {1: "#181818", 2: "#265adb", 3: "#148447"}
for slot, group in finite.groupby("environment_slot"):
    ax.scatter(group.heldout_frames_pf, group.external_pf_r2, s=35, color=colors[slot],
               alpha=0.7, edgecolors="white", linewidths=0.4, label=f"Env #{slot} (n={len(group)})")
ax.axhline(0, color="#9ca3af", linestyle="--", linewidth=0.9, zorder=0)
ax.set_xlim(left=0)
ax.set_xlabel("Number of held-out frames", labelpad=9)
ax.set_ylabel("External PF R²", labelpad=9)
ax.set_title("External place-field model: R² versus held-out frames", loc="left", fontsize=16, pad=27)
ax.text(0, 1.025, f"Each point is one mouse × session × environment · {len(finite)} observations · {finite.mouse.nunique()} mice",
        transform=ax.transAxes, fontsize=10, color="#4b5563")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", linewidth=0.65, zorder=0)
ax.set_axisbelow(True)
ax.legend(title="Environment experience order", frameon=False, loc="upper right", fontsize=10, title_fontsize=10)
for extension in ["png", "pdf"]:
    fig.savefig(output / f"external_pf_r2_vs_frames.{extension}", dpi=180, facecolor="white")
print("OUTPUT", output / "external_pf_r2_vs_frames.png", flush=True)
