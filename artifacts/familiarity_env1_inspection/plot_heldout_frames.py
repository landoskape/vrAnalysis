"""Plot cached held-out frames for every mouse/session/environment; never fit models."""
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vrAnalysis.database import get_database
from dimensionality_manuscript import RegressionConfig, ResultsAggregator, ResultsStore
from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript.figure_scripts.figure2._scores import chronological_mouse_sessions
from dimensionality_manuscript.env_order import load_env_order


class ReadOnlyStore(ResultsStore):
    def __init__(self):
        self.db_path = RegistryPaths.pipeline_v2_db_path
        self._blob_cache = {}
        self._blob_cache_maxsize = 0

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path.as_uri() + "?mode=ro", uri=True)
        try:
            yield conn
        finally:
            conn.close()


output = Path(__file__).parent
sessions = get_database("vrSessions").iter_sessions(imaging=True, experimentType="Blender VR")
results = ResultsAggregator(RegressionConfig(), ReadOnlyStore(), sessions, keys=["num_frames_env", "env_slot_ids"])
pf = results.sel(model_name="external_placefield_1d", avg_by_mouse=False)
residual = results.sel(model_name="internal_placefield_1d_structured_additive", avg_by_mouse=False)
order = load_env_order()
rows = []
for mouse in sorted(results.unique_mice):
    counters = [0] * len(order[mouse])
    for overall_session, index in enumerate(chronological_mouse_sessions(results, mouse)):
        for slot, env_id in enumerate(order[mouse]):
            present = env_id in sessions[index].environments
            n_pf = float(pf["num_frames_env"][index, slot])
            n_residual = float(residual["num_frames_env"][index, slot])
            cached_env_id = pf["env_slot_ids"][index, slot]
            assert not np.isfinite(cached_env_id) or int(cached_env_id) == env_id
            assert not np.isfinite(n_pf) or (n_pf >= 0 and n_pf == int(n_pf))
            rows.append(dict(
                mouse=mouse, session=results.session_ids[index], date=str(sessions[index].date),
                overall_session=overall_session, environment_slot=slot + 1, environment_id=env_id,
                ran_environment=present, environment_session=counters[slot] if present else np.nan,
                heldout_frames_pf=n_pf, heldout_frames_residual=n_residual,
            ))
            if present:
                counters[slot] += 1
table = pd.DataFrame(rows)
table.to_csv(output / "all_heldout_frames.csv", index=False)
visits = table[table.ran_environment].copy()
assert len(set(table.session)) == len(sessions)
same = np.isclose(table.heldout_frames_pf, table.heldout_frames_residual, equal_nan=True)
print(f"Mice={table.mouse.nunique()}, sessions={table.session.nunique()}, visited env-session combinations={len(visits)}", flush=True)
print(f"PF/Residual matching counts: {int(same.sum())}/{len(table)}", flush=True)
print("Counts for visited combinations:", visits.heldout_frames_pf.describe().to_dict(), flush=True)
print("Missing cached counts:", int(visits.heldout_frames_pf.isna().sum()), flush=True)
print("Zero-frame visited combinations:", int((visits.heldout_frames_pf == 0).sum()), flush=True)
print("Fewest frames:\n" + visits.nsmallest(12, "heldout_frames_pf").to_string(index=False), flush=True)
visits.groupby(["mouse", "environment_slot", "environment_id"]).heldout_frames_pf.agg(
    ["count", "min", "median", "max"],
).to_csv(output / "heldout_frames_by_mouse_environment.csv")


def draw_heatmap(value_column, coordinate, suffix, title):
    row_keys = list(visits.groupby(["mouse", "environment_slot", "environment_id"], sort=True).groups)
    width = int(visits[coordinate].max()) + 1
    data = np.full((len(row_keys), width), np.nan)
    missing = np.zeros_like(data, dtype=bool)
    row_lookup = {key: i for i, key in enumerate(row_keys)}
    for row in visits.itertuples():
        i = row_lookup[(row.mouse, row.environment_slot, row.environment_id)]
        j = int(getattr(row, coordinate))
        data[i, j] = getattr(row, value_column)
        missing[i, j] = pd.isna(getattr(row, value_column))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")
    norm = Normalize(0, float(np.nanmax(data)))
    fig, ax = plt.subplots(figsize=(17, max(10, len(row_keys) * 0.36 + 2.2)))
    fig.subplots_adjust(left=0.23, right=0.93, bottom=0.09, top=0.88)
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    labels = [f"{mouse}   |   Env #{slot} (ID {env})" for mouse, slot, env in row_keys]
    ax.set_yticks(range(len(row_keys)), labels, fontsize=9)
    slot_colors = ["#111111", "#2458ce", "#168747"]
    for label, key in zip(ax.get_yticklabels(), row_keys):
        label.set_color(slot_colors[key[1] - 1])
    ax.set_xticks(range(width), range(width), fontsize=10)
    ax.set_xlabel(
        "Environment session # (zero-based, counted separately for each mouse and environment)"
        if coordinate == "environment_session" else "Overall session # (zero-based, within mouse)",
        fontsize=12, labelpad=12,
    )
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_keys), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="y", length=0)
    for i, j in np.ndindex(data.shape):
        if np.isfinite(data[i, j]):
            rgba = cmap(norm(data[i, j]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, str(int(data[i, j])), ha="center", va="center", fontsize=7.8,
                    color="white" if luminance < 0.48 else "#151515")
        elif missing[i, j]:
            ax.text(j, i, "?", color="#be123c", ha="center", va="center", fontsize=9)
    for i in range(1, len(row_keys)):
        if row_keys[i][0] != row_keys[i - 1][0]:
            ax.axhline(i - 0.5, color="white", linewidth=3)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.018)
    colorbar.set_label("Held-out frames", fontsize=11)
    fig.suptitle(title, fontsize=19, x=0.23, ha="left", y=0.97)
    fig.text(0.23, 0.935, f"{table.mouse.nunique()} mice · {table.session.nunique()} sessions · {len(visits)} environment-session observations"
             "\nEach cell is an exact frame count. Env # follows experience order; ID is the actual environment."
             "\nGray = no such environment session" + ("; ? = missing cached result" if missing.any() else "")
             + ("" if coordinate == "environment_session" else " / environment not run"), fontsize=10, va="top")
    fig.savefig(output / f"heldout_frames_{suffix}.png", dpi=160, facecolor="white")
    fig.savefig(output / f"heldout_frames_{suffix}.pdf", facecolor="white")
    plt.close(fig)


draw_heatmap("heldout_frames_pf", "environment_session", "by_environment_session", "Held-out frames for every mouse and environment session")
draw_heatmap("heldout_frames_pf", "overall_session", "by_overall_session", "Held-out frames aligned to each mouse's overall sessions")
if not same.all():
    print("MISMATCHES\n" + table.loc[~same].to_string(index=False), flush=True)
    draw_heatmap("heldout_frames_residual", "environment_session", "residual_by_environment_session", "Residual model: held-out frames by mouse and environment session")
print("OUTPUT", output, flush=True)
