"""Inspect cached scores only; open the results database read-only, never fit models."""
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vrAnalysis.database import get_database
from dimensionality_manuscript import RegressionConfig, ResultsAggregator, ResultsStore
from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript.figure_scripts.figure2 import RegressionFamiliarityByEnvViewer
from dimensionality_manuscript.figure_scripts.figure2._scores import chronological_mouse_sessions


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
store = ReadOnlyStore()
sessions = get_database("vrSessions").iter_sessions(imaging=True, experimentType="Blender VR")
print(f"Sessions: {len(sessions)}", flush=True)
metadata = store.summary_table(as_dataframe=True, analysis_type="regression")
print("Stored schemas:", metadata.groupby("schema_version").size().to_dict(), flush=True)
results = ResultsAggregator(RegressionConfig(), store, sessions, keys=["r2_env", "num_frames_env", "env_slot_ids"])
viewer = RegressionFamiliarityByEnvViewer(
    results, baseline_model="external_placefield_1d", reference_model="internal_placefield_1d_structured_additive",
    comparison_mode="ratio", reference_mode="reference_mean_error", min_mice=3, show_zero_line=False,
    figsize=(4.5, 1.75), fontsize=7, legend_options={"visible": False},
)
details = results.sel(model_name="external_placefield_1d", avg_by_mouse=False)
rows = []
for mouse in results.unique_mice:
    indices = chronological_mouse_sessions(results, mouse)
    indices = indices[np.isfinite(viewer._baseline_scores[indices, 0])]
    for x, index in enumerate(indices):
        baseline = float(viewer._baseline_scores[index, 0])
        reference = float(viewer._reference_scores[index, 0])
        rows.append(dict(
            mouse=mouse, environment_session=x, session=results.session_ids[index],
            date=str(sessions[index].date), environments=str(sessions[index].environments.tolist()),
            baseline_r2=baseline, reference_r2=reference,
            ratio=baseline / reference if reference != 0 else np.nan,
            heldout_frames=details["num_frames_env"][index, 0], environment_id=details["env_slot_ids"][index, 0],
        ))
table = pd.DataFrame(rows)
table.to_csv(output / "per_mouse_session_scores.csv", index=False)
summary = table.groupby("environment_session").agg(
    n_mice=("mouse", "size"), baseline_mean=("baseline_r2", "mean"), baseline_median=("baseline_r2", "median"),
    reference_mean=("reference_r2", "mean"), reference_median=("reference_r2", "median"),
    ratio_mean=("ratio", "mean"), ratio_median=("ratio", "median"),
)
summary.to_csv(output / "summary.csv")
print("\nCURVES\n" + summary.to_string(float_format=lambda x: f"{x:.6f}"), flush=True)
nearby = table[table.environment_session.between(4, 8)].sort_values(["mouse", "environment_session"])
print("\nNEARBY SESSIONS\n" + nearby.to_string(index=False, float_format=lambda x: f"{x:.6f}"), flush=True)
influence = []
for mouse in results.unique_mice:
    leave = table[table.mouse != mouse].groupby("environment_session").agg(
        baseline=("baseline_r2", "mean"), reference=("reference_r2", "mean"), ratio=("ratio", "mean"),
    )
    influence.append(dict(
        excluded_mouse=mouse,
        **{f"{metric}_at6": leave.loc[6, metric] for metric in ["baseline", "reference", "ratio"]},
        **{f"{metric}_change_5_to_6": leave.loc[6, metric] - leave.loc[5, metric] for metric in ["baseline", "reference", "ratio"]},
    ))
influence = pd.DataFrame(influence)
influence.to_csv(output / "leave_one_mouse_out.csv", index=False)
print("\nLEAVE ONE MOUSE OUT\n" + influence.to_string(index=False, float_format=lambda x: f"{x:.6f}"), flush=True)
print("\nAT X=6\n" + table[table.environment_session == 6].sort_values("baseline_r2").to_string(index=False), flush=True)
print("\nCOVERAGE FOR MICE THAT STOP CONTRIBUTING AT X=6", flush=True)
at5 = set(table.loc[table.environment_session == 5, "mouse"])
at6 = set(table.loc[table.environment_session == 6, "mouse"])
for mouse in sorted(at5 - at6):
    for index in chronological_mouse_sessions(results, mouse):
        print(results.session_ids[index], sessions[index].environments.tolist(), viewer._baseline_scores[index].tolist(), flush=True)
for statistic in ["mean", "median"]:
    viewer.set_parameter_value("statistic", statistic)
    fig = viewer.plot(viewer.state)
    fig.savefig(output / f"{statistic}_curves.png", dpi=180)
print("\nOUTPUT", output, flush=True)
