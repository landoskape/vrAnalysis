"""Render notebook viewers with temporary frame-count filters; never modify viewer code."""
from contextlib import contextmanager
from pathlib import Path
import ast
import hashlib
import json
import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from vrAnalysis.database import get_database
from dimensionality_manuscript import RegressionConfig, RegressionPlacefieldResidualConfig, ResultsAggregator, ResultsStore
from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript.pipeline.aggregate import average_by_mouse
from dimensionality_manuscript.figure_scripts.figure2 import (
    ModelPerformanceViewer, ModelPlacefieldResidualInsetViewer, RegressionFamiliarityByEnvViewer,
)

OUTPUT = Path(__file__).parent
THRESHOLD = 100
NOTEBOOK = ROOT / "dimensionality_manuscript/dim_regression_figures_refactor.ipynb"
VIEWER_FILES = [ROOT / f"dimensionality_manuscript/figure_scripts/figure2/{name}.py"
                for name in ["performance", "pf_residual", "familiarity", "_scores"]]
source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in VIEWER_FILES}


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


def notebook_kwargs(class_name):
    """Read just the constructor's literal kwargs, without executing notebook cells."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    matches = []
    for cell_number, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if f"viewer = {class_name}(" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == class_name:
                values = {}
                for kw in node.keywords:
                    if isinstance(kw.value, ast.Call) and isinstance(kw.value.func, ast.Name) and kw.value.func.id == "dict":
                        values[kw.arg] = {item.arg: ast.literal_eval(item.value) for item in kw.value.keywords}
                    else:
                        values[kw.arg] = ast.literal_eval(kw.value)
                matches.append((cell_number, values))
    assert len(matches) == 1, (class_name, matches)
    return matches[0]


class FrameFilteredResults:
    """Ad hoc results adapter: mask session rows or environment cells BEFORE mouse averaging."""
    def __init__(self, results, frame_results, scope):
        self.base = results
        self.frame_results = frame_results
        self.scope = scope
        self._frame_cache = {}
        self.audit = {}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def sel(self, *, avg_by_mouse=False, **selection):
        assert "mouse" not in selection, "This one-off adapter expects all mice."
        model_name = selection["model_name"]
        if model_name not in self._frame_cache:
            self._frame_cache[model_name] = np.asarray(self.frame_results.sel(
                model_name=model_name, keys=["num_frames_env"], avg_by_mouse=False,
            )["num_frames_env"], dtype=float)
        frames = self._frame_cache[model_name]
        selected = self.base.sel(avg_by_mouse=False, **selection)
        if self.scope == "session":
            # Known-environment frames are a lower bound on the whole held-out split.
            # Every selected session clears 100 using this bound, so unknown frames cannot
            # affect inclusion and no model prediction/frame data need to be loaded.
            totals = np.sum(frames, axis=1)
            keep = np.isfinite(totals) & (totals >= THRESHOLD)
            self.audit[model_name] = {"retained_sessions": int(keep.sum()), "sessions": len(keep),
                                      "minimum_known_environment_frames_per_session": float(np.nanmin(totals))}
        else:
            keep = np.isfinite(frames) & (frames >= THRESHOLD)
            self.audit[model_name] = {"retained_environment_scores": int(keep.sum())}
        masked = {}
        for key, value in selected.items():
            values = np.array(value, dtype=float, copy=True)
            assert values.shape[0] == len(self.base.sessions)
            if self.scope == "environment":
                assert values.shape == keep.shape, (key, values.shape, keep.shape)
            values[~keep] = np.nan
            masked[key] = values
        return average_by_mouse(masked, self.base.mouse_names) if avg_by_mouse else masked


store = ReadOnlyStore()
sessions = get_database("vrSessions").iter_sessions(imaging=True, experimentType="Blender VR")
frame_results = ResultsAggregator(RegressionConfig(), store, sessions, keys=["num_frames_env"])
whole_results = ResultsAggregator(RegressionConfig(), store, sessions, keys=["r2"])
residual_keys = ["median_infold_within_pf_r2_weighted", "median_infold_outside_pf_r2_weighted"]
residual_results = ResultsAggregator(RegressionPlacefieldResidualConfig(), store, sessions, keys=residual_keys)
env_results = ResultsAggregator(RegressionConfig(), store, sessions, keys=["r2_env"])
jobs = [
    (ModelPerformanceViewer, whole_results, "session", "model_performance"),
    (ModelPlacefieldResidualInsetViewer, residual_results, "session", "placefield_residual_inset"),
    (RegressionFamiliarityByEnvViewer, env_results, "environment", "familiarity_by_env"),
]
manifest = {"notebook": str(NOTEBOOK), "threshold": THRESHOLD, "sessions": len(sessions), "viewers": {}}


def save_figure(viewer, name):
    fig = viewer.plot(viewer.state)
    fig.canvas.draw()
    for extension in ["png", "pdf"]:
        fig.savefig(OUTPUT / f"{name}.{extension}", dpi=350, facecolor="white")
    return fig


for viewer_class, raw_results, scope, filename in jobs:
    cell, kwargs = notebook_kwargs(viewer_class.__name__)
    adapter = FrameFilteredResults(raw_results, frame_results, scope)
    filtered = viewer_class(adapter, **kwargs)
    unfiltered = viewer_class(raw_results, **kwargs)
    if scope == "session":
        if isinstance(filtered._scores, dict):
            for region in filtered._scores:
                np.testing.assert_allclose(filtered._scores[region], unfiltered._scores[region], equal_nan=True)
        else:
            np.testing.assert_allclose(filtered._scores, unfiltered._scores, equal_nan=True)
        assert all(item["retained_sessions"] == len(sessions) for item in adapter.audit.values())
    else:
        for model_name, masked, original in [
            (kwargs["baseline_model"], filtered._baseline_scores, unfiltered._baseline_scores),
            (kwargs["reference_model"], filtered._reference_scores, unfiltered._reference_scores),
        ]:
            frames = adapter._frame_cache[model_name]
            assert np.all(np.isnan(masked[frames < THRESHOLD]))
            np.testing.assert_allclose(masked[frames >= THRESHOLD], original[frames >= THRESHOLD], equal_nan=True)
            assert int(np.isfinite(masked).sum()) == 251
        rows = []
        for index, session in enumerate(sessions):
            for slot in range(filtered._baseline_scores.shape[1]):
                rows.append(dict(session=session.session_uid, mouse=session.mouse_name, slot=slot + 1,
                                 heldout_frames=adapter._frame_cache[kwargs["baseline_model"]][index, slot],
                                 baseline_r2=filtered._baseline_scores[index, slot],
                                 reference_r2=filtered._reference_scores[index, slot]))
        pd.DataFrame(rows).to_csv(OUTPUT / "filtered_environment_scores.csv", index=False)
    save_figure(filtered, f"{filename}_frames_ge100")
    save_figure(unfiltered, f"{filename}_unfiltered")
    manifest["viewers"][viewer_class.__name__] = dict(
        notebook_cell=cell, constructor_kwargs=kwargs, resolved_state=dict(filtered.state),
        filter_scope=scope, coverage=adapter.audit, unchanged_scores=scope == "session",
    )
    print(viewer_class.__name__, json.dumps(adapter.audit), flush=True)
    plt.close("all")

assert source_hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in VIEWER_FILES}
manifest["viewer_source_unchanged"] = True
(OUTPUT / "settings_and_coverage.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
print("Viewer source hashes unchanged. OUTPUT", OUTPUT, flush=True)
