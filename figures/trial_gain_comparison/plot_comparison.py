"""Compare stored raw/sqrt Trial Gain on held-out activity using the existing viewer."""

from pathlib import Path
from types import SimpleNamespace
import json
import os

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "cachedir" / "numba_tests"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dimensionality_manuscript.configs.trial_regression import TrialPlacecellRegressionConfig
from dimensionality_manuscript.pipeline import ResultsAggregator, ResultsStore
from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript.figure_scripts.figure2.trial_performance import (
    SQRT_GAIN_MODEL_NAME,
    SQRT_GAIN_MODEL_COLOR,
    TrialModelPerformanceViewer,
    trial_performance_scores,
)

RAW = "internal_placefield_1d_structured_gain"


class GainOnlyViewer(TrialModelPerformanceViewer):
    base_model_names = (RAW, SQRT_GAIN_MODEL_NAME)
    base_model_labels = ("Trial Gain\nraw", "Trial Gain\nsqrt")
    base_model_colors = ("#008000", SQRT_GAIN_MODEL_COLOR)


class SqrtPerformanceViewer(TrialModelPerformanceViewer):
    base_model_names = tuple(SQRT_GAIN_MODEL_NAME if name == RAW else name for name in TrialModelPerformanceViewer.base_model_names)
    base_model_labels = tuple("Trial Gain\n(sqrt)" if name == "Trial\nGain" else name for name in TrialModelPerformanceViewer.base_model_labels)
    base_model_colors = tuple(SQRT_GAIN_MODEL_COLOR if name == "#008000" else name for name in TrialModelPerformanceViewer.base_model_colors)


def limits(values, padding=.12):
    finite = np.asarray(values)[np.isfinite(values)]
    lo, hi = min(0., float(finite.min())), max(0., float(finite.max()))
    pad = max((hi - lo) * padding, .005)
    return (lo - pad, hi + pad)


def save(viewer, name, title, n_mice):
    fig = viewer.plot(viewer.state)
    fig.axes[0].set_ylabel(r"$\Delta$ R$^2$", labelpad=10, fontsize=11)
    fig.suptitle(title, fontsize=13)
    fig.supxlabel(f"Environment mean → session mean per mouse; {n_mice} mice\nThin lines: mice · thick line: mouse mean · inset: absolute activity R²", fontsize=8)
    fig.savefig(OUT / (name + ".png"), dpi=200)
    fig.savefig(OUT / (name + ".pdf"))
    plt.close(fig)


store = ResultsStore(RegistryPaths().pipeline_v2_db_path)
rows = store.summary_table(analysis_type="trial_placecell_regression")
ids = sorted({row["session_id"] for row in rows})
# Aggregation only needs stored session identities and mouse labels, not session activity.
sessions = [SimpleNamespace(session_uid=sid, mouse_name=sid.split(".")[0]) for sid in ids]
results = ResultsAggregator(TrialPlacecellRegressionConfig, store, sessions, keys=["r2", "mse"], lazy=False)

raw_env = np.asarray(results.sel(model_name=RAW)["r2"])
sqrt_env = np.asarray(results.sel(model_name=SQRT_GAIN_MODEL_NAME)["r2"])
assert np.array_equal(np.isfinite(raw_env), np.isfinite(sqrt_env)), "Raw/sqrt coverage differs; match environments before comparison"
paired = trial_performance_scores(results, [RAW, SQRT_GAIN_MODEL_NAME], "r2", {}, "Average", True)
mice = results.unique_mice
table = pd.DataFrame({"mouse": mice, "raw_activity_r2": paired[0], "sqrt_activity_r2": paired[1], "sqrt_minus_raw": paired[1] - paired[0]})
table.to_csv(OUT / "paired_mouse_scores.csv", index=False)

common = dict(metric="r2", environment="Average", aggregation="mean", avg_by_mouse=True,
              fontsize=11, inset_fontsize=8, figsize=(5.8, 4.5),
              inset_x=.13, inset_y=.52, inset_width=.32, inset_height=.40,
              subject_alpha=.35, subject_linewidth=.7, zero_line_pad=.25)
delta = paired - paired[0]
comparison = GainOnlyViewer(results, **common, ylim=limits(delta), ytick_max=.01,
                            inset_ylim=limits(paired), inset_ytick_max=float(np.nanmax(np.nanmean(paired, axis=1))))
save(comparison, "trial_gain_raw_vs_sqrt", "Trial Gain: raw vs sqrt", len(mice))

model_order = ["external_placefield_1d", "internal_placefield_1d", "internal_placefield_1d_gain", RAW, SQRT_GAIN_MODEL_NAME, "rrr"]
scores = trial_performance_scores(results, model_order, "r2", {}, "Average", True)
shared_limits = limits(scores - scores[0])
shared_absolute_limits = limits(scores)
full_settings = dict(**common, include_external_placefield=True, ylim=shared_limits, ytick_max=.1,
                     inset_ylim=shared_absolute_limits, inset_ytick_max=.2)
for cls, name, title in [(TrialModelPerformanceViewer, "performance_raw", "Trial model performance · raw Trial Gain"),
                         (SqrtPerformanceViewer, "performance_sqrt", "Trial model performance · sqrt Trial Gain")]:
    viewer = cls(results, **full_settings)
    save(viewer, name, title, len(mice))

pd.DataFrame(scores.T, index=mice, columns=model_order).rename_axis("mouse").to_csv(OUT / "all_model_mouse_scores.csv")
stats = dict(sessions=len(ids), usable_gain_sessions=int(np.isfinite(raw_env).any(axis=1).sum()), mice=len(mice),
             mean_raw=float(np.nanmean(paired[0])), mean_sqrt=float(np.nanmean(paired[1])),
             mean_paired_difference=float(np.nanmean(paired[1]-paired[0])),
             mice_improved=int(np.sum(paired[1] > paired[0])))
(OUT / "summary.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2), flush=True)
print(table.to_string(index=False), flush=True)
print("Saved plots to", OUT, flush=True)
