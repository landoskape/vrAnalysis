"""Render real-data examples from the completed PFPredQualityConfig v12 sweep.

Run with the vrAnalysis Conda interpreter from the repository root. Only reads stored
analysis/session data; writes PNGs, SVGs, CSV, and example metadata next to this script.
"""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dimensionality_manuscript.configs.pfpred_quality import PFPredQualityConfig
from dimensionality_manuscript.pipeline import ResultsAggregator, ResultsStore
from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript.figure_scripts.figure1.pf_variance import (
    PlacefieldVarianceFocus, PlacefieldVarianceSummary, PlacefieldVarianceFamiliarity,
)
from vrAnalysis.sessions import B2Session


def save(viewer, name, output):
    fig = viewer.plot(viewer.state)
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.svg")
    plt.close(fig)
    print("Saved", name, flush=True)


def main():
    output = Path(__file__).resolve().parent
    store = ResultsStore(RegistryPaths().pipeline_v2_db_path, blob_cache_maxsize=1)
    config = PFPredQualityConfig()
    rows = store.summary_table(analysis_type=config.display_name, schema_version=config.schema_version)
    ids = sorted({row["session_id"] for row in rows if row["analysis_key"] == config.key() and row["result_stored"]})
    if not ids:
        raise RuntimeError("No completed default v12 results found.")
    sessions = [B2Session.create(*session_id.split("."), params={"spks_type": config.spks_type}) for session_id in ids]
    results = ResultsAggregator(PFPredQualityConfig, store, sessions)
    print("Loaded", len(sessions), "sessions", flush=True)
    target = next((s for s in sessions if s.session_uid == "ATL027.2023-08-07.701"), sessions[0])
    stored = store.get(target.session_uid, config)
    slot = int(stored["best_env_slot"])
    summary = PlacefieldVarianceSummary(results, mouse=target.mouse_name, session=target.session_print(),
                                       env_slot=slot, cells="split", display_percentile=95.)
    save(summary, "session_overview", output)
    summary.session_statistics().to_csv(output / "session_statistics.csv", index=False)
    familiarity = PlacefieldVarianceFamiliarity(results, mouse=target.mouse_name, cells="all")
    save(familiarity, "environment_familiarity", output)
    split = PlacefieldVarianceFamiliarity(results, mouse=target.mouse_name, cells="split")
    save(split, "environment_familiarity_quality", output)

    rel = stored["reliability_slot"][slot]
    active = stored["fraction_active_slot"][slot]
    gain = stored["pf_frac_var_gain_cv_slot"][slot]
    additive = stored["pf_frac_var_additive_cv_slot"][slot]
    keep = np.flatnonzero(np.isfinite(gain) & np.isfinite(additive) & (rel > .7) & (active > .2))
    if len(keep) < 2:
        keep = np.flatnonzero(np.isfinite(gain) & np.isfinite(additive))
    # Illustrative selections: strong gain versus strong additive fraction, among reliable ROIs.
    choices = [("gain_example", int(keep[np.argmax(gain[keep] - additive[keep])])),
               ("residual_example", int(keep[np.argmax(additive[keep] - gain[keep])]))]
    metadata = dict(session=target.session_print(), environment=int(stored["env_slot_ids"][slot]),
                    n_sessions=len(sessions), reliability_threshold=.7, fraction_active_threshold=.2,
                    examples=[])
    for name, roi in choices:
        focus = PlacefieldVarianceFocus(results, mouse=target.mouse_name, session=target.session_print(), roi=roi,
                                       all_environments=False, env_slot=slot, xslice=slice(0, 1200))
        # Zoom within the VR bout containing the peak; large inter-environment gaps retain
        # their true times but do not force a mostly empty example spanning the session.
        peak = int(np.argmax(focus.data))
        bout_boundaries = np.r_[0, np.flatnonzero(np.diff(focus.timestamps[focus.frames]) > 15.) + 1, len(focus.frames)]
        bout = np.searchsorted(bout_boundaries, peak, side="right") - 1
        bout_start, bout_stop = bout_boundaries[bout:bout + 2]
        start = max(int(bout_start), peak - 140)
        stop = min(start + 360, int(bout_stop))
        focus.update_integer("xslice_start", value=start)
        focus.update_integer("xslice_stop", value=stop)
        save(focus, name, output)
        metadata["examples"].append(dict(name=name, roi=roi, env_slot=slot,
                                         gain_ratio=float(gain[roi]), additive_ratio=float(additive[roi])))
    (output / "examples.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
