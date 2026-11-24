"""Render the position-scaled and compact distribution variants from the real v12 sweep."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from dimensionality_manuscript.configs.pfpred_quality import PFPredQualityConfig
from dimensionality_manuscript.pipeline import ResultsAggregator, ResultsStore
from dimensionality_manuscript.figure_scripts.figure1 import PlacefieldVarianceTrialFocus, PlacefieldVarianceDistributions
from vrAnalysis.sessions import B2Session


def save(viewer, name):
    path = Path(__file__).resolve().parent / name
    fig = viewer.plot(viewer.state)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    print("Saved", name, flush=True)


def main():
    config = PFPredQualityConfig()
    store = ResultsStore(blob_cache_maxsize=1)
    records = store.summary_table(analysis_type=config.display_name, schema_version=config.schema_version)
    ids = sorted({r["session_id"] for r in records if r["analysis_key"] == config.key() and r["result_stored"]})
    sessions = [B2Session.create(*uid.split("."), params={"spks_type": config.spks_type}) for uid in ids]
    results = ResultsAggregator(PFPredQualityConfig, store, sessions)
    target = next(s for s in sessions if s.session_uid == "ATL027.2023-08-07.701")
    stored = store.get(target.session_uid, config)
    slot = int(stored["best_env_slot"])
    print("Loaded", len(sessions), "sessions", flush=True)
    focus = PlacefieldVarianceTrialFocus(results, mouse=target.mouse_name, session=target.session_print(),
                                         env_slot=slot, roi=1808, xslice=slice(0, 5))
    save(focus, "trial_position_gain")
    focus = PlacefieldVarianceTrialFocus(results, mouse=target.mouse_name, session=target.session_print(),
                                         env_slot=slot, roi=662, sort_by="residual", xslice=slice(0, 5))
    # Center five trials around the strongest observed event for the second illustration.
    peak = int(np.argmax(focus.data))
    trial = focus.behavior.trial[focus.frames[peak]]
    index = int(np.searchsorted(focus.trial_numbers, trial))
    start = max(0, min(index - 2, len(focus.trial_numbers) - 5))
    focus.update_integer("xslice_start", value=start)
    focus.update_integer("xslice_stop", value=min(start + 5, len(focus.trial_numbers)))
    save(focus, "trial_position_residual")
    distributions = PlacefieldVarianceDistributions(results, mouse=target.mouse_name, session=target.session_print(),
                                                    env_slot=slot, session_cells="split", overview_cells="split")
    save(distributions, "compact_distributions")
    distributions.update_boolean("include_covariance", value=True)
    distributions.update_selection("plot_style", value="each")
    save(distributions, "compact_distributions_covariance_each")


if __name__ == "__main__":
    main()
