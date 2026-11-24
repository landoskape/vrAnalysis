"""Read-only inspection of ATL012's late-session reliability dip."""
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vrAnalysis.database import get_database
from dimensionality_manuscript import PFPredQualityConfig, ResultsAggregator, ResultsStore
from dimensionality_manuscript.registry import RegistryPaths
from vrAnalysis.processors.spkmaps import SpkmapProcessor, SpkmapParams


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


if __name__ == "__main__":
    sessions = list(get_database("vrSessions").iter_sessions(imaging=True, experimentType="Blender VR", mouseName="ATL012"))
    sessions.sort(key=lambda s: (str(s.date), str(s.session_id)))
    print("SESSIONS", [(str(s), str(s.data_path)) for s in sessions], flush=True)
    results = ResultsAggregator(PFPredQualityConfig, ReadOnlyStore(), sessions,
                               keys=["reliability_slot", "fraction_active_slot", "env_slot_ids"])
    print("AXES", results.param_axes, flush=True)
    print("LAST VALUES", vars(sessions[-2].values), flush=True)
    print("ONE", [p.name for p in sessions[-2].get_saved_one()], flush=True)
    arrays = results.sel(squeeze_ones=False)
    print("SHAPES", {k: v.shape for k, v in arrays.items()}, flush=True)
    output = Path(__file__).parent / "atl012_reliability_inspection"
    output.mkdir(exist_ok=True)
    rows, plane_rows, motion_rows = [], [], []
    for row, session in enumerate(sessions):
        if row < len(sessions) - 5:
            continue
        session.params.spks_type = PFPredQualityConfig().spks_type
        selected = session.idx_rois
        planes = session.get_plane_idx()[selected]
        times = session.loadone("mpci.times")
        bt = session.loadone("positionTracking.times")
        smp = SpkmapProcessor(session, params=SpkmapParams(autosave=False))
        pos, speed, env_frames, trial_frames = smp.get_frame_behavior()
        cached, valid = smp.load_from_cache("reliability")
        print("DETAIL", str(session), vars(session.values), "selected", int(selected.sum()),
              "params", session.params, "cache", valid, flush=True)
        if valid:
            cached.values = cached.values[:, selected]
            print("CACHE MATCH", np.allclose(cached.values, arrays["reliability_slot"][row, :, :selected.sum()], equal_nan=True), flush=True)
        for plane in session.values.planeIDs:
            ops = np.load(session.s2p_path / f"plane{plane}" / "ops.npy", allow_pickle=True).item()
            entry = dict(date=str(session.date), plane=int(plane), selected_rois=int(np.sum(planes == plane)))
            for key in ("corrXY", "xoff", "yoff", "badframes", "meanImg"):
                if key in ops:
                    values = np.asarray(ops[key])
                    entry[key + "_median"] = float(np.nanmedian(values))
                    entry[key + "_p95_abs_centered"] = float(np.nanpercentile(np.abs(values - np.nanmedian(values)), 95))
                    if key == "badframes":
                        entry["badframes_fraction"] = float(np.mean(values))
            motion_rows.append(entry)
        for slot, env in enumerate(arrays["env_slot_ids"][row]):
            rel = arrays["reliability_slot"][row, slot]
            fa = arrays["fraction_active_slot"][row, slot]
            keep = (rel > 0.3) & (fa > 0.1)
            active = env_frames == env
            moving = active & np.isfinite(pos) & (speed > smp.params.speed_threshold)
            entry = dict(date=str(session.date), session_id=str(session.session_id), overall_session=row + 1,
                         environment=int(env), trials=int(np.sum(session.trial_environment == env)),
                         total_frames=len(times), duration_minutes=float((times[-1] - times[0]) / 60),
                         max_frame_gap=float(np.max(np.diff(times))), median_frame_interval=float(np.median(np.diff(times))),
                         behavior_duration_minutes=float((bt[-1] - bt[0]) / 60),
                         behavior_aligned_frames=int(np.sum(np.isfinite(pos))), environment_frames=int(active.sum()),
                         moving_frames=int(moving.sum()), median_running_speed=float(np.nanmedian(speed[moving])),
                         rois_total=int(session.values.numROIs), rois_selected=int(selected.sum()),
                         finite_reliability=int(np.isfinite(rel).sum()), quality_cells=int(keep.sum()),
                         quality_fraction=float(keep.sum() / selected.sum()),
                         reliability_all_mean=float(np.nanmean(rel)), reliability_quality_mean=float(np.mean(rel[keep])),
                         fraction_active_mean=float(np.nanmean(fa[:selected.sum()])))
            rows.append(entry)
            print("SUMMARY", entry, flush=True)
            for plane in np.unique(planes):
                plane_mask = planes == plane
                plane_rel = rel[:len(planes)][plane_mask]
                plane_keep = keep[:len(planes)][plane_mask]
                plane_rows.append(dict(date=str(session.date), environment=int(env), plane=int(plane),
                                       rois=int(plane_mask.sum()), quality_cells=int(plane_keep.sum()),
                                       reliability_quality_mean=float(np.mean(plane_rel[plane_keep])),
                                       reliability_all_median=float(np.nanmedian(plane_rel))))
        session.clear_cache()
    pd.DataFrame(rows).to_csv(output / "session_environment_summary.csv", index=False)
    pd.DataFrame(plane_rows).to_csv(output / "plane_reliability.csv", index=False)
    pd.DataFrame(motion_rows).to_csv(output / "motion_summary.csv", index=False)
    print("SAVED", output, flush=True)
