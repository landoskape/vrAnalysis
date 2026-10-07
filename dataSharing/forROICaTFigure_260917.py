"""Export the data underlying the CA1 ROICaT tracking figure.

This script is the repository-aware half of the data-availability workflow.  It
uses the original ``vrAnalysis`` data and analysis objects to produce one
portable NumPy archive.  The companion ``roicat_ca1_make_figure.py`` module has
no dependency on ``vrAnalysis`` and recreates the figure from that archive.

The default output is::

    storage_path() / "roicat_figure" / "roicat_ca1_figure_data.npz"

The archive contains numeric/string arrays only and can therefore be loaded
with ``numpy.load(..., allow_pickle=False)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse, stats

from vrAnalysis.files import storage_path
from _old_vrAnalysis import analysis, helpers, tracking


ARCHIVE_NAME = "roicat_ca1_figure_data.npz"
SCHEMA_VERSION = 1
KEEP_PLANES = [1, 2, 3, 4]
RELIABILITY_CUTOFFS = (0.4, 0.7)
DISTANCE_LIMIT_PIXELS = 10.0
RANDOM_SEED = 0

# Exact examples used in panels A and B of the original figure.
AB_MOUSE = "ATL022"
AB_ENVIRONMENT = 3
AB_SESSIONS = [11, 14]
AB_MATCH_ROIS = [3469, 3304]
AB_NONMATCH_ROIS = [5216, 4918]
AB_CROP_HALF_WIDTH = 30.0
AB_ZSCORE_LIMIT = 3.0

# Exact selection used in the original tracked-cell snake plot.
C_MOUSE = "ATL027"
C_ENVIRONMENT = 3
C_SESSIONS = list(range(7, 14))
C_SORT_SESSION = 10

# Exact selection used in the same/different-cell comparison.
D_MOUSE = "ATL027"
D_ENVIRONMENT = 3
D_SESSIONS = [8, 9, 10, 11]
D_QUANTILE_PROBABILITIES = np.linspace(0.0, 1.0, 1001)

# Mice present in the six-mouse version of panel E.
E_MICE = [
    "CR_Hippocannula6",
    "CR_Hippocannula7",
    "ATL022",
    "ATL027",
    "ATL020",
    "ATL012",
]


def default_output_path() -> Path:
    """Return the default archive path."""
    return storage_path() / "roicat_figure" / ARCHIVE_NAME


def _roistat(mouse_name: str) -> analysis.RoicatStats:
    return analysis.RoicatStats(tracking.tracker(mouse_name), keep_planes=KEEP_PLANES)


def _zscore_map(spkmap: np.ndarray) -> np.ndarray:
    """Match the original per-ROI z-scoring while returning float32 data."""
    return stats.zscore(spkmap, axis=None, nan_policy="omit").astype(np.float32)


def _finite_quantiles(values: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.full(D_QUANTILE_PROBABILITIES.size, np.nan, dtype=np.float32), 0
    return np.quantile(values, D_QUANTILE_PROBABILITIES).astype(np.float32), int(values.size)


def export_panels_ab() -> dict[str, np.ndarray]:
    """Export registered FOVs, masks, and trial maps for panels A and B."""
    np.random.seed(RANDOM_SEED + 1)
    roistat = _roistat(AB_MOUSE)
    track = roistat.track

    # Trial-resolved maps are concatenated over the four retained planes in the
    # same order as the ROI indices used by the original example script.
    spkmaps = roistat.get_spkmaps(
        AB_ENVIRONMENT,
        trials="full",
        average=False,
        tracked=False,
        idx_ses=AB_SESSIONS,
        by_plane=False,
    )[0]
    # The sessions have different trial counts, so maps are stored separately.
    arrays: dict[str, np.ndarray] = {
        "ab_sessions": np.asarray(AB_SESSIONS, dtype=np.int16),
        "ab_match_rois": np.asarray(AB_MATCH_ROIS, dtype=np.int32),
        "ab_nonmatch_rois": np.asarray(AB_NONMATCH_ROIS, dtype=np.int32),
        "ab_crop_half_width": np.asarray(AB_CROP_HALF_WIDTH, dtype=np.float32),
        "ab_zscore_limit": np.asarray(AB_ZSCORE_LIMIT, dtype=np.float32),
    }
    for i, session in enumerate(AB_SESSIONS):
        arrays[f"ab_match_map_session_{session}"] = _zscore_map(spkmaps[i][AB_MATCH_ROIS[i]])
        arrays[f"ab_nonmatch_map_session_{session}"] = _zscore_map(spkmaps[i][AB_NONMATCH_ROIS[i]])

    plane_idx = roistat.get_from_pcss("roiPlaneIdx", AB_SESSIONS)
    selected_planes = np.asarray([plane_idx[i][AB_MATCH_ROIS[i]] for i in range(2)], dtype=np.int16)
    nonmatch_planes = np.asarray([plane_idx[i][AB_NONMATCH_ROIS[i]] for i in range(2)], dtype=np.int16)
    if not (np.all(selected_planes == selected_planes[0]) and np.all(nonmatch_planes == selected_planes[0])):
        raise RuntimeError("Panel A/B ROIs no longer resolve to one common imaging plane")
    plane = int(selected_planes[0])
    arrays["ab_plane"] = np.asarray(plane, dtype=np.int16)

    fovs = [track.rundata[plane]["aligner"]["ims_registered_nonrigid"][session] for session in AB_SESSIONS]
    arrays["ab_fovs"] = np.stack(fovs).astype(np.float32)

    rois = track.get_ROIs(as_coo=False, idx_ses=AB_SESSIONS, keep_planes=KEEP_PLANES)
    rois = [sparse.vstack(session_rois, format="csr") for session_rois in rois]
    image_side = int(np.sqrt(rois[0].shape[1]))
    if image_side * image_side != rois[0].shape[1]:
        raise RuntimeError("ROI masks are not square images")
    match_masks = [rois[i][[AB_MATCH_ROIS[i]]].toarray().reshape(image_side, image_side) for i in range(2)]
    nonmatch_masks = [rois[i][[AB_NONMATCH_ROIS[i]]].toarray().reshape(image_side, image_side) for i in range(2)]
    arrays["ab_match_masks"] = np.stack(match_masks).astype(np.float32)
    arrays["ab_nonmatch_masks"] = np.stack(nonmatch_masks).astype(np.float32)

    centers = []
    for i in range(2):
        union = match_masks[i] + nonmatch_masks[i]
        y, x = np.nonzero(union)
        centers.append([np.mean(y), np.mean(x)])
    arrays["ab_crop_center_yx"] = np.mean(centers, axis=0).astype(np.float32)
    arrays["ab_position_edges"] = np.asarray(roistat.pcss[AB_SESSIONS[0]].distedges, dtype=np.float32)
    return arrays


def export_panel_c() -> dict[str, np.ndarray]:
    """Export the exact filtered and sorted matrices displayed in panel C."""
    np.random.seed(RANDOM_SEED + 2)
    roistat = _roistat(C_MOUSE)
    # This spells out the legacy ``make_snake_data`` calculation.  The legacy
    # helper itself predates the later addition of a third reliability measure
    # and no longer unpacks that API correctly.
    spkmaps, extras = roistat.get_spkmaps(
        C_ENVIRONMENT,
        idx_ses=C_SESSIONS,
        trials="full",
        average=False,
        tracked=True,
        pop_nan=False,
    )
    sort_idx = C_SESSIONS.index(C_SORT_SESSION)
    reliable_on_sort = (extras["relmse"][sort_idx] >= RELIABILITY_CUTOFFS[0]) & (extras["relcor"][sort_idx] >= RELIABILITY_CUTOFFS[1])
    train_indices, test_indices = helpers.named_transpose([helpers.cvFoldSplit(spkmap.shape[1], 2) for spkmap in spkmaps])
    sort_order = roistat.pcss[C_SORT_SESSION].get_place_field(spkmaps[sort_idx][reliable_on_sort][:, train_indices[sort_idx]], method="max")[1]
    snake_data = []
    for session_idx, spkmap in enumerate(spkmaps):
        selected = spkmap[reliable_on_sort][sort_order]
        if session_idx == sort_idx:
            selected = selected[:, test_indices[session_idx]]
        valid_counts = np.sum(np.isfinite(selected), axis=1)
        trial_sum = np.nansum(selected, axis=1)
        trial_mean = np.full(trial_sum.shape, np.nan, dtype=float)
        np.divide(trial_sum, valid_counts, out=trial_mean, where=valid_counts > 0)
        snake_data.append(trial_mean)
    if len({data.shape for data in snake_data}) != 1:
        raise RuntimeError("Panel C snake matrices do not share a common shape")

    reward_zone_data = [helpers.environmentRewardZone(roistat.pcss[i].vrexp) for i in C_SESSIONS]
    environment_indices = [roistat.pcss[i].envnum_to_idx(C_ENVIRONMENT)[0] for i in C_SESSIONS]
    reward_positions = [reward[0][env_idx] for reward, env_idx in zip(reward_zone_data, environment_indices)]
    reward_halfwidths = [reward[1][env_idx] for reward, env_idx in zip(reward_zone_data, environment_indices)]
    if not (np.allclose(reward_positions, reward_positions[0]) and np.allclose(reward_halfwidths, reward_halfwidths[0])):
        raise RuntimeError("Reward zone is inconsistent across panel C sessions")

    return {
        "c_sessions": np.asarray(C_SESSIONS, dtype=np.int16),
        "c_sort_session": np.asarray(C_SORT_SESSION, dtype=np.int16),
        "c_snake_maps": np.stack(snake_data).astype(np.float32),
        "c_position_edges": np.asarray(roistat.pcss[C_SESSIONS[0]].distedges, dtype=np.float32),
        "c_reward_center": np.asarray(reward_positions[0], dtype=np.float32),
        "c_reward_half_width": np.asarray(reward_halfwidths[0], dtype=np.float32),
        "c_display_percentile": np.asarray(80.0, dtype=np.float32),
    }


def _comparison(mouse_name: str, environment: int, sessions: list[int]):
    roistat = _roistat(mouse_name)
    return roistat.make_roicat_comparison(
        environment,
        idx_ses=sessions,
        sim_name="sConj",
        cutoffs=RELIABILITY_CUTOFFS,
        both_reliable=False,
        pop_nan=False,
    )


def export_panel_d() -> dict[str, np.ndarray]:
    """Export full-distribution quantiles for tracked/non-tracked ROI pairs."""
    np.random.seed(RANDOM_SEED + 3)
    _, correlations, tracked_pairs, _, _, _, _, parameters = _comparison(D_MOUSE, D_ENVIRONMENT, D_SESSIONS)
    quantiles = np.empty((len(correlations), 2, D_QUANTILE_PROBABILITIES.size), dtype=np.float32)
    counts = np.empty((len(correlations), 2), dtype=np.int64)
    for pair_idx, (corr, tracked_pair) in enumerate(zip(correlations, tracked_pairs)):
        for group_idx, mask in enumerate((tracked_pair.astype(bool), ~tracked_pair.astype(bool))):
            quantiles[pair_idx, group_idx], counts[pair_idx, group_idx] = _finite_quantiles(corr[mask])

    return {
        "d_session_pairs": np.asarray(parameters["idx_ses_pairs"], dtype=np.int16),
        "d_group_names": np.asarray(["Same", "Diff"]),
        "d_quantile_probabilities": D_QUANTILE_PROBABILITIES.astype(np.float32),
        "d_correlation_quantiles": quantiles,
        "d_counts": counts,
    }


def _select_environment_and_sessions(roistat: analysis.RoicatStats, mouse_name: str) -> tuple[int, list[int]]:
    environment = 2 if "CR" in mouse_name else roistat.env_selector(envmethod="most")
    sessions = list(roistat.idx_ses_selector(environment, sesmethod="all"))
    if len(sessions) > 7:
        sessions = sessions[-6:-2]
    elif len(sessions) > 4:
        sessions = sessions[-4:]
    if len(sessions) != 4:
        raise RuntimeError(f"Expected four panel E sessions for {mouse_name}, got {sessions}")
    return int(environment), sessions


def _panel_e_mouse(mouse_name: str, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    np.random.seed(seed)
    roistat = _roistat(mouse_name)
    environment, sessions = _select_environment_and_sessions(roistat, mouse_name)
    _, correlations, tracked_pairs, distances, nearest_pairs, _, _, parameters = roistat.make_roicat_comparison(
        environment,
        idx_ses=sessions,
        sim_name="sConj",
        cutoffs=RELIABILITY_CUTOFFS,
        both_reliable=False,
        pop_nan=False,
    )
    values = np.full((3, len(correlations)), np.nan, dtype=np.float32)
    for pair_idx, (corr, tracked_pair, distance, nearest_pair) in enumerate(zip(correlations, tracked_pairs, distances, nearest_pairs)):
        tracked_pair = tracked_pair.astype(bool)
        nearest_pair = nearest_pair.astype(bool) & (distance < DISTANCE_LIMIT_PIXELS)
        groups = (tracked_pair, nearest_pair, ~tracked_pair)
        for group_idx, mask in enumerate(groups):
            values[group_idx, pair_idx] = np.nanmean(corr[mask])
    return values, np.asarray(parameters["idx_ses_pairs"], dtype=np.int16), environment


def export_panel_e() -> dict[str, np.ndarray]:
    """Export per-session-pair means for the original six-mouse summary."""
    values = []
    session_pairs = []
    environments = []
    for mouse_idx, mouse_name in enumerate(E_MICE):
        print(f"Panel E: processing {mouse_name}", flush=True)
        mouse_values, mouse_pairs, environment = _panel_e_mouse(mouse_name, RANDOM_SEED + 100 + mouse_idx)
        values.append(mouse_values)
        session_pairs.append(mouse_pairs)
        environments.append(environment)
    return {
        "e_mice": np.asarray(E_MICE),
        "e_group_names": np.asarray(["tracked", "nearest neighbors", "random pairs"]),
        "e_values": np.stack(values).astype(np.float32),
        "e_session_pairs": np.stack(session_pairs).astype(np.int16),
        "e_environments": np.asarray(environments, dtype=np.int16),
        "e_distance_limit_pixels": np.asarray(DISTANCE_LIMIT_PIXELS, dtype=np.float32),
    }


def build_archive(output_path: Path | None = None) -> Path:
    """Generate all panel data and write the compressed portable archive."""
    output_path = Path(output_path) if output_path is not None else default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "title": "ROICaT tracking performance with CA1 place-field data",
        "source_repository": "vrAnalysis",
        "reliability_cutoffs": list(RELIABILITY_CUTOFFS),
        "keep_planes": KEEP_PLANES,
        "random_seed": RANDOM_SEED,
        "notes": "Numeric/string arrays only; load with allow_pickle=False.",
    }
    arrays: dict[str, np.ndarray] = {"metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))}

    print("Exporting panels A/B...", flush=True)
    arrays.update(export_panels_ab())
    print("Exporting panel C...", flush=True)
    arrays.update(export_panel_c())
    print("Exporting panel D...", flush=True)
    arrays.update(export_panel_d())
    print("Exporting panel E...", flush=True)
    arrays.update(export_panel_e())

    np.savez_compressed(output_path, **arrays)
    print(f"Saved {output_path} ({output_path.stat().st_size / 1024**2:.1f} MiB)")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Output .npz path (default: storage_path()/roicat_figure/roicat_ca1_figure_data.npz)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_archive(_parse_args().output)
