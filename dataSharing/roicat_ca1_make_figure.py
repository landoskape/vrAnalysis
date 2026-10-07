"""Recreate the CA1 ROICaT tracking figure from its portable data archive.

Only NumPy and Matplotlib are required.  The archive is produced by
``forROICaTFigure_260917.py`` inside the source repository, but this module and
the resulting ``.npz`` file can be distributed and run independently.

Example
-------
python roicat_ca1_make_figure.py roicat_ca1_figure_data.npz \
    --output roicat_ca1_figure.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ARCHIVE_NAME = "roicat_ca1_figure_data.npz"
REQUIRED_KEYS = {
    "metadata_json",
    "ab_sessions",
    "ab_fovs",
    "ab_match_masks",
    "ab_nonmatch_masks",
    "ab_crop_center_yx",
    "ab_crop_half_width",
    "ab_position_edges",
    "c_sessions",
    "c_sort_session",
    "c_snake_maps",
    "c_position_edges",
    "c_reward_center",
    "c_reward_half_width",
    "d_session_pairs",
    "d_quantile_probabilities",
    "d_correlation_quantiles",
    "e_mice",
    "e_group_names",
    "e_values",
}


def load_archive(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a figure archive without enabling pickle."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"Archive is missing required arrays: {sorted(missing)}")
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(data["metadata_json"]))
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported archive schema: {metadata.get('schema_version')}")
    return data


def _normalize(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    image_min = np.nanmin(image)
    image_max = np.nanmax(image)
    if image_max == image_min:
        return np.zeros_like(image)
    return (image - image_min) / (image_max - image_min)


def _panel_ab_overlays(data: dict[str, np.ndarray], roi_scale: float = 2.5) -> np.ndarray:
    fovs = data["ab_fovs"]
    match_masks = data["ab_match_masks"]
    nonmatch_masks = data["ab_nonmatch_masks"]
    overlays = []
    for idx in range(2):
        fov = _normalize(fovs[idx])
        match = _normalize(match_masks[idx])
        nonmatch = _normalize(nonmatch_masks[idx])
        rgb = np.repeat(fov[..., None], 3, axis=2)
        rgb[..., 0] += match * roi_scale
        rgb[..., 1 if idx == 0 else 2] += nonmatch * roi_scale
        overlays.append(_normalize(rgb))
    return np.stack(overlays)


def _quantile_at(probabilities: np.ndarray, quantiles: np.ndarray, probability: float) -> float:
    return float(np.interp(probability, probabilities, quantiles))


def _box_stats(probabilities: np.ndarray, quantiles: np.ndarray) -> dict[str, object]:
    """Create Matplotlib box statistics from exported full-data quantiles."""
    q1 = _quantile_at(probabilities, quantiles, 0.25)
    median = _quantile_at(probabilities, quantiles, 0.50)
    q3 = _quantile_at(probabilities, quantiles, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    finite = np.isfinite(quantiles)
    q = quantiles[finite]
    if q.size == 0:
        return {"q1": np.nan, "med": np.nan, "q3": np.nan, "whislo": np.nan, "whishi": np.nan, "fliers": []}
    within = q[(q >= lower_bound) & (q <= upper_bound)]
    whislo = float(within[0]) if within.size else q1
    whishi = float(within[-1]) if within.size else q3
    lower_fliers = q[q < whislo]
    upper_fliers = q[q > whishi]
    # Quantile-spaced fliers convey the tails without encoding millions of
    # redundant individual points in the archive or rendering them all.
    fliers = np.concatenate((lower_fliers[:: max(1, lower_fliers.size // 20)], upper_fliers[:: max(1, upper_fliers.size // 20)]))
    return {"q1": q1, "med": median, "q3": q3, "whislo": whislo, "whishi": whishi, "fliers": fliers}


def make_figure(data_or_path: dict[str, np.ndarray] | str | Path):
    """Create and return the complete A–E figure."""
    data = load_archive(data_or_path) if isinstance(data_or_path, (str, Path)) else data_or_path
    sessions_ab = data["ab_sessions"].astype(int)

    fig = plt.figure(figsize=(16, 7), layout="constrained")
    outer = fig.add_gridspec(2, 3, width_ratios=(1.05, 1.35, 1.35), height_ratios=(0.9, 1.15))

    # Panel A: registered FOVs and selected ROI masks.
    gs_a = outer[0, 0].subgridspec(1, 2, wspace=0.08)
    axes_a = [fig.add_subplot(gs_a[0, idx]) for idx in range(2)]
    overlays = _panel_ab_overlays(data)
    center_y, center_x = data["ab_crop_center_yx"]
    half_width = float(data["ab_crop_half_width"])
    for idx, ax in enumerate(axes_a):
        ax.imshow(overlays[idx], origin="upper")
        ax.set_xlim(center_x - half_width, center_x + half_width)
        ax.set_ylim(center_y - half_width, center_y + half_width)
        ax.set_title(f"Session {sessions_ab[idx]}", fontsize=9)
        ax.tick_params(labelsize=7)
        if idx:
            ax.set_yticklabels([])

    # Panel B: trial-resolved maps for the matched and unmatched cells.
    gs_b = outer[1, 0].subgridspec(2, 2, wspace=0.08, hspace=0.12)
    axes_b = np.asarray([[fig.add_subplot(gs_b[row, col]) for col in range(2)] for row in range(2)])
    edges = data["ab_position_edges"]
    map_cmaps = (("Reds", "Reds"), ("Greens", "Blues"))
    for col, session in enumerate(sessions_ab):
        maps = (data[f"ab_match_map_session_{session}"], data[f"ab_nonmatch_map_session_{session}"])
        for row, spkmap in enumerate(maps):
            ax = axes_b[row, col]
            ax.imshow(
                spkmap.T,
                aspect="auto",
                origin="upper",
                extent=(0, spkmap.shape[0], float(edges[0]), float(edges[-1])),
                cmap=map_cmaps[row][col],
                vmin=0,
                vmax=float(data["ab_zscore_limit"]),
                interpolation="nearest",
            )
            ax.tick_params(labelsize=7)
            if col:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(f"Session {session}", fontsize=9)
            if row == 1:
                ax.set_xlabel("Trials", fontsize=8)
    axes_b[0, 0].set_ylabel("Virtual Position (cm)", fontsize=8)
    axes_b[1, 0].set_ylabel("Virtual Position (cm)", fontsize=8)

    # Panel C: tracked-cell place-field snake.
    gs_c = outer[0, 1:].subgridspec(1, 8, width_ratios=(*([1] * 7), 0.05), wspace=0.05)
    axes_c = [fig.add_subplot(gs_c[0, idx]) for idx in range(7)]
    colorbar_ax = fig.add_subplot(gs_c[0, 7])
    snake_maps = data["c_snake_maps"]
    c_edges = data["c_position_edges"]
    max_per_roi = np.concatenate([np.nanmax(np.abs(panel), axis=1) for panel in snake_maps])
    vmax = float(np.nanpercentile(max_per_roi, float(data["c_display_percentile"])))
    reward_center = float(data["c_reward_center"])
    reward_half_width = float(data["c_reward_half_width"])
    sort_session = int(data["c_sort_session"])
    image_c = None
    for idx, (ax, session, panel) in enumerate(zip(axes_c, data["c_sessions"].astype(int), snake_maps)):
        image_c = ax.imshow(
            panel,
            cmap="bwr",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
            origin="upper",
            extent=(float(c_edges[0]), float(c_edges[-1]), 0, panel.shape[0]),
            interpolation="none",
        )
        ax.axvspan(reward_center - reward_half_width, reward_center + reward_half_width, color="k", alpha=0.16, lw=0)
        title = f"Session: {session}"
        if session == sort_session:
            title = "sorted here\n" + title
        ax.set_title(title, fontsize=6)
        ax.set_xlabel("VrPos (cm)", fontsize=6)
        ax.tick_params(labelsize=5, length=2)
        if idx == 0:
            ax.set_ylabel("ROIs", fontsize=7)
        else:
            ax.set_yticklabels([])
    fig.colorbar(image_c, cax=colorbar_ax, label="Activity (σ)")
    colorbar_ax.tick_params(labelsize=5)
    colorbar_ax.yaxis.label.set_size(6)

    # Panel D: correlations for ROICaT same-cell assignments and other pairs.
    ax_d = fig.add_subplot(outer[1, 1])
    probabilities = data["d_quantile_probabilities"]
    quantiles = data["d_correlation_quantiles"]
    session_pairs = data["d_session_pairs"].astype(int)
    positions_same = np.arange(len(session_pairs)) * 2.0 - 0.32
    positions_diff = np.arange(len(session_pairs)) * 2.0 + 0.32
    for group_idx, (positions, color) in enumerate(((positions_same, "#3278a6"), (positions_diff, "#e6862d"))):
        box_stats = [_box_stats(probabilities, quantiles[pair_idx, group_idx]) for pair_idx in range(len(session_pairs))]
        artists = ax_d.bxp(box_stats, positions=positions, widths=0.58, showfliers=True, patch_artist=True)
        for box in artists["boxes"]:
            box.set(facecolor=color, edgecolor="0.25", linewidth=0.5)
        for median in artists["medians"]:
            median.set(color="0.2", linewidth=0.8)
        for item in artists["whiskers"] + artists["caps"]:
            item.set(color="0.35", linewidth=0.5)
        for flier in artists["fliers"]:
            flier.set(marker="o", markerfacecolor="none", markeredgecolor="0.4", markersize=2, markeredgewidth=0.5)
    centers = np.arange(len(session_pairs)) * 2.0
    ax_d.set_xticks(centers, [f"{a},{b}" for a, b in session_pairs], fontsize=7)
    ax_d.set_xlabel("Session Pair", fontsize=8)
    ax_d.set_ylabel("PF Correlation", fontsize=8)
    ax_d.set_ylim(-1.05, 1.08)
    ax_d.tick_params(axis="y", labelsize=7)
    ax_d.legend(handles=[Patch(facecolor="#3278a6", label="Same"), Patch(facecolor="#e6862d", label="Diff")], fontsize=7, loc="lower right")

    # Panel E: per-mouse means and the constituent session-pair curves.
    ax_e = fig.add_subplot(outer[1, 2])
    values = data["e_values"]
    mice = data["e_mice"].astype(str)
    group_names = data["e_group_names"].astype(str)
    colors = plt.colormaps["Dark2"]
    x = np.arange(len(group_names))
    legend_handles = []
    for mouse_idx, (mouse, mouse_values) in enumerate(zip(mice, values)):
        color = colors(mouse_idx)
        for session_pair_values in mouse_values.T:
            ax_e.plot(x, session_pair_values, color=color, alpha=0.28, linewidth=0.7, linestyle="-.", zorder=0)
        ax_e.plot(x, np.nanmean(mouse_values, axis=1), color=color, marker="o", markersize=3, linewidth=1.2, zorder=1)
        legend_handles.append(Line2D([0], [0], color=color, marker="o", markersize=3, linewidth=1.2, label=f"Mouse {mouse_idx}"))
    ax_e.set_xticks(x, group_names, fontsize=7)
    ax_e.set_xlim(-0.25, len(group_names) - 0.75)
    ax_e.set_ylabel("Place Field Correlation", fontsize=8)
    ax_e.tick_params(axis="y", labelsize=7)
    ax_e.legend(handles=legend_handles, fontsize=6, loc="upper right")

    # Panel labels are placed relative to the first axis in each panel group.
    panel_axes = (axes_a[0], axes_b[0, 0], axes_c[0], ax_d, ax_e)
    panel_x = (-0.20, -0.24, -0.16, -0.16, -0.16)
    for label, ax, x_position in zip("ABCDE", panel_axes, panel_x):
        ax.text(x_position, 1.10, label, transform=ax.transAxes, fontsize=18, va="top", ha="right")
    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name(ARCHIVE_NAME),
        help="Portable .npz archive (default: next to this module)",
    )
    parser.add_argument("--output", type=Path, default=Path("roicat_ca1_figure.png"), help="Output figure path")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output resolution")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively")
    return parser.parse_args()


def main() -> Path:
    args = _parse_args()
    figure = make_figure(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {args.output}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return args.output


if __name__ == "__main__":
    main()
