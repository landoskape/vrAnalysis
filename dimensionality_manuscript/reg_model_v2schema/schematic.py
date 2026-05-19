from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors


def _lighten(color: str, amount: float = 0.6) -> str:
    """Blend color toward white. amount=0 → original, amount=1 → white."""
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


# Matches registry.py ModelName literal exactly
ModelName = Literal[
    "external_placefield_1d",
    "internal_placefield_1d",
    "external_placefield_1d_gain",
    "internal_placefield_1d_gain",
    "external_placefield_1d_vector_gain",
    "internal_placefield_1d_vector_gain",
    "rbfpos_decoder_only",
    "rbfpos",
    "rbfpos_leak",
    "pos_speed_decoder_only",
    "pos_speed",
    "pos_speed_leak",
    "fullregressor_decoder_only",
    "fullregressor",
    "fullregressor_leak",
    "pos_speed_decoder_only_1dspeed",
    "pos_speed_1dspeed",
    "pos_speed_leak_1dspeed",
    "fullregressor_decoder_only_1dspeed",
    "fullregressor_1dspeed",
    "fullregressor_leak_1dspeed",
    "rbfpos_decoder_only_no_intercept",
    "rbfpos_no_intercept",
    "rbfpos_leak_no_intercept",
    "pos_speed_decoder_only_no_intercept",
    "pos_speed_no_intercept",
    "pos_speed_leak_no_intercept",
    "fullregressor_decoder_only_no_intercept",
    "fullregressor_no_intercept",
    "fullregressor_leak_no_intercept",
    "rrr",
    "rrr_no_intercept",
]

# Default vertical order mirrors position in MODEL_NAMES tuple (same as registry.py)
MODEL_NAMES: tuple[str, ...] = (
    "external_placefield_1d",
    "internal_placefield_1d",
    "external_placefield_1d_gain",
    "internal_placefield_1d_gain",
    "external_placefield_1d_vector_gain",
    "internal_placefield_1d_vector_gain",
    "rbfpos_decoder_only",
    "rbfpos",
    "rbfpos_leak",
    "pos_speed_decoder_only",
    "pos_speed",
    "pos_speed_leak",
    "fullregressor_decoder_only",
    "fullregressor",
    "fullregressor_leak",
    "pos_speed_decoder_only_1dspeed",
    "pos_speed_1dspeed",
    "pos_speed_leak_1dspeed",
    "fullregressor_decoder_only_1dspeed",
    "fullregressor_1dspeed",
    "fullregressor_leak_1dspeed",
    "rbfpos_decoder_only_no_intercept",
    "rbfpos_no_intercept",
    "rbfpos_leak_no_intercept",
    "pos_speed_decoder_only_no_intercept",
    "pos_speed_no_intercept",
    "pos_speed_leak_no_intercept",
    "fullregressor_decoder_only_no_intercept",
    "fullregressor_no_intercept",
    "fullregressor_leak_no_intercept",
    "rrr",
    "rrr_no_intercept",
)

_DEFAULT_ORDER: dict[str, int] = {name: i for i, name in enumerate(MODEL_NAMES)}

# Six feature columns — order defines x-axis left to right
FEATURE_COLUMNS: list[str] = [
    "1-D Pos",
    "Gain",
    "High-D Pos",
    "Speed",
    "Reward",
    "Unconstrained",
]

# ModelLine field name -> index in FEATURE_COLUMNS
_FEATURE_FIELD_IDX: dict[str, int] = {
    "pos_1d": 0,
    "gain": 1,
    "pos_hd": 2,
    "speed": 3,
    "reward": 4,
    "unconstrained": 5,
}

# Default colors by model — override per call via get_model_config(color=...)
DEFAULT_COLOR_LOOKUP: dict[str, str] = {
    "external_placefield_1d": "#4477AA",
    "internal_placefield_1d": "#4477AA",
    "external_placefield_1d_gain": "#66CCEE",
    "internal_placefield_1d_gain": "#66CCEE",
    "external_placefield_1d_vector_gain": "#66CCEE",
    "internal_placefield_1d_vector_gain": "#66CCEE",
    "rbfpos_decoder_only": "#228833",
    "rbfpos": "#228833",
    "rbfpos_leak": "#228833",
    "pos_speed_decoder_only": "#CCBB44",
    "pos_speed": "#CCBB44",
    "pos_speed_leak": "#CCBB44",
    "fullregressor_decoder_only": "#EE6677",
    "fullregressor": "#EE6677",
    "fullregressor_leak": "#EE6677",
    "pos_speed_decoder_only_1dspeed": "#CCBB44",
    "pos_speed_1dspeed": "#CCBB44",
    "pos_speed_leak_1dspeed": "#CCBB44",
    "fullregressor_decoder_only_1dspeed": "#EE6677",
    "fullregressor_1dspeed": "#EE6677",
    "fullregressor_leak_1dspeed": "#EE6677",
    "rbfpos_decoder_only_no_intercept": "#228833",
    "rbfpos_no_intercept": "#228833",
    "rbfpos_leak_no_intercept": "#228833",
    "pos_speed_decoder_only_no_intercept": "#CCBB44",
    "pos_speed_no_intercept": "#CCBB44",
    "pos_speed_leak_no_intercept": "#CCBB44",
    "fullregressor_decoder_only_no_intercept": "#EE6677",
    "fullregressor_no_intercept": "#EE6677",
    "fullregressor_leak_no_intercept": "#EE6677",
    "rrr": "#AA3377",
    "rrr_no_intercept": "#AA3377",
}


@dataclass
class ModelLine:
    """Visual config for one model row in the schematic.

    Feature booleans map 1-to-1 with FEATURE_COLUMNS:
        pos_1d        -> "1-D Pos"
        gain          -> "Gain"
        pos_hd        -> "High-D Pos"
        speed         -> "Speed"
        reward        -> "Reward"
        unconstrained -> "Unconstrained"

    Line style derives from ``internal``: dashed when True, solid when False.
    Node fill derives from ``leak``: hollow circles when True, filled when False.
    ``order`` controls vertical stacking (lower = higher / earlier on the plot).
    """

    label: str
    pos_1d: bool = False
    gain: bool = False
    pos_hd: bool = False
    speed: bool = False
    reward: bool = False
    unconstrained: bool = False
    internal: bool = False
    leak: bool = False
    order: int = 0
    color: str = "black"

    @property
    def line_style(self) -> str:
        return "dashed" if self.internal else "solid"

    @property
    def node_filled(self) -> bool:
        return not self.leak

    @property
    def active_col_indices(self) -> list[int]:
        return [idx for field, idx in _FEATURE_FIELD_IDX.items() if getattr(self, field)]


# Per-model kwargs — color and order injected by get_model_config
_MODEL_LINE_KWARGS: dict[str, dict] = {
    "external_placefield_1d": dict(label="PF", pos_1d=True),
    "internal_placefield_1d": dict(label="Int. PF", pos_1d=True, internal=True),
    "external_placefield_1d_gain": dict(label="PF+Gain", pos_1d=True, gain=True),
    "internal_placefield_1d_gain": dict(label="Int. PF+Gain", pos_1d=True, gain=True, internal=True),
    "external_placefield_1d_vector_gain": dict(label="PF+vGain", pos_1d=True, gain=True),
    "internal_placefield_1d_vector_gain": dict(label="Int. PF+vGain", pos_1d=True, gain=True, internal=True),
    "rbfpos_decoder_only": dict(label="HighD-Pos", pos_hd=True),
    "rbfpos": dict(label="Int. HighD-Pos", pos_hd=True, internal=True),
    "rbfpos_leak": dict(label="Leak HighD-Pos", pos_hd=True, internal=True, leak=True),
    "pos_speed_decoder_only": dict(label="Pos+HiDSpeed", pos_hd=True, speed=True),
    "pos_speed": dict(label="Int. Pos+HiDSpeed", pos_hd=True, speed=True, internal=True),
    "pos_speed_leak": dict(label="Leak Pos+HiDSpeed", pos_hd=True, speed=True, internal=True, leak=True),
    "fullregressor_decoder_only": dict(label="Full HiDSpeed", pos_hd=True, speed=True, reward=True),
    "fullregressor": dict(label="Int. Full HiDSpeed", pos_hd=True, speed=True, reward=True, internal=True),
    "fullregressor_leak": dict(label="Leak Full HiDSpeed", pos_hd=True, speed=True, reward=True, internal=True, leak=True),
    "pos_speed_decoder_only_1dspeed": dict(label="Pos+Speed", pos_hd=True, speed=True),
    "pos_speed_1dspeed": dict(label="Int. Pos+Speed", pos_hd=True, speed=True, internal=True),
    "pos_speed_leak_1dspeed": dict(label="Leak Pos+Speed", pos_hd=True, speed=True, internal=True, leak=True),
    "fullregressor_decoder_only_1dspeed": dict(label="Full", pos_hd=True, speed=True, reward=True),
    "fullregressor_1dspeed": dict(label="Int. Full", pos_hd=True, speed=True, reward=True, internal=True),
    "fullregressor_leak_1dspeed": dict(label="Leak Full", pos_hd=True, speed=True, reward=True, internal=True, leak=True),
    "rbfpos_decoder_only_no_intercept": dict(label="HighD-Pos (-b)", pos_hd=True),
    "rbfpos_no_intercept": dict(label="Int. HighD-Pos (-b)", pos_hd=True, internal=True),
    "rbfpos_leak_no_intercept": dict(label="Leak HighD-Pos (-b)", pos_hd=True, internal=True, leak=True),
    "pos_speed_decoder_only_no_intercept": dict(label="Pos+HiDSpeed (-b)", pos_hd=True, speed=True),
    "pos_speed_no_intercept": dict(label="Int. Pos+HiDSpeed (-b)", pos_hd=True, speed=True, internal=True),
    "pos_speed_leak_no_intercept": dict(label="Leak Pos+HiDSpeed (-b)", pos_hd=True, speed=True, internal=True, leak=True),
    "fullregressor_decoder_only_no_intercept": dict(label="Full HiDSpeed (-b)", pos_hd=True, speed=True, reward=True),
    "fullregressor_no_intercept": dict(label="Int. Full HiDSpeed (-b)", pos_hd=True, speed=True, reward=True, internal=True),
    "fullregressor_leak_no_intercept": dict(label="Leak Full HiDSpeed (-b)", pos_hd=True, speed=True, reward=True, internal=True, leak=True),
    "rrr": dict(label="RRR", unconstrained=True),
    "rrr_no_intercept": dict(label="RRR (-b)", unconstrained=True),
}


def get_model_config(
    model_name: ModelName,
    order: int | None = None,
    color: str | None = None,
) -> ModelLine:
    """Return a ModelLine for ``model_name`` with defaults from the lookup tables.

    Parameters
    ----------
    model_name : ModelName
        Key matching registry.py ModelName literal.
    order : int | None
        Override vertical stack order. Default = position in MODEL_NAMES tuple.
    color : str | None
        Override line/node color. Default from DEFAULT_COLOR_LOOKUP.

    Returns
    -------
    ModelLine
    """
    if model_name not in _MODEL_LINE_KWARGS:
        raise ValueError(f"Unknown model: {model_name!r}")
    kwargs = _MODEL_LINE_KWARGS[model_name].copy()
    kwargs["order"] = order if order is not None else _DEFAULT_ORDER[model_name]
    kwargs["color"] = color if color is not None else DEFAULT_COLOR_LOOKUP.get(model_name, "black")
    return ModelLine(**kwargs)


def get_model_names(
    include_placefield: bool = True,
    include_vector_gain: bool = False,
    include_highd_pos: bool = True,
    include_speed: bool = True,
    include_reward: bool = True,
    include_decoder_only: bool = True,
    include_doublecv: bool = False,
    include_leak: bool = False,
    include_1dspeed: bool = True,
    include_highd_speed: bool = False,
    include_no_intercept: bool = False,
    debug: bool = False,
):
    """Return a list of model names to include in the schematic, based on the boolean args."""
    names: list[str] = list(copy(MODEL_NAMES))
    for name in MODEL_NAMES:
        if not include_placefield and "placefield" in name:
            names.remove(name)
            if debug:
                print("Excluding placefield model:", name, " .. because include_placefield=False")
            continue
        if not include_vector_gain and "vector_gain" in name:
            names.remove(name)
            if debug:
                print("Excluding placefield model:", name, " .. because include_vector_gain=False")
            continue
        if not include_highd_pos and "rbfpos" in name:
            names.remove(name)
            if debug:
                print("Excluding placefield model:", name, " .. because include_highd_pos=False")
            continue
        if not include_speed and "pos_speed" in name:
            names.remove(name)
            if debug:
                print("Excluding placefield model:", name, " .. because include_speed=False")
            continue
        if not include_1dspeed and "1dspeed" in name:
            names.remove(name)
            if debug:
                print("Excluding placefield model:", name, " .. because include_1dspeed=False")
            continue
        if not include_highd_speed:
            if ("pos_speed" in name or "fullregressor" in name) and "1dspeed" not in name:
                names.remove(name)
                if debug:
                    print("Excluding placefield model:", name, " .. because include_highd_speed=False")
                continue
        if not include_reward and "fullregressor" in name:
            names.remove(name)
            if debug:
                print("Excluding placefield model:", name, " .. because include_reward=False")
            continue
        if not include_no_intercept and "no_intercept" in name:
            names.remove(name)
            if debug:
                print("Excluding placefield model:", name, " .. because include_no_intercept=False")
            continue
        if not include_leak and "leak" in name:
            names.remove(name)
            if debug:
                print("Excluding placefield model:", name, " .. because include_leak=False")
            continue
        if not include_decoder_only and "decoder_only" in name:
            names.remove(name)
            if debug:
                print("Excluding placefield model:", name, " .. because include_decoder_only=False")
            continue
        if not include_doublecv:
            if "rbfpos" in name or "pos_speed" in name or "fullregressor" in name:
                if "decoder_only" not in name and "leak" not in name:
                    names.remove(name)
                    if debug:
                        print("Excluding placefield model:", name, " .. because include_doublecv=False")
                    continue

    return names


# Default colors per model family group used by get_model_col_groups
_COL_GROUP_COLORS: dict[str, str] = {
    "placefield_1d": "#4477AA",
    "highd": "#228833",
    "no_intercept": "#994455",
    "rrr": "#AA3377",
}


def _classify_model(name: str) -> str:
    if "placefield" in name:
        return "placefield_1d"
    if "no_intercept" in name:
        return "no_intercept"
    if "rrr" in name:
        return "rrr"
    return "highd"  # rbfpos, pos_speed_*, fullregressor_* (any speed variant)


_COL_GROUP_ORDER: tuple[str, ...] = (
    "placefield_1d",
    "highd",
    "no_intercept",
    "rrr",
)


def get_model_col_groups(
    model_names: list[str],
    colors: dict[str, str] | None = None,
) -> list[tuple[list[int], list[int], str]]:
    """Return col_groups for plot_schematic, grouped by model family.

    Each entry: ``(model_rank_indices, feature_row_indices, color)``.
    Patches are fully opaque — choose light colors so model lines/nodes
    remain readable on top.
    The patch spans those model columns × the union of active feature rows
    across all models in the group (tight, not full plot height).

    Groups (present only when at least one matching model is in model_names):
        placefield_1d — models with ``"placefield"`` in name
        highd         — rbfpos / pos_speed / fullregressor (any speed variant), no ``no_intercept``
        no_intercept  — ``*_no_intercept``
        rrr           — ``"rrr"`` in name

    Parameters
    ----------
    model_names : list[str]
        Output of :func:`get_model_names` or any subset of MODEL_NAMES.
    colors : dict[str, str] | None
        Override colors per group key (see ``_COL_GROUP_COLORS``).

    Returns
    -------
    list[tuple[list[int], list[int], str]]
        Ready to pass as ``col_groups`` to :func:`plot_schematic`.
    """
    color_map = {**_COL_GROUP_COLORS, **(colors or {})}
    sorted_names = sorted(model_names, key=lambda n: _DEFAULT_ORDER.get(n, 999))

    group_ranks: dict[str, list[int]] = {g: [] for g in _COL_GROUP_ORDER}
    group_feat_rows: dict[str, set[int]] = {g: set() for g in _COL_GROUP_ORDER}

    for rank, name in enumerate(sorted_names):
        group = _classify_model(name)
        group_ranks[group].append(rank)
        cfg = get_model_config(name)
        group_feat_rows[group].update(cfg.active_col_indices)

    result = []
    for group in _COL_GROUP_ORDER:
        ranks = group_ranks[group]
        feat_rows = sorted(group_feat_rows[group])
        if ranks and feat_rows:
            result.append((ranks, feat_rows, color_map[group]))

    return result


def plot_schematic(
    model_names: list[str] | None = None,
    configs: list[ModelLine] | None = None,
    ax: plt.Axes | None = None,
    feature_columns: list[str] = FEATURE_COLUMNS,
    col_spacing: float = 1.0,
    row_spacing: float = 0.8,
    node_radius: float = 0.12,
    figsize: tuple[float, float] | None = None,
    show_legend: bool = True,
    xticklabel_fontsize: float = 9.0,
    ylabel_fontsize: float = 9.0,
    row_groups: list[tuple[list[int], str, float]] | None = None,
    row_group_pad: float = 0.25,
    col_groups: list[tuple[list[int], list[int], str]] | None = None,
    col_group_pad: float = 0.25,
    col_group_lighten: float = 0.9,
    show_feature_lines: bool = False,
    feature_line_color: str = "lightgray",
    feature_line_lw: float = 0.8,
) -> plt.Axes:
    """Draw the model schematic table (models on x-axis, features on y-axis).

    Coordinate system
    -----------------
    Model rank k sits at x = k * col_spacing (left to right by sort order).
    Feature row i sits at y = -i * row_spacing (top = index 0).
    Use :func:`get_coords` to find (x, y) for annotation placement.

    Parameters
    ----------
    model_names : list[str] | None
        Model name keys. Calls get_model_config for each. Mutually exclusive with configs.
    configs : list[ModelLine] | None
        Pre-built ModelLine objects. Use for custom configs.
    ax : plt.Axes | None
        Existing axes. Created if None.
    feature_columns : list[str]
        Row label strings. Length must equal number of feature boolean fields (6).
    col_spacing : float
        Horizontal distance between model columns.
    row_spacing : float
        Vertical distance between feature rows.
    node_radius : float
        Circle radius in data units.
    figsize : tuple | None
        Figure size. Auto-computed if None.
    show_legend : bool
        Draw solid/dashed legend for extrinsic/intrinsic.
    xticklabel_fontsize, ylabel_fontsize : float
        Font sizes for model name ticks and feature row labels.
    row_groups : list[tuple[list[int], str, float]] | None
        Optional horizontal background bands spanning all model columns.
        Each entry: (feature_indices, color, alpha). Patch spans those feature rows.
        Example::

            row_groups = [
                ([0, 1],    "#4477AA", 0.12),  # 1-D Pos, Gain
                ([2, 3, 4], "#228833", 0.12),  # High-D Pos, Speed, Reward
                ([5],       "#AA3377", 0.12),  # Unconstrained
            ]

    row_group_pad : float
        Rounding/expansion (data units) for row patch FancyBboxPatch ``round`` boxstyle.
    col_groups : list[tuple[list[int], list[int], str, float]] | None
        Optional background patches grouped by model family.
        Each entry: (model_rank_indices, feature_row_indices, color).
        Patches are fully opaque (cover feature trace lines beneath them).
        Patch spans those model columns × those feature rows (tight, not full height).
        Use :func:`get_model_col_groups` to generate from model names automatically.
    col_group_pad : float
        Rounding/expansion (data units) for column patch FancyBboxPatch ``round`` boxstyle.

    Returns
    -------
    ax : plt.Axes
    """
    if model_names is None and configs is None:
        raise ValueError("Provide model_names or configs.")
    if configs is None:
        configs = [get_model_config(n) for n in model_names]

    configs = sorted(configs, key=lambda c: c.order)

    feat_ys: list[float] = [-i * row_spacing for i in range(len(feature_columns))]

    if ax is None:
        if figsize is None:
            w = len(configs) * col_spacing + 2.0
            h = (len(feature_columns) - 1) * row_spacing + 2.5
            figsize = (w, h)
        _, ax = plt.subplots(figsize=figsize)

    # zorder hierarchy:
    #   0  feature trace lines (bottommost)
    #   1  col_group patches   (opaque, cover lines)
    #   2  row_group patches   (alpha, overlay on col patches)
    #   3  model lines
    #   4  filled nodes
    #   5  hollow nodes

    if show_feature_lines:
        n_models = len(configs)
        x_lo = -col_spacing * 0.5
        x_hi = (n_models - 1) * col_spacing + col_spacing * 0.5
        for fy in feat_ys:
            ax.plot(
                [x_lo, x_hi],
                [fy, fy],
                color=feature_line_color,
                linewidth=feature_line_lw,
                zorder=0,
            )

    if col_groups is not None:
        for rank_indices, feat_indices, color in col_groups:
            xs = [r * col_spacing for r in rank_indices]
            ys = [feat_ys[i] for i in feat_indices]
            patch = mpatches.FancyBboxPatch(
                (min(xs), min(ys)),
                max(xs) - min(xs),
                max(ys) - min(ys),
                boxstyle=f"round,pad={col_group_pad}",
                facecolor=_lighten(color, col_group_lighten),
                edgecolor="none",
                zorder=1,
            )
            ax.add_patch(patch)

    if row_groups is not None:
        all_xs = [rank * col_spacing for rank in range(len(configs))]
        x_tight_min = min(all_xs)
        x_tight_max = max(all_xs)
        for feat_indices, color, alpha in row_groups:
            ys = [feat_ys[idx] for idx in feat_indices]
            patch = mpatches.FancyBboxPatch(
                (x_tight_min, min(ys)),
                x_tight_max - x_tight_min,
                max(ys) - min(ys),
                boxstyle=f"round,pad={row_group_pad}",
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=2,
            )
            ax.add_patch(patch)

    for rank, cfg in enumerate(configs):
        x = rank * col_spacing
        active_ys = [feat_ys[i] for i in cfg.active_col_indices]
        if not active_ys:
            continue

        ls = (0, (3, 2)) if cfg.line_style == "dashed" else "-"
        y_pad = row_spacing * 0.45
        ax.plot(
            [x, x],
            [min(active_ys) - y_pad, max(active_ys) + y_pad],
            color=cfg.color,
            linestyle=ls,
            linewidth=1.5,
            zorder=3,
            solid_capstyle="round",
        )

        for yi in active_ys:
            if cfg.node_filled:
                patch = mpatches.Circle((x, yi), node_radius, color=cfg.color, zorder=4)
            else:
                patch = mpatches.Circle(
                    (x, yi),
                    node_radius,
                    edgecolor=cfg.color,
                    facecolor="white",
                    linewidth=1.5,
                    zorder=5,
                )
            ax.add_patch(patch)

    # Model names as x-ticks at bottom
    model_xs = [rank * col_spacing for rank in range(len(configs))]
    ax.set_xticks(model_xs)
    ax.set_xticklabels(
        [cfg.label for cfg in configs],
        fontsize=xticklabel_fontsize,
        rotation=45,
        rotation_mode="anchor",
        ha="right",
    )
    ax.xaxis.tick_bottom()
    ax.tick_params(axis="x", which="both", length=0, pad=4)

    # Feature names as y-ticks on left
    ax.set_yticks(feat_ys)
    ax.set_yticklabels(feature_columns, fontsize=ylabel_fontsize, fontweight="bold")
    ax.yaxis.tick_left()
    ax.tick_params(axis="y", which="both", length=0, pad=4)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")

    x_pad = col_spacing * 0.5
    y_pad = row_spacing * 0.6
    ax.set_xlim(-x_pad, (len(configs) - 1) * col_spacing + x_pad)
    ax.set_ylim(-(len(feature_columns) - 1) * row_spacing - y_pad, y_pad)

    if show_legend:
        handles = [
            mlines.Line2D([], [], color="gray", linestyle="-", linewidth=1.5, label="extrinsic"),
            mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5, label="intrinsic"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)

    return ax


def get_coords(
    rank: int,
    feature_idx: int,
    col_spacing: float = 1.0,
    row_spacing: float = 0.8,
) -> tuple[float, float]:
    """Return (x, y) for annotation placement on a plot_schematic output.

    ``rank`` = position in sorted config list (0 = leftmost column).
    ``feature_idx`` = index into FEATURE_COLUMNS (0 = top row).
    Pass the same col_spacing/row_spacing as used in plot_schematic.
    """
    return rank * col_spacing, -feature_idx * row_spacing
