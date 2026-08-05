"""Variance-ratio-over-familiarity curves, shared by the familiarity and composite panels.

The data half turns a ``StimSpaceSpectraConfig`` selection into ``{curve_label: {"svr" / "total":
{mouse: 1D array}}}``; the drawing half renders those curves as one or two panels. Both are used
by :mod:`familiarity` on their own and by :mod:`complete_spectrum` inside a wider grid.
"""

import numpy as np

from vrAnalysis.helpers.plotting import errorPlot, format_spines
from dimensionality_manuscript.pipeline import ResultsAggregator
from ...env_order import ENV_SLOT_COLORS, MAX_ENV_SLOTS

from ._curves import chronological_mouse_sessions, mean_with_min_support, pad_stack_by_mouse, support_length

# Shared orange-family palette for behaving/ITI/spontaneous condition splits, used by both the
# familiarity panel (plot_mode="all") and the ratios beeswarms.
CONDITION_COLORS = {
    "behaving": "darkorange",
    "itis": "orangered",
    "spontaneous": "sienna",
}

FAMILIARITY_COLORS = {
    "Behaving": CONDITION_COLORS["behaving"],
    "w/ ITIs": CONDITION_COLORS["itis"],
    "w/ Spont.": CONDITION_COLORS["spontaneous"],
    **{f"Env #{slot + 1}": color for slot, color in enumerate(ENV_SLOT_COLORS)},
}

ENV_FULL_SCOPES = ["within_env", "outside_env", "with_iti", "with_spontaneous"]

# How each curve group is drawn: every mouse faint plus the bold mean, or the mean +/- SE band.
FAMILIARITY_STYLES = ["errorPlot", "all"]


def select_by_env_spectra(
    results: ResultsAggregator,
    sel_params: dict,
    env_full_scope: str,
    full_within_env: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the spectra and session mask selected by the ``by_env`` controls.

    The returned PF array always has shape ``(sessions, env slots, dimensions)``. The Full-CA1
    array has that shape when an environment-specific denominator is selected, otherwise it is
    the whole-session ``(sessions, dimensions)`` spectrum. Keeping this selection in one place
    ensures the Env # curves and any environment-averaged spectrum use identical stored keys,
    ITI state, and spontaneous-session filtering.
    """
    if env_full_scope not in ENV_FULL_SCOPES:
        raise ValueError(f"Unknown env_full_scope {env_full_scope!r}. Options: {ENV_FULL_SCOPES}")

    if env_full_scope == "within_env":
        sf_key = "sf_cv_env_full1"
        ff_key = "ff_env_full1"
        include_iti = False
    else:
        sf_key = "sf_cv_env_fullall"
        ff_key = "ff_env_full1_fullall" if full_within_env else "ff"
        include_iti = env_full_scope != "outside_env"

    out = results.sel(
        keys=[sf_key, ff_key],
        squeeze_ones=False,
        avg_by_mouse=False,
        include_iti=include_iti,
        **sel_params,
    )
    sf_all_slots = np.asarray(out[sf_key], dtype=float)
    ff_all_slots = np.asarray(out[ff_key], dtype=float)

    if env_full_scope == "with_iti":
        session_mask = ~np.array([session.has_spontaneous() for session in results.sessions])
    elif env_full_scope == "with_spontaneous":
        session_mask = np.array([session.has_spontaneous() for session in results.sessions])
    else:
        session_mask = np.ones(len(results.sessions), dtype=bool)
    return sf_all_slots, ff_all_slots, session_mask


def _curves_from_defs(
    results: ResultsAggregator,
    curve_defs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    exclude_bad_envs: bool,
    within_condition: bool = True,
) -> dict[str, dict]:
    """Turn ``{curve_label: (ratio, total, keep_mask)}`` session-level arrays into per-mouse curves.

    Shared tail end of :func:`familiarity_curves` for both ``mode="all"`` (curve_label = ITI
    status) and ``mode="by_env"`` (curve_label = env slot): reorders each mouse's sessions
    chronologically, then densifies to sessions where ``keep`` is True.

    When ``within_condition``, kept sessions are renumbered 0..n_kept-1 (the env/ITI-conditioned
    session index). When not, kept sessions instead keep their position in the mouse's full
    chronological session order (so e.g. a mouse's first spontaneous session, if it's their 6th
    session overall, lands in bin 5, not bin 0), with NaN filling the dropped-session gaps.
    """
    unique_mice = results.unique_mice
    curves: dict[str, dict] = {}
    for curve_label, (ratio, total, keep) in curve_defs.items():
        svr_per_mouse, total_per_mouse = {}, {}
        for mouse in unique_mice:
            idx_sorted = chronological_mouse_sessions(results, mouse, exclude_bad_envs=exclude_bad_envs)
            mouse_keep = keep[idx_sorted]
            if within_condition:
                idx_use = idx_sorted[mouse_keep]
                svr_per_mouse[mouse] = ratio[idx_use]
                total_per_mouse[mouse] = total[idx_use]
            else:
                svr_full = np.full(len(idx_sorted), np.nan)
                total_full = np.full(len(idx_sorted), np.nan)
                svr_full[mouse_keep] = ratio[idx_sorted[mouse_keep]]
                total_full[mouse_keep] = total[idx_sorted[mouse_keep]]
                svr_per_mouse[mouse] = svr_full
                total_per_mouse[mouse] = total_full
        curves[curve_label] = {"svr": svr_per_mouse, "total": total_per_mouse}
    return curves


def familiarity_curves(
    results: ResultsAggregator,
    sel_params: dict,
    mode: str,
    env_full_scope: str = "within_env",
    full_within_env: bool = True,
    within_condition: bool = True,
    max_dims: int | None = None,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Per-mouse variance-ratio / total-variance curves, keyed by curve label.

    Returns ``{curve_label: {"svr": {mouse: 1D array}, "total": {mouse: 1D array}}}``. In both
    modes the x-axis is, by default, the env/ITI-conditioned session index (session number
    within the kept subset, not the mouse's overall session number).

    ``within_condition`` (``mode == "all"`` only; ignored for ``"by_env"``): if False, curves are
    instead aligned to the mouse's overall chronological session index, so e.g. ``"w/ Spont."``
    data from a mouse's 6th session lands in bin 5 rather than bin 0.

    ``max_dims`` limits both the shared-variance numerator and full-variance denominator to the
    first N dimensions. The default, ``None``, preserves the full-spectrum calculation.

    ``mode == "all"``: whole-session ``sf_cv``/``ff`` keys, curve labels ``"Behaving"`` /
    ``"w/ ITIs"`` / ``"w/ Spont."`` overlaid together (unaffected by ``env_full_scope``).

    ``mode == "by_env"``: ``env_full_scope`` selects exactly one ``(sf_key, ff_key, include_iti,
    session_mask)`` combination -- ``"within_env"`` (``sf_cv_env_full1`` / ``ff_env_full1``,
    env-only frames, no ITI variant exists), ``"outside_env"`` (``sf_cv_env_fullall`` /
    ``ff_env_full1_fullall``, env-stim vs all-env-func, behaving-only), ``"with_iti"`` (same keys,
    ``include_iti=True``, non-spontaneous sessions), ``"with_spontaneous"`` (same keys,
    ``include_iti=True``, spontaneous sessions). Curve labels are ``"Env #1"``/``"Env #2"``/``"Env #3"``
    (all ``MAX_ENV_SLOTS`` experience-order slots are always overlaid together).
    """
    if max_dims is not None and max_dims < 1:
        raise ValueError("max_dims must be at least 1 or None")

    if mode == "all":
        session_has_spontaneous = np.array([s.has_spontaneous() for s in results.sessions])
        all_sessions = np.ones_like(session_has_spontaneous, dtype=bool)

        def _fetch(include_iti: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            out = results.sel(keys=["sf_cv", "ff"], squeeze_ones=False, avg_by_mouse=False, include_iti=include_iti, **sel_params)
            sf = out["sf_cv"][:, :max_dims]
            ff = out["ff"][:, :max_dims]
            total = np.nansum(ff, axis=1)
            ratio = np.nansum(sf, axis=1) / total
            valid = np.isfinite(ff).any(axis=1) & np.isfinite(sf).any(axis=1)
            return ratio, total, valid

        ratio_b, total_b, valid_b = _fetch(False)
        ratio_i, total_i, valid_i = _fetch(True)
        curve_defs = {
            "Behaving": (ratio_b, total_b, valid_b & all_sessions),
            "w/ ITIs": (ratio_i, total_i, valid_i & ~session_has_spontaneous),
            "w/ Spont.": (ratio_i, total_i, valid_i & session_has_spontaneous),
        }
        return _curves_from_defs(results, curve_defs, exclude_bad_envs=True, within_condition=within_condition)

    sf_all_slots, ff_all_slots, session_mask = select_by_env_spectra(
        results,
        sel_params,
        env_full_scope,
        full_within_env,
    )

    curve_defs = {}
    for env_slot in range(MAX_ENV_SLOTS):
        sf = sf_all_slots[:, env_slot, :max_dims]
        if ff_all_slots.ndim == 3:
            ff = ff_all_slots[:, env_slot, :max_dims]
        else:
            ff = ff_all_slots[:, :max_dims]
        total = np.nansum(ff, axis=1)
        ratio = np.nansum(sf, axis=1) / total
        valid = np.isfinite(ff).any(axis=1) & np.isfinite(sf).any(axis=1)
        curve_defs[f"Env #{env_slot + 1}"] = (ratio, total, valid & session_mask)

    return _curves_from_defs(results, curve_defs, exclude_bad_envs=False)


# ======================================================================================
# Drawing
# ======================================================================================


def familiarity_panel(ax, axis_curves: dict, metric: str, xlabel: str, ylabel: str, style: str, fontsize: float) -> float:
    """Plot one familiarity metric panel (svr or total) across curve labels; returns the max finite value drawn."""
    max_val = 0.0
    for curve_label, data in axis_curves.items():
        color = FAMILIARITY_COLORS[curve_label]
        per_mouse = data[metric]
        stack = pad_stack_by_mouse(per_mouse)
        length = support_length(stack)
        stack = stack[:, :length]
        if style == "all":
            for values in per_mouse.values():
                ax.plot(np.arange(len(values)), values, color=(color, 0.3), linewidth=0.5)
            ax.plot(np.arange(length), mean_with_min_support(stack), color=color, linewidth=2.0, label=curve_label)
            visible = stack
        elif length:
            errorPlot(np.arange(length), stack, axis=0, se=True, ax=ax, color=color, linewidth=2.0, label=curve_label, alpha=0.25)
            # Only the mean +/- SE band is actually drawn, not the raw per-mouse traces, so
            # the ymax should track the band's upper edge rather than individual-mouse outliers.
            num_valid = np.sum(~np.isnan(stack), axis=0)
            se = np.nanstd(stack, axis=0) / np.sqrt(num_valid)
            visible = np.nanmean(stack, axis=0) + se
        else:
            visible = stack
        finite = visible[np.isfinite(visible)]
        if finite.size:
            max_val = max(max_val, float(finite.max()))
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    return max_val


def format_familiarity_ylim(ax, max_val: float, fontsize: float, round_to_tenth: bool = True) -> None:
    """Set ``ylim = (0, 1.05 * max_val)`` and spine ``xbounds``/``ybounds`` up to clean values.

    ``ybounds`` top is ``ceil(ylim_top * 10) / 10`` when ``round_to_tenth`` (0-1 ratio panels),
    else ``ceil(ylim_top)`` (unbounded total-variance panel).
    """
    ylim_top = 1.05 * max_val if max_val > 0 else 1.0
    ybound_top = np.ceil(ylim_top * 10) / 10 if round_to_tenth else np.ceil(ylim_top)
    ylim_top = max(ylim_top, ybound_top)  # ceil can push the bound above the padded max; keep it on-screen
    ax.set_ylim(0, ylim_top)
    xbounds = [0, ax.get_xlim()[1]]
    format_spines(
        ax,
        x_pos=-0.02,
        y_pos=-0.02,
        spines_visible=["left", "bottom"],
        xbounds=xbounds,
        ybounds=[0, ybound_top],
        tick_fontsize=fontsize,
    )


def familiarity_xlabel(mode: str) -> str:
    """X label for a familiarity panel: overall session number, or session number within an env."""
    return "Session #" if mode == "all" else "Env Session #"


def render_familiarity_panels(ax_ratio, ax_total, curves: dict, mode: str, style: str, fontsize: float) -> None:
    """Render the Variance Ratio / Total Variance panel pair of the standalone familiarity figure."""
    xlabel = familiarity_xlabel(mode)
    max_ratio = familiarity_panel(ax_ratio, curves, "svr", xlabel, "Variance Ratio", style, fontsize)
    max_total = familiarity_panel(ax_total, curves, "total", xlabel, "Total Variance", style, fontsize)
    ax_ratio.legend(loc="lower right", fontsize=fontsize, frameon=False, markerfirst=False)
    format_familiarity_ylim(ax_ratio, max_ratio, fontsize)
    format_familiarity_ylim(ax_total, max_total, fontsize, round_to_tenth=False)


def render_familiarity_ratio_panel(ax_ratio, curves: dict, mode: str, style: str, fontsize: float) -> None:
    """Render only the Variance Ratio panel (no Total Variance), for the composite figure."""
    xlabel = familiarity_xlabel(mode)
    max_ratio = familiarity_panel(ax_ratio, curves, "svr", xlabel, "Variance Ratio", style, fontsize)
    ax_ratio.legend(loc="upper left", fontsize=fontsize, frameon=False, markerfirst=True, handlelength=0.8, handletextpad=0.5)
    format_familiarity_ylim(ax_ratio, max_ratio, fontsize)
