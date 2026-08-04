"""Mouse running speed over VR position, per environment, with a mixed-effects test of the difference."""

from collections import defaultdict
from itertools import combinations

import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import chi2, norm
from statsmodels.regression.mixed_linear_model import MixedLM

from vrAnalysis.helpers.plotting import errorPlot

from dimensionality_manuscript.configs.behavior_speed_env import ENV_REWARD_MAP, REFERENCE_ENV_LENGTH_CM
from dimensionality_manuscript.env_order import ENV_NUM_COLORS
from dimensionality_manuscript.figure_scripts.panels import (
    FigureViewer,
    add_data_selection_widgets,
    data_selection,
)
from dimensionality_manuscript.pipeline import ResultsAggregator

from ._shared import style_axis

# Result keys of BehaviorSpeedEnvConfig this panel draws, and the human-readable name of each
# trial set for the printed statistics report.
SPEED_CURVE_KEYS: dict[str, str] = {"all": "speed_curve_all", "first": "speed_curve_first"}
SPEED_TRIAL_SET_LABELS: dict[str, str] = {"all": "all trials", "first": "1st trial of block"}


def _significance_label(pvalue: float) -> str:
    """Star notation for a p-value: ``***`` < 0.001, ``**`` < 0.01, ``*`` < 0.05, else ``n.s.``."""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "n.s."


def _holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment of a family of p-values."""
    order = np.argsort(pvalues)
    m = pvalues.size
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def env_speed_mixedlm(window_speed: np.ndarray, env_list: list[int]) -> dict | None:
    """
    Mixed-effects test of an environment effect on per-mouse windowed speed.

    Fits ``speed ~ env + (1 | mouse)`` (treatment-coded environments, ML rather than REML so
    the likelihoods are comparable) and compares it against the intercept-only null
    ``speed ~ 1 + (1 | mouse)`` with a likelihood-ratio test. Mice contribute only the
    environments they actually ran, which the random intercept handles as unbalanced data.

    Parameters
    ----------
    window_speed : np.ndarray
        ``(n_mice, n_envs)`` window-averaged speed, NaN where a mouse lacks an environment.
    env_list : list[int]
        Environment identity of each column of ``window_speed``.

    Returns
    -------
    dict or None
        Omnibus and Holm-adjusted pairwise results, or None if fewer than two environments
        have data.
    """
    mouse_idx, col_idx = np.nonzero(np.isfinite(window_speed))
    y = window_speed[mouse_idx, col_idx]
    columns = sorted(set(col_idx.tolist()))
    if len(columns) < 2:
        return None

    # Treatment coding: intercept + one dummy per environment beyond the first (the reference).
    k = len(columns)
    exog = np.zeros((y.size, k))
    exog[:, 0] = 1.0
    for j, col in enumerate(columns[1:], start=1):
        exog[:, j] = col_idx == col

    full = MixedLM(y, exog, groups=mouse_idx).fit(reml=False)
    null = MixedLM(y, exog[:, :1], groups=mouse_idx).fit(reml=False)
    lr_stat = 2.0 * (full.llf - null.llf)
    lr_df = k - 1

    # Wald contrasts on the fixed effects; the reference level has an implicit coefficient of 0.
    # MixedLM orders the fixed effects first in the parameter vector, so the leading k x k block
    # of the covariance matrix is their covariance.
    fe = np.asarray(full.fe_params)
    cov_fe = np.asarray(full.cov_params())[:k, :k]
    pairs, estimates, standard_errors = [], [], []
    for a, b in combinations(range(k), 2):
        contrast = np.zeros(k)
        if a > 0:
            contrast[a] = -1.0
        if b > 0:
            contrast[b] = 1.0
        pairs.append((env_list[columns[a]], env_list[columns[b]]))
        estimates.append(float(contrast @ fe))
        standard_errors.append(float(np.sqrt(contrast @ cov_fe @ contrast)))

    estimates = np.asarray(estimates)
    standard_errors = np.asarray(standard_errors)
    z = estimates / standard_errors
    p_raw = 2.0 * norm.sf(np.abs(z))

    return dict(
        envs=[env_list[col] for col in columns],
        n_mice_per_env=[int(np.sum(col_idx == col)) for col in columns],
        n_mice=int(np.unique(mouse_idx).size),
        n_obs=int(y.size),
        env_means=[float(np.mean(y[col_idx == col])) for col in columns],
        lr_stat=float(lr_stat),
        lr_df=int(lr_df),
        lr_pvalue=float(chi2.sf(lr_stat, lr_df)),
        pairs=pairs,
        differences=estimates,
        standard_errors=standard_errors,
        z_values=z,
        p_values=p_raw,
        p_values_holm=_holm_adjust(p_raw),
    )


def format_speed_stats(stats: dict, window: tuple[float, float]) -> str:
    """Render the output of :func:`env_speed_mixedlm` (per trial set) as a text report."""
    lines = [f"speed ~ env + (1 | mouse), window [{window[0]:g}, {window[1]:g}) cm"]
    for key, result in stats.items():
        label = SPEED_TRIAL_SET_LABELS[key]
        if result is None:
            lines.append(f"  {label}: fewer than two environments with data -- skipped")
            continue
        counts = ", ".join(f"env {env}: {n}" for env, n in zip(result["envs"], result["n_mice_per_env"]))
        means = ", ".join(f"env {env}: {mu:.2f}" for env, mu in zip(result["envs"], result["env_means"]))
        lines.append(f"  {label}: {result['n_mice']} mice, {result['n_obs']} mouse-environment observations ({counts})")
        lines.append(f"    mean speed ({means}) cm/s")
        lines.append(f"    omnibus LRT vs env-free null: chi2({result['lr_df']}) = {result['lr_stat']:.3f}, p = {result['lr_pvalue']:.3g}")
        for (a, b), diff, se, z, p, p_holm in zip(
            result["pairs"],
            result["differences"],
            result["standard_errors"],
            result["z_values"],
            result["p_values"],
            result["p_values_holm"],
        ):
            lines.append(f"    env {b} - env {a} = {diff:+.3f} cm/s (SE {se:.3f}), z = {z:+.2f}, p = {p:.3g}, Holm p = {p_holm:.3g}")
    return "\n".join(lines)


class MouseSpeedFocus(FigureViewer):
    """Per-environment mouse speed over VR position, loaded from precomputed results.

    Reads the ``BehaviorSpeedEnvConfig`` aggregator: for the selected config parameters and trial
    set (all trials vs the first trial of each block) it assembles a ``(mice, envs, bins)`` speed
    array by mapping each session's stored ``speed_curve_*`` rows onto the global environment axis
    via its stored ``environments`` key, then averaging across sessions within a mouse.

    For each selected environment the mouse-average speed curve is drawn with a shaded
    ±standard-error band, colored per environment. Both trial sets are drawn: all trials (solid)
    and the first trial of each block (dashed). Each environment's reward zone (from
    :data:`ENV_REWARD_MAP`) is marked either by a dotted line at its start or by a shaded patch
    spanning the zone, colored to match.

    Windowed statistics (:meth:`compute_stats`) fit ``speed ~ env + (1 | mouse)`` to the per-mouse
    mean speed over a fixed position window, separately for each trial set, and compare that
    against an env-free null by a likelihood-ratio test with Holm-corrected pairwise contrasts.

    Parameters
    ----------
    results : ResultsAggregator
        Aggregated ``BehaviorSpeedEnvConfig`` results (which already exclude the CR_Hippocannula
        mice and any session whose reward layout does not match ``ENV_REWARD_MAP``).
    environments : list[int] or None
        Environments to draw and to include in the statistics. None uses all available.
    show_first : bool
        Draw the first-trial-of-block curve (dashed, no band).
    show_reward : bool
        Mark each environment's reward zone.
    reward_style : {"patch", "vlines"}
        Shade the whole zone, or draw a dotted line at its start.
    reward_width_cm : float
        Drawn width of the reward patch, from the zone start.
    reward_alpha : float
        Opacity of the reward patch.
    show_legend : bool
        Draw the legend at all.
    show_env_legend : bool
        Include the multi-colored "Envs" entry in it.
    extra_y_space : bool
        Extend the y-axis below 0, leaving blank space under the curves for the legend.
    alpha_band : float
        Opacity of the ±standard-error band.
    linewidth_average : float
        Line width of the mouse-average curves.
    fontsize : float
        Font size of every text element: axis labels, tick labels, legend, significance marker.
    stat_window : tuple[float, float]
        Position window (cm) the statistics average over, half-open: ``[start, stop)``.
    print_stats : bool
        Print the mixed-effects report when plotting.
    show_significance : bool
        Mark the omnibus environment effect above the curves at the window center.
    significance_trial_set : {"all", "first"}
        Which model the marked p-value comes from.
    figsize : tuple[float, float]
        Figure size in inches.
    **selection_defaults
        Starting values for the config's param axes (``num_bins``, ``regularization``,
        ``speed_threshold``), by name.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        environments: list[int] | None = None,
        show_first: bool = True,
        show_reward: bool = True,
        reward_style: str = "patch",
        reward_width_cm: float = 20.0,
        reward_alpha: float = 0.2,
        show_legend: bool = True,
        show_env_legend: bool = False,
        extra_y_space: bool = False,
        alpha_band: float = 0.2,
        linewidth_average: float = 2.0,
        fontsize: float = 9.0,
        stat_window: tuple[float, float] = (40.0, 50.0),
        print_stats: bool = False,
        show_significance: bool = True,
        significance_trial_set: str = "all",
        figsize: tuple[float, float] = (7.0, 5.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize

        # Global environment axis (identity-based, independent of the selected parameters).
        env_arr = np.asarray(results.arrays["environments"], dtype=float)
        self.env_list = sorted({int(e) for e in env_arr[np.isfinite(env_arr)] if e >= 0})
        self.reward_position = {env: float(ENV_REWARD_MAP[env]) for env in self.env_list}

        # --- data selection ---
        # num_bins is used to trim the padded curves, not only to index the grid, so it must
        # resolve to a value whether or not the config still sweeps it.
        self.selection_names = add_data_selection_widgets(self, results, defaults=selection_defaults, require=("num_bins",))
        env_options = [str(env) for env in self.env_list]
        selected = env_options if environments is None else [str(env) for env in environments]
        self.add_multiple_selection("environments", options=env_options, value=selected)

        # --- what is drawn ---
        self.add_boolean("show_first", value=show_first)
        self.add_boolean("show_reward", value=show_reward)
        self.add_selection("reward_style", options=["patch", "vlines"], value=reward_style)
        self.add_float("reward_width_cm", value=reward_width_cm, min=1.0, max=100.0)
        self.add_float("reward_alpha", value=reward_alpha, min=0.0, max=1.0)
        self.add_boolean("show_legend", value=show_legend)
        self.add_boolean("show_env_legend", value=show_env_legend)
        self.add_boolean("extra_y_space", value=extra_y_space)

        # --- style ---
        self.add_float("alpha_band", value=alpha_band, min=0.0, max=1.0)
        self.add_float("linewidth_average", value=linewidth_average, min=0.1, max=5.0)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

        # --- statistics ---
        self.add_float("stat_window_start", value=stat_window[0], min=0.0, max=REFERENCE_ENV_LENGTH_CM)
        self.add_float("stat_window_end", value=stat_window[1], min=0.0, max=REFERENCE_ENV_LENGTH_CM)
        self.add_boolean("print_stats", value=print_stats)
        self.add_boolean("show_significance", value=show_significance)
        self.add_selection("significance_trial_set", options=list(SPEED_CURVE_KEYS), value=significance_trial_set)

        self.on_change(list(self.selection_names), self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Assemble ``(mice, envs, bins)`` speed arrays (per trial set) + position axis."""
        curve_keys = list(SPEED_CURVE_KEYS.values())
        selection = data_selection(state, self.results, self.selection_names)
        sel = self.results.sel(keys=curve_keys + ["environments", "dist_fraction_centers"], squeeze_ones=False, **selection)

        # Pad keys are padded to the grid-wide max num_bins; trim to the selected bin count.
        num_bins = int(state["num_bins"])
        env_sel = np.asarray(sel["environments"], dtype=float)  # (n_sess, max_env)
        frac = np.asarray(sel["dist_fraction_centers"], dtype=float)[:, :num_bins]  # (n_sess, num_bins)

        mouse_per_session = list(self.results.mouse_names)
        # Which (session, row) pairs land on each (mouse, env) slot. Env coverage is the same
        # for every trial set, so this mapping is built once and reused for both curves.
        rows_by_slot: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
        for s in range(env_sel.shape[0]):
            mouse = mouse_per_session[s]
            for r in range(env_sel.shape[1]):
                env = env_sel[s, r]
                if not np.isfinite(env) or env < 0:
                    continue
                rows_by_slot[(mouse, int(env))].append((s, r))

        present = {mouse for mouse, _ in rows_by_slot}
        self.mouse_names = [m for m in dict.fromkeys(mouse_per_session) if m in present]

        # One (mice, envs, bins) array per trial set, averaged across sessions within a mouse.
        self.speed: dict[str, np.ndarray] = {}
        for key in curve_keys:
            curves = np.asarray(sel[key], dtype=float)[..., :num_bins]  # (n_sess, max_env, num_bins)
            arr = np.full((len(self.mouse_names), len(self.env_list), num_bins), np.nan)
            for m, mouse in enumerate(self.mouse_names):
                for e, env in enumerate(self.env_list):
                    rows = rows_by_slot.get((mouse, env))
                    if rows:
                        arr[m, e] = np.nanmean(np.stack([curves[s, r] for s, r in rows], axis=0), axis=0)
            self.speed[key] = arr

        # Position axis (cm): fraction-of-track is identical across matching sessions.
        finite_rows = np.where(np.all(np.isfinite(frac), axis=1))[0]
        frac_centers = frac[finite_rows[0]] if finite_rows.size else frac[0]
        self.dist_centers = frac_centers * REFERENCE_ENV_LENGTH_CM

    def compute_stats(self, state) -> dict[str, dict | None]:
        """
        Mixed-effects test of the environment effect on windowed speed, per trial set.

        Averages each mouse's speed curve over the position window
        ``[stat_window_start, stat_window_end)`` and fits ``speed ~ env + (1 | mouse)`` against
        an env-free null (see :func:`env_speed_mixedlm`). Only the selected environments are
        included; a mouse missing an environment simply contributes fewer rows.

        Returns
        -------
        dict
            ``{"all": result, "first": result}``, each the dict returned by
            :func:`env_speed_mixedlm` (or None when fewer than two environments have data).
        """
        window = (float(state["stat_window_start"]), float(state["stat_window_end"]))
        selected = [int(env) for env in state["environments"]]
        env_cols = [e for e, env in enumerate(self.env_list) if env in selected]
        envs = [self.env_list[e] for e in env_cols]
        # Half-open window: a bin center at exactly stat_window_end belongs to the next window.
        in_window = (self.dist_centers >= window[0]) & (self.dist_centers < window[1])

        stats = {}
        for name, key in SPEED_CURVE_KEYS.items():
            block = self.speed[key][:, env_cols, :][..., in_window]  # (mice, selected envs, window bins)
            # Mean over the window, NaN where a mouse never ran that environment (nanmean would
            # warn on those all-NaN slices).
            valid = np.isfinite(block)
            counts = valid.sum(axis=-1)
            totals = np.where(valid, block, 0.0).sum(axis=-1)
            window_speed = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
            stats[name] = env_speed_mixedlm(window_speed, envs)
        return stats

    def draw(self, state, ax):
        """Draw the speed curves onto ``ax``.

        Split out of :meth:`plot` so the combined schematic+speed figure
        (:func:`~.schematic.vr_schematic_and_speed`) can draw onto an externally placed axes
        instead of a standalone figure.
        """
        selected = [int(env) for env in state["environments"]]
        xvals = self.dist_centers
        env_length = REFERENCE_ENV_LENGTH_CM
        all_key = SPEED_CURVE_KEYS["all"]
        first_key = SPEED_CURVE_KEYS["first"]
        lw = state["linewidth_average"]
        fontsize = state["fontsize"]

        # ------------------------------------------------------------------ speed curves --
        drawn_envs = [env for env in self.env_list if env in selected]
        for e, env in enumerate(self.env_list):
            if env not in selected:
                continue
            color = ENV_NUM_COLORS[env]
            # All trials: mouse-average with a shaded ±SE band (average over the mouse axis).
            errorPlot(xvals, self.speed[all_key][:, e, :], axis=0, se=True, ax=ax, color=color, alpha=state["alpha_band"], linewidth=lw)
            if state["show_first"]:
                # First trial of each block: mouse-average only (dashed), no band -- keeps the
                # panel readable when both trial sets are shown together.
                ax.plot(xvals, np.nanmean(self.speed[first_key][:, e, :], axis=0), color=color, linewidth=lw, linestyle="--")
        xticks = np.arange(0, env_length + 1, 50)
        ax.set_xlabel("Position (cm)", fontsize=fontsize)
        ax.set_ylabel("Speed (cm/s)", fontsize=fontsize)
        ax.set_xlim(0, env_length)
        ax.set_xticks(xticks)
        _, ymax = ax.get_ylim()

        # Significance of the omnibus environment effect over the statistics window, at the window
        # center, top-aligned with the reward-zone patches (which span 0 to ymax).
        if state["show_significance"] and len(drawn_envs) > 1:
            stats = self.compute_stats(state)[state["significance_trial_set"]]
            window_center = 0.5 * (float(state["stat_window_start"]) + float(state["stat_window_end"]))
            ax.text(window_center, ymax, _significance_label(stats["lr_pvalue"]), ha="center", va="top", fontsize=fontsize)

        # Optionally drop the lower limit below 0, making room for a legend under the curves.
        ax.set_ylim(-10 if state["extra_y_space"] else 0, ymax)

        # Reward markers: drawn after ymax is known so they span only the data range [0, ymax],
        # not the negative legend margin. ``patch`` shades the whole reward zone, ``vlines``
        # marks only its start.
        if state["show_reward"]:
            for env in drawn_envs:
                start = self.reward_position[env]
                if state["reward_style"] == "patch":
                    width = min(state["reward_width_cm"], env_length - start)
                    # zorder=0 keeps the patch behind the speed curves and their error bands.
                    ax.add_patch(
                        Rectangle(
                            (start, 0),
                            width,
                            ymax,
                            facecolor=ENV_NUM_COLORS[env],
                            edgecolor="none",
                            alpha=state["reward_alpha"],
                            zorder=0,
                        )
                    )
                else:
                    ax.vlines(start, 0, ymax, color=ENV_NUM_COLORS[env], linestyle=":", linewidth=1.0)

        # Keep y-ticks at physical speeds (>= 0); the negative margin is legend space only.
        yticks = [t for t in ax.get_yticks() if 0 <= t <= ymax]
        style_axis(ax, fontsize=fontsize, xbounds=[0, env_length], ybounds=[0, ymax], xticks=xticks, yticks=yticks)

        self._draw_legend(ax, state, drawn_envs, lw, fontsize)

        if state["print_stats"]:
            window = (float(state["stat_window_start"]), float(state["stat_window_end"]))
            print(format_speed_stats(self.compute_stats(state), window))

    def _draw_legend(self, ax, state, drawn_envs, lw, fontsize):
        """Custom legend: the optional multi-colored "Envs" swatch, plus the line styles in use.

        The "Envs" entry is one handle whose segments run over the environment colors;
        HandlerTuple packs the sub-lines side by side across that single handle.
        """
        handles, labels = [], []
        if state["show_env_legend"] and drawn_envs:
            handles.append(tuple(Line2D([0], [0], color=ENV_NUM_COLORS[env], linewidth=lw) for env in drawn_envs))
            labels.append("Envs")
        if state["show_first"]:
            handles.append(Line2D([0], [0], color="0.3", linewidth=lw, linestyle="--"))
            labels.append("1st of block")
        # Patches are self-explanatory in place (colored by environment, at each reward zone), so
        # only the dotted-line style earns a legend entry.
        if state["show_reward"] and state["reward_style"] == "vlines":
            handles.append(Line2D([0], [0], color="0.3", linewidth=1.0, linestyle=":"))
            labels.append("reward zones")
        if state["show_legend"] and handles:
            # pad=0 packs the env segments flush against each other (one continuous swatch).
            ax.legend(
                handles,
                labels,
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
                loc="lower right",
                frameon=True,
                fontsize=fontsize,
            )

    def plot(self, state):
        fig, ax = self.new_subplots(1, 1, figsize=self.figsize, layout="constrained")
        self.draw(state, ax)
        return fig
