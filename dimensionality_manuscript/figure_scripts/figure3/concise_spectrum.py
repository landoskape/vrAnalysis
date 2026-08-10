"""Concise spectrum composite using a first-10-versus-all comparison."""

import pandas as pd

from dimensionality_manuscript.figure_scripts.legends import add_legend_widgets, apply_legend, update_legend_widgets
from dimensionality_manuscript.figure_scripts.panels import FigureViewer
from dimensionality_manuscript.pipeline import ResultsAggregator

from ._familiarity import CONDITION_COLORS, ENV_FULL_SCOPES, FAMILIARITY_STYLES, familiarity_curves, render_familiarity_ratio_panel
from ._ratios import (
    COMPOSITE_SPINE_OFFSET,
    plot_ratios_beeswarms_concise,
    plot_ratios_spectrum,
    ratios_arrays,
    spectrum_display_arrays,
)
from ._selection import add_stimspace_selection_widgets, stimspace_selection
from ._slopes import ENV_SLOPE_STYLES, env_slope_stats, env_slope_table, format_env_slope_stats, plot_env_slopes

LEGEND_DEFAULTS = {
    "spectrum_legend": dict(fontsize_scale=1.0),
    "env_legend": dict(fontsize_scale=1.0, handlelength=0.8, handletextpad=0.5),
}
LEGEND_AUTO_LOCS = {"spectrum_legend": "upper right", "env_legend": "upper left"}
TEXT_HORIZONTAL_ALIGNMENTS = ("left", "center", "right")
TEXT_VERTICAL_ALIGNMENTS = ("top", "center", "bottom", "baseline")
ANNOTATION_MODES = ("X", "X-Cov", "None")
ANNOTATION_SVR_XPAD = 0.015


def draw_shared_reliable_annotation(
    ax,
    xy: tuple[float, float],
    yoffset: float,
    line_yoffset: float,
    line_xpad: float,
    fontsize: float,
    *,
    mode: str = "X",
    numerator_color: str = CONDITION_COLORS["behaving"],
    denominator_color: str = "black",
):
    """Draw the selected two-row shared/reliable spectrum definition on ``ax``.

    All positions are in axes-fraction coordinates. ``xy`` is the center of the top row,
    ``yoffset`` is added to its y coordinate for the second row, and ``line_yoffset`` moves
    the separator away from the midpoint of those rows. ``line_xpad`` is the separator's
    half-width on either side of their shared x center. Both rows are bold; the numerator uses
    the PF curve color while the denominator and separator use the Full-CA1 curve color.

    Returns
    -------
    tuple[matplotlib.artist.Artist, ...]
        The top text, bottom text, and solid separator line, followed by the ``=SVR``
        text in ``"X-Cov"`` mode. Empty in ``"None"`` mode.
    """
    if mode not in ANNOTATION_MODES:
        raise ValueError(f"Unknown annotation mode {mode!r}; expected one of {ANNOTATION_MODES}")
    if mode == "None":
        return ()

    x, top_y = xy
    bottom_y = top_y + yoffset
    line_y = (top_y + bottom_y) / 2 + line_yoffset
    text_kwargs = {
        "transform": ax.transAxes,
        "fontsize": fontsize,
        "fontweight": "bold",
        "ha": "center",
        "va": "center",
    }
    if mode == "X":
        top_label = r" PF $\times$ CA1"
        bottom_label = r"CA1 $\times$ CA1"
    else:
        top_label = r"$\mathbf{xcov}(\mathbf{PF}_{j};\,\mathbf{CA1}_{k})$"
        bottom_label = r"$\mathbf{xcov}(\mathbf{CA1}_{j};\,\mathbf{CA1}_{k})$"
    top_text = ax.text(x, top_y, top_label, color=numerator_color, **text_kwargs)
    bottom_text = ax.text(x, bottom_y, bottom_label, color=denominator_color, **text_kwargs)
    (separator,) = ax.plot(
        [x - line_xpad, x + line_xpad],
        [line_y, line_y],
        transform=ax.transAxes,
        color=denominator_color,
        linestyle="-",
        linewidth=1.0,
        clip_on=False,
    )
    artists = (top_text, bottom_text, separator)
    # Suppressing this for now by comment... might bring back?
    # if mode == "X-Cov":
    #     svr_text = ax.text(
    #         x + line_xpad + ANNOTATION_SVR_XPAD,
    #         line_y,
    #         r"$=\mathrm{SVR}$",
    #         transform=ax.transAxes,
    #         fontsize=fontsize,
    #         color=denominator_color,
    #         ha="left",
    #         va="center",
    #     )
    #     artists += (svr_text,)
    return artists


class ConciseSpectrumViewer(FigureViewer):
    """Spectrum, optional variance-ratio swarms, and environment-familiarity curves.

    ``spectrum_mode="all"`` draws the whole-session spectra on ``ax[0]``. ``"avg_env"`` uses
    the same stored keys, ITI selection, and session mask as the Env # panel, averages available
    environment slots within each session, then averages sessions within mouse. Display clipping
    and smoothing remain isolated from every cumulative SVR calculation.

    ``first10`` sets the leading-dimension cutoff. ``include_first10_beeswarm`` and
    ``include_first10_by_env`` independently control whether that leading subset appears in the
    beeswarm and environment-familiarity views, while ``include_beeswarm`` controls whether the
    beeswarm panel itself is present. With the leading by-environment view included,
    ``by_env_layout`` either keeps it and the all-dimension curves on separate axes or overlays
    them on one shared axis. The optional final panel always summarizes all-dimension slopes.
    """

    def __init__(
        self,
        results: ResultsAggregator,
        *,
        env_full_scope: str = "within_env",
        full_within_env: bool = False,
        style_by_env: str = "errorPlot",
        spectrum_mode: str = "all",
        first10: int = 10,
        include_first10_beeswarm: bool = True,
        include_first10_by_env: bool = True,
        include_beeswarm: bool = True,
        by_env_layout: str = "separate",
        annotation_mode: str = "X",
        annotation_xy: tuple[float, float] = (0.72, 0.86),
        annotation_yoffset: float = -0.14,
        annotation_line_yoffset: float = 0.0,
        annotation_line_xpad: float = 0.22,
        first10_text_xy: tuple[float, float] = (0.05, 0.95),
        first10_text_ha: str = "left",
        first10_text_va: str = "top",
        all_text_xy: tuple[float, float] = (0.05, 0.95),
        all_text_ha: str = "left",
        all_text_va: str = "top",
        pf_smooth_method: str = "gaussian",
        pf_smooth_width: float = 3.0,
        full_smooth_method: str = "gaussian",
        full_smooth_width: float = 3.0,
        normalize: bool = True,
        clip_negative: bool = True,
        as_percent: bool = False,
        ylims: tuple[float, float] = (-5.0, -1.2),
        shared_variance_ratio_ylims: tuple[float, float] = (0.0, 1.0),
        ax1_width_ratio: float = 0.8,
        ax23_width_ratio: float = 1.0,
        show_slope_panel: bool = True,
        slope_style: str = "beeswarm",
        min_slope_sessions: int = 4,
        show_slope_stats: bool = True,
        spectrum_legend: dict | None = None,
        env_legend: dict | None = None,
        fontsize: float = 9.0,
        figsize: tuple[float, float] = (17.0, 3.0),
        **selection_defaults,
    ):
        self.results = results
        self.figsize = figsize
        self.show_slope_panel = bool(show_slope_panel)

        # Data-grid smoothing is selected independently from the display-only smoothing controls
        # below. The tuple-valued ``smooth_widths`` axis is label-encoded for Syd by the shared
        # stimspace selection helper.
        self.selection_names, self._tuple_labels = add_stimspace_selection_widgets(
            self,
            results,
            selection_defaults,
        )
        self.add_selection("env_full_scope", value=env_full_scope, options=ENV_FULL_SCOPES)
        self.add_boolean("full_within_env", value=full_within_env)
        self.add_selection("style_by_env", value=style_by_env, options=FAMILIARITY_STYLES)
        self.add_selection("spectrum_mode", value=spectrum_mode, options=("all", "avg_env"))
        self.add_integer("first10", value=first10, min=1, max=30)
        self.add_boolean("include_first10_beeswarm", value=include_first10_beeswarm)
        self.add_boolean("include_first10_by_env", value=include_first10_by_env)
        self.add_boolean("include_beeswarm", value=include_beeswarm)
        self.add_selection("by_env_layout", value=by_env_layout, options=("separate", "shared"))
        self.add_selection("annotation_mode", value=annotation_mode, options=ANNOTATION_MODES)
        self.add_float("annotation_x", value=annotation_xy[0], min=-1.0, max=2.0, step=0.01)
        self.add_float("annotation_y", value=annotation_xy[1], min=-1.0, max=2.0, step=0.01)
        self.add_float("annotation_yoffset", value=annotation_yoffset, min=-1.0, max=1.0, step=0.001)
        self.add_float("annotation_line_yoffset", value=annotation_line_yoffset, min=-1.0, max=1.0, step=0.001)
        self.add_float("annotation_line_xpad", value=annotation_line_xpad, min=0.0, max=1.0, step=0.001)
        text_controls = [
            ("first10_text", first10_text_xy, first10_text_ha, first10_text_va),
            ("all_text", all_text_xy, all_text_ha, all_text_va),
        ]
        for prefix, xy, ha, va in text_controls:
            self.add_float(f"{prefix}_x", value=xy[0], min=0.0, max=1.0, step=0.01)
            self.add_float(f"{prefix}_y", value=xy[1], min=0.0, max=1.0, step=0.01)
            self.add_selection(f"{prefix}_ha", value=ha, options=TEXT_HORIZONTAL_ALIGNMENTS)
            self.add_selection(f"{prefix}_va", value=va, options=TEXT_VERTICAL_ALIGNMENTS)

        self.add_selection("pf_smooth_method", value=pf_smooth_method, options=("none", "boxcar", "gaussian"))
        self.add_float("pf_smooth_width", value=pf_smooth_width, min=0.0, max=50.0, step=0.5)
        self.add_selection("full_smooth_method", value=full_smooth_method, options=("none", "boxcar", "gaussian"))
        self.add_float("full_smooth_width", value=full_smooth_width, min=0.0, max=50.0, step=0.5)
        self.add_boolean("normalize", value=normalize)
        self.add_boolean("clip_negative", value=clip_negative)
        self.add_boolean("as_percent", value=as_percent)
        self.add_float_range("ylims", value=ylims, min=-8.0, max=2.0, step=0.1)
        self.add_float_range(
            "shared_variance_ratio_ylims",
            value=shared_variance_ratio_ylims,
            min=0.0,
            max=1.0,
            step=0.01,
        )
        self.add_float("ax1_width_ratio", value=ax1_width_ratio, min=0.2, max=4.0, step=0.05)
        self.add_float("ax23_width_ratio", value=ax23_width_ratio, min=0.2, max=4.0, step=0.05)

        # Slope inclusion is constructor-level. When off, neither its data nor its Syd controls
        # are created; when on, only controls that actually configure the included panel appear.
        if self.show_slope_panel:
            self.add_selection("slope_style", value=slope_style, options=ENV_SLOPE_STYLES)
            self.add_integer("min_slope_sessions", value=min_slope_sessions, min=3, max=20)
            self.add_boolean("show_slope_stats", value=show_slope_stats)

        overrides = dict(spectrum_legend=spectrum_legend, env_legend=env_legend)
        for prefix, defaults in LEGEND_DEFAULTS.items():
            add_legend_widgets(self, prefix=prefix)
            update_legend_widgets(self, {**defaults, **(overrides[prefix] or {})}, prefix=prefix)

        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0, step=0.5)
        data_callbacks = [
            *self.selection_names,
            "env_full_scope",
            "full_within_env",
            "spectrum_mode",
            "first10",
            "include_first10_by_env",
        ]
        if self.show_slope_panel:
            data_callbacks.extend(("show_slope_stats", "min_slope_sessions"))
        for name in data_callbacks:
            self.on_change(name, self.refresh_data)
        self.refresh_data(self.state)

    def _sel_params(self, state: dict) -> dict:
        return stimspace_selection(state, self.selection_names, self._tuple_labels)

    def _env_curves(self, state: dict, max_dims: int | None) -> dict:
        return familiarity_curves(
            self.results,
            self._sel_params(state),
            "by_env",
            env_full_scope=state["env_full_scope"],
            full_within_env=state["full_within_env"],
            max_dims=max_dims,
        )

    def refresh_data(self, state):
        sel_params = self._sel_params(state)
        self._ratios_arrays = ratios_arrays(
            self.results,
            sel_params,
            spectrum_mode=state["spectrum_mode"],
            env_full_scope=state["env_full_scope"],
            full_within_env=state["full_within_env"],
            first_n=state["first10"],
        )
        if state["include_first10_by_env"]:
            self._curves_first10 = self._env_curves(state, state["first10"])
        self._curves_all = self._env_curves(state, None)
        if self.show_slope_panel:
            self._slope_table = env_slope_table(self._curves_all, state["min_slope_sessions"])
            self._slope_stats_text = (
                format_env_slope_stats(env_slope_stats(self._slope_table), list(self._curves_all)) if state["show_slope_stats"] else None
            )

    def slope_table_and_stats(self, state: dict | None = None) -> tuple[pd.DataFrame, dict | None]:
        """Return the all-dimension, per-environment slope table and mixed-effects tests."""
        if not self.show_slope_panel:
            raise RuntimeError("Slope support was disabled when this ConciseSpectrumViewer was created")
        state = self.state if state is None else state
        table = env_slope_table(self._env_curves(state, None), state["min_slope_sessions"])
        return table, env_slope_stats(table)

    def plot(self, state: dict):
        fontsize = state["fontsize"]
        show_ratio = state["include_beeswarm"]
        show_first10_beeswarm = state["include_first10_beeswarm"]
        show_first10_by_env = state["include_first10_by_env"]
        show_first10_indicator = (show_ratio and show_first10_beeswarm) or show_first10_by_env
        value_scale = 100.0 if state["as_percent"] else 1.0
        shared_by_env = show_first10_by_env and state["by_env_layout"] == "shared"
        width_ratios = [1.0]
        if show_ratio:
            width_ratios.append(state["ax1_width_ratio"])
        if show_first10_by_env and not shared_by_env:
            width_ratios.append(state["ax23_width_ratio"])
        width_ratios.append(state["ax23_width_ratio"])
        if self.show_slope_panel:
            width_ratios.append(1.0)
        fig, ax = self.new_subplots(
            1,
            len(width_ratios),
            figsize=self.figsize,
            layout="constrained",
            width_ratios=tuple(width_ratios),
        )

        display_arrays = spectrum_display_arrays(
            self._ratios_arrays,
            state["pf_smooth_method"],
            state["pf_smooth_width"],
            state["full_smooth_method"],
            state["full_smooth_width"],
            clip_negative=state["clip_negative"],
            normalize=state["normalize"],
        )
        plot_ratios_spectrum(
            ax[0],
            display_arrays,
            fontsize,
            standard_log_yticklabels=True,
            ylim=(10 ** state["ylims"][0], 10 ** state["ylims"][1]),
            show_first10_indicator=show_first10_indicator,
            leading_dims=state["first10"],
        )
        apply_legend(ax[0], state, fontsize, prefix="spectrum_legend", auto_loc=LEGEND_AUTO_LOCS["spectrum_legend"])
        draw_shared_reliable_annotation(
            ax[0],
            (state["annotation_x"], state["annotation_y"]),
            state["annotation_yoffset"],
            state["annotation_line_yoffset"],
            state["annotation_line_xpad"],
            fontsize,
            mode=state["annotation_mode"],
        )

        axis_index = 1
        ratio_axis = None
        if show_ratio:
            ratio_axis = ax[axis_index]
            axis_index += 1
            plot_ratios_beeswarms_concise(
                ratio_axis,
                self._ratios_arrays,
                fontsize,
                include_first10=show_first10_beeswarm,
                value_scale=value_scale,
                leading_dims=state["first10"],
            )

        familiarity_axes = []
        first10_axis = None
        if show_first10_by_env:
            first10_axis = ax[axis_index]
            if not shared_by_env:
                axis_index += 1
            if ratio_axis is not None:
                first10_axis.sharey(ratio_axis)
            render_familiarity_ratio_panel(
                first10_axis,
                self._curves_first10,
                "by_env",
                state["style_by_env"],
                fontsize,
                value_scale=value_scale,
            )
            familiarity_axes.append(first10_axis)

        all_axis = first10_axis if shared_by_env else ax[axis_index]
        if not shared_by_env:
            axis_index += 1
        share_axis = ratio_axis if ratio_axis is not None else first10_axis
        if share_axis is not None and all_axis is not share_axis:
            all_axis.sharey(share_axis)
        render_familiarity_ratio_panel(
            all_axis,
            self._curves_all,
            "by_env",
            state["style_by_env"],
            fontsize,
            value_scale=value_scale,
        )
        if all_axis not in familiarity_axes:
            familiarity_axes.append(all_axis)
        if first10_axis is not None and not shared_by_env:
            # Both familiarity axes use the same env colors, so keep the legend only on the full panel.
            first10_legend = first10_axis.get_legend()
            if first10_legend is not None:
                first10_legend.remove()
        handles, labels = all_axis.get_legend_handles_labels()
        if ratio_axis is not None:
            all_handles, all_labels = ratio_axis.get_legend_handles_labels()
            handles.extend(all_handles)
            labels.extend(all_labels)
        # A shared familiarity axis draws each environment label twice. Preserve the first
        # handle for each label so its legend remains identical to the separate-axis layout.
        unique_handles = {}
        for label, handle in zip(labels, handles):
            unique_handles.setdefault(label, handle)
        labels, handles = list(unique_handles), list(unique_handles.values())
        ordinal_labels = {"Env #1": "1st", "Env #2": "2nd", "Env #3": "3rd"}
        apply_legend(
            all_axis,
            state,
            fontsize,
            prefix="env_legend",
            handles=handles,
            labels=[ordinal_labels.get(label, label) for label in labels],
            auto_loc=LEGEND_AUTO_LOCS["env_legend"],
        )
        legend = all_axis.get_legend()
        if legend is not None:
            legend.set_title("Env #")
            legend.get_title().set_fontsize(fontsize * state["env_legend_fontsize_scale"])

        text_specs = [(first10_axis, "first10_text", "Head"), (all_axis, "all_text", "Full")] if show_first10_by_env else []
        for axis, prefix, text in text_specs:
            axis.text(
                state[f"{prefix}_x"],
                state[f"{prefix}_y"],
                text,
                transform=axis.transAxes,
                ha=state[f"{prefix}_ha"],
                va=state[f"{prefix}_va"],
                fontsize=fontsize,
            )

        if self.show_slope_panel:
            slope_axis = ax[axis_index]
            plot_env_slopes(
                slope_axis,
                self._slope_table,
                list(self._curves_all),
                state["slope_style"],
                fontsize,
                self._slope_stats_text,
            )

        ratio_ylims = tuple(limit * value_scale for limit in state["shared_variance_ratio_ylims"])
        leading_ratio_axis = ratio_axis if ratio_axis is not None else familiarity_axes[0]
        leading_ratio_axis.set_ylim(*ratio_ylims)
        leading_ratio_axis.spines["left"].set_bounds(*ratio_ylims)
        ylabel = "Placefield Component (%)" if state["as_percent"] else "Placefield Component"
        leading_ratio_axis.set_ylabel(ylabel, fontsize=fontsize)
        if ratio_axis is not None:
            hidden_y_axes = familiarity_axes
        else:
            hidden_y_axes = familiarity_axes[1:]
        for axis in hidden_y_axes:
            if axis is not leading_ratio_axis:
                axis.set_ylabel("")
                axis.spines["left"].set_visible(False)
                axis.tick_params(axis="y", left=False, labelleft=False)
                axis.spines["bottom"].set_position(("data", COMPOSITE_SPINE_OFFSET))

        return fig
