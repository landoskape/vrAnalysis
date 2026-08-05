"""The VR environments as rendered room stills over reward-zone tracks, and the speed panel beneath."""

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from dimensionality_manuscript.blender import RIG_HFOV_DEG, RenderParams, load_vr_room_images
from dimensionality_manuscript.configs.behavior_speed_env import ENV_REWARD_MAP, REFERENCE_ENV_LENGTH_CM
from dimensionality_manuscript.figure_scripts.panels import FigureViewer

from ._shared import env_slot_color
from .speed import MouseSpeedFocus

# Environments shown, top row first. These are the ATL cohort's three environments, ordered so
# the reward zone walks leftward down the figure (150 -> 100 -> 50 cm).
VR_SCHEMATIC_ENVS: tuple[int, ...] = (1, 3, 4)

# Reward-zone geometry is stored per session, not as a colony-wide constant: ENV_REWARD_MAP
# gives the zone *start* in cm, and the drawn width is a presentation choice.
VR_REWARD_ZONE_WIDTH_CM: float = 20.0
VR_REWARD_LEGEND_LABEL: str = "reward zones (90% of trials)"

# RenderParams fields, in the order the viewer registers them. Changing any of these makes the
# viewer shell out to Blender; every other parameter is pure matplotlib.
VR_RENDER_PARAMS: tuple[str, ...] = (
    "entrance_offset_cm",
    "hfov_deg",
    "panel_aspect",
    "panel_width_px",
    "camera_height_cm",
    "yaw_deg",
    "use_dof",
    "light_scale",
    "exposure",
    "samples",
)


class VREnvironmentSchematic(FigureViewer):
    """The VR environments as rendered room stills, stacked with reward-zone tracks.

    One row per environment: four panels showing what the mouse sees standing at the entrance of
    each room, and below them a track arrow with the reward zone marked. Arrow and zone take the
    environment's experience-slot color, the same palette the ``by_env`` panels of figure 3 use.
    The whole thing is drawn into a single axes in units of one panel height. Those relative
    units are converted to inches from the requested ``figsize``, so every gap, arrow, and swatch
    keeps its proportion under any scaling.

    Parameters split into two groups. The render parameters (see
    :class:`~dimensionality_manuscript.blender.RenderParams`) control the camera and are resolved
    by driving Blender headlessly; results are cached on disk, so revisiting a setting is instant
    but a new one costs a couple of seconds per environment. The layout parameters are plain
    matplotlib and redraw immediately.

    Parameters
    ----------
    envs : tuple of int
        Environments to draw, top row first.
    entrance_offset_cm : float
        Camera position in cm past each room's doorway plane. 0 sits in the doorway itself;
        negative values look into the room from the previous one.
    hfov_deg : float
        Horizontal field of view. The rig's real optics are ~152 deg (``RIG_HFOV_DEG``), which is
        heavily fisheyed; 90 deg reads better at panel size.
    panel_aspect : float
        Panel width / height. Also sets the rendered aspect, so the vertical field of view follows
        from it and ``hfov_deg``.
    panel_width_px : int
        Rendered panel width in pixels.
    camera_height_cm : float
        Eye height above the floor; the corridor walls are 15 cm tall.
    yaw_deg : float
        Camera rotation off the track axis. 0 looks straight down the corridor.
    use_dof : bool
        Enable the camera's depth of field, which softens the far end of the corridor.
    light_scale : float
        Multiplier on every light's energy -- changes shading contrast.
    exposure : float
        Color-management exposure in stops -- brightness only, shading untouched.
    samples : int
        EEVEE render samples.
    figsize : tuple of float
        Requested ``(width, height)`` in inches. See ``force_width`` for whether the height is a
        hard bound or is derived from the width.
    force_width : bool
        If True, the requested width determines inches per layout unit and the figure height is
        adjusted to preserve the schematic's aspect ratio. If False, both requested dimensions
        are hard bounds and the more constraining one determines the scale; unused space is
        centered along the other dimension.
    panel_height_in : float or None
        Deprecated compatibility input for older notebooks. When supplied without ``figsize``,
        it is converted once to an equivalent initial figure size and is not exposed as a Syd
        control.
    room_gap, env_gap, arrow_gap : float
        Gaps between panels in a row, between environment rows, and between a row's panels and its
        track arrow. In panel-height units.
    track_height : float
        Height of the arrow band and the reward-zone box, in panel-height units.
    margin : float
        Padding around the whole layout, in panel-height units.
    panel_border : float
        Line width of a black border around each panel; 0 draws none.
    show_reward_zones : bool
        Draw the reward-zone box on each track.
    reward_zone_width_cm : float
        Drawn width of the reward zone. Reward geometry is stored per session rather than as a
        colony constant, so only the zone start comes from the data; this is presentation.
    reward_zone_alpha : float
        Opacity of the reward-zone boxes, which take their environment's color.
    show_legend : bool
        Draw the reward-zone swatch and label below the last environment.
    legend_yoffset : float
        Extra vertical shift of the legend row, in panel-height units; positive pushes it further
        below the last track. The figure grows or shrinks to follow it.
    show_scalebar : bool
        Draw the track-length label at the right of the legend row.
    fontsize : float
        Font size in points for the legend and scale labels.

    Notes
    -----
    The layout is built in abstract units where 1 unit is one panel height. Writing ``n_rooms``
    for the rooms per environment (4) and ``n_envs`` for the number of rows::

        track_w   = n_rooms * panel_aspect + (n_rooms - 1) * room_gap
        row_pitch = 1 + arrow_gap + track_height + env_gap

        width_units  = track_w + 2 * margin
        height_units = n_envs * row_pitch - env_gap + 2 * margin
                       + (env_gap + track_height + legend_yoffset if show_legend else 0)

    So width responds to ``panel_aspect``, ``room_gap`` and ``margin``; height responds to
    ``arrow_gap``, ``track_height``, ``env_gap``, ``margin``, ``show_legend`` and
    ``legend_yoffset``. The inches-per-unit scale is then either::

        fig_width / width_units                                  # force_width=True
        min(fig_width / width_units, fig_height / height_units)  # force_width=False

    With ``force_width=True``, the actual height is ``height_units * scale``. With False, the
    requested figure size is retained and the schematic is centered in whichever dimension has
    spare room. The axes box always has the data aspect, so one data unit occupies the same
    physical size in both directions.

    Two things do not scale, because they are specified in points rather than layout units:
    ``fontsize`` and ``panel_border``. Doubling ``figsize`` leaves the legend text at the same
    physical size, so it reads as relatively smaller.
    """

    def __init__(
        self,
        *,
        envs: tuple[int, ...] = VR_SCHEMATIC_ENVS,
        entrance_offset_cm: float | None = None,
        hfov_deg: float | None = None,
        panel_aspect: float | None = None,
        panel_width_px: int | None = None,
        camera_height_cm: float | None = None,
        yaw_deg: float | None = None,
        use_dof: bool | None = None,
        light_scale: float | None = None,
        exposure: float | None = None,
        samples: int | None = None,
        figsize: tuple[float, float] | None = None,
        force_width: bool = False,
        panel_height_in: float | None = None,
        room_gap: float = 0.05,
        env_gap: float = 0.34,
        arrow_gap: float = 0.14,
        track_height: float = 0.16,
        margin: float = 0.06,
        panel_border: float = 0.0,
        show_reward_zones: bool = True,
        reward_zone_width_cm: float = VR_REWARD_ZONE_WIDTH_CM,
        reward_zone_alpha: float = 0.4,
        show_legend: bool = True,
        legend_yoffset: float = 0.0,
        show_scalebar: bool = True,
        fontsize: float = 9.0,
    ):
        self.envs = tuple(envs)

        # --- render parameters (each change re-renders in Blender) ---
        # RenderParams owns the camera defaults, so None here means "whatever it says" -- the
        # camera is its parameter set, and restating its values would be a second copy of them.
        defaults = RenderParams()

        def camera(name, override):
            return getattr(defaults, name) if override is None else override

        self.add_float("entrance_offset_cm", value=camera("entrance_offset_cm", entrance_offset_cm), min=-10.0, max=45.0)
        self.add_float("hfov_deg", value=camera("hfov_deg", hfov_deg), min=30.0, max=RIG_HFOV_DEG)
        self.add_float("panel_aspect", value=camera("panel_aspect", panel_aspect), min=0.6, max=4.0)
        self.add_integer("panel_width_px", value=camera("panel_width_px", panel_width_px), min=160, max=1600)
        self.add_float("camera_height_cm", value=camera("camera_height_cm", camera_height_cm), min=0.5, max=14.5)
        self.add_float("yaw_deg", value=camera("yaw_deg", yaw_deg), min=-90.0, max=90.0)
        self.add_boolean("use_dof", value=camera("use_dof", use_dof))
        self.add_float("light_scale", value=camera("light_scale", light_scale), min=0.1, max=5.0)
        self.add_float("exposure", value=camera("exposure", exposure), min=-3.0, max=3.0)
        self.add_integer("samples", value=camera("samples", samples), min=4, max=128)

        # --- layout parameters (immediate redraw) ---
        self.add_float("room_gap", value=room_gap, min=0.0, max=0.6)
        self.add_float("env_gap", value=env_gap, min=0.0, max=1.5)
        self.add_float("arrow_gap", value=arrow_gap, min=0.0, max=1.0)
        self.add_float("track_height", value=track_height, min=0.04, max=0.5)
        self.add_float("margin", value=margin, min=0.0, max=0.5)
        self.add_float("panel_border", value=panel_border, min=0.0, max=3.0)
        self.add_boolean("show_reward_zones", value=show_reward_zones)
        self.add_float("reward_zone_width_cm", value=reward_zone_width_cm, min=2.0, max=60.0)
        self.add_float("reward_zone_alpha", value=reward_zone_alpha, min=0.0, max=1.0)
        self.add_boolean("show_legend", value=show_legend)
        self.add_float("legend_yoffset", value=legend_yoffset, min=-0.5, max=1.5)
        self.add_boolean("show_scalebar", value=show_scalebar)
        self.add_float("fontsize", value=fontsize, min=4.0, max=24.0)

        # ``num_rooms`` is four for every supported Blender VR environment. Set the expected
        # value before rendering so a legacy panel-height scale can be converted to figsize;
        # refresh_data verifies the count returned by Blender and replaces it with the observed
        # value below.
        self.num_rooms = 4
        if figsize is not None and panel_height_in is not None:
            raise ValueError("Specify figsize or panel_height_in, not both")
        if figsize is None:
            scale = 0.85 if panel_height_in is None else panel_height_in
            metrics = self.layout_metrics(self.state)
            figsize = (metrics["width"] * scale, metrics["height"] * scale)

        # Figure dimensions are Syd controls; all other layout values remain relative units.
        self.add_float("fig_width", value=figsize[0], min=0.5, max=20.0, step=0.01)
        self.add_float("fig_height", value=figsize[1], min=0.5, max=20.0, step=0.01)
        self.add_boolean("force_width", value=force_width)

        self.on_change(list(VR_RENDER_PARAMS), self.refresh_data)
        self.refresh_data(self.state)

    def refresh_data(self, state):
        """Render (or fetch from cache) the room stills for every environment."""
        params = RenderParams(**{name: state[name] for name in VR_RENDER_PARAMS})
        self.images = {env: load_vr_room_images(env, params) for env in self.envs}

        room_counts = {env: len(images) for env, images in self.images.items()}
        if len(set(room_counts.values())) != 1:
            raise RuntimeError(f"Environments returned different room counts: {room_counts}. Every environment should have four rooms.")
        self.num_rooms = next(iter(room_counts.values()))

    def _env_color(self, env: int):
        """Experience-slot color of an environment, by its position in ``envs``."""
        return env_slot_color(self.envs.index(env))

    def layout_metrics(self, state) -> dict[str, float]:
        """Layout geometry in panel-height units, ahead of any ``panel_height_in`` scaling.

        Reused by :meth:`plot` (to size its own standalone figure) and by
        :func:`vr_schematic_and_speed` (to size the schematic's row within a shared figure)
        without duplicating the row/legend placement math.

        Returns
        -------
        dict
            ``track_w`` (the 0-to-track_w span the arrow/reward-zones occupy), ``width`` and
            ``height`` (the full axes extent including ``margin``), ``arrow_ys`` (per-environment
            arrow y-positions), ``panel_tops``, ``legend_y``, and ``y_bottom``.
        """
        panel_w = state["panel_aspect"]
        track_height = state["track_height"]
        env_gap = state["env_gap"]
        margin = state["margin"]

        # Everything here is in units of one panel height; the figure is then sized so that unit
        # is panel_height_in inches, which is what keeps the layout rigid under scaling.
        track_w = self.num_rooms * panel_w + (self.num_rooms - 1) * state["room_gap"]
        row_pitch = 1.0 + state["arrow_gap"] + track_height + env_gap
        panel_tops = [-row * row_pitch for row in range(len(self.envs))]
        arrow_ys = [top - 1.0 - state["arrow_gap"] - track_height / 2 for top in panel_tops]

        # The legend sits one environment-gap below the last track, in its own band, nudged by
        # legend_yoffset (positive pushes it further down).
        legend_y = arrow_ys[-1] - track_height / 2 - env_gap - track_height / 2 - state["legend_yoffset"]
        # min() rather than legend_y alone: a negative offset can lift the legend above the last
        # track, and the bottom edge has to follow whichever band ends up lowest.
        y_bottom = (min(legend_y, arrow_ys[-1]) if state["show_legend"] else arrow_ys[-1]) - track_height / 2

        return {
            "track_w": track_w,
            "width": track_w + 2 * margin,
            "height": (0.0 - y_bottom) + 2 * margin,
            "panel_tops": panel_tops,
            "arrow_ys": arrow_ys,
            "legend_y": legend_y,
            "y_bottom": y_bottom,
        }

    def fitted_figure_layout(
        self,
        state,
        *,
        figsize: tuple[float, float] | None = None,
        force_width: bool | None = None,
    ) -> dict[str, float | tuple[float, float] | tuple[float, float, float, float]]:
        """Convert relative layout units to inches and an aspect-correct axes rectangle.

        Parameters default to the corresponding Syd controls, but callers composing this panel
        into another figure may provide their own bounding box and policy. ``axes_bounds`` is in
        Matplotlib figure fractions.
        """
        metrics = self.layout_metrics(state)
        requested = (state["fig_width"], state["fig_height"]) if figsize is None else figsize
        use_width_only = state["force_width"] if force_width is None else force_width
        fig_width, fig_height = (float(requested[0]), float(requested[1]))
        if fig_width <= 0 or fig_height <= 0:
            raise ValueError(f"figsize dimensions must be positive, got {requested!r}")

        width_scale = fig_width / metrics["width"]
        if use_width_only:
            unit_in = width_scale
            actual_figsize = (fig_width, metrics["height"] * unit_in)
            axes_bounds = (0.0, 0.0, 1.0, 1.0)
        else:
            height_scale = fig_height / metrics["height"]
            unit_in = min(width_scale, height_scale)
            content_width = metrics["width"] * unit_in
            content_height = metrics["height"] * unit_in
            actual_figsize = (fig_width, fig_height)
            axes_width = content_width / fig_width
            axes_height = content_height / fig_height
            axes_bounds = ((1.0 - axes_width) / 2, (1.0 - axes_height) / 2, axes_width, axes_height)

        return {
            "figsize": actual_figsize,
            "axes_bounds": axes_bounds,
            "unit_in": unit_in,
            "content_width_in": metrics["width"] * unit_in,
            "content_height_in": metrics["height"] * unit_in,
        }

    def draw(self, state, ax):
        """Draw the schematic onto ``ax``, whose data limits are set to the layout's own units.

        Split out of :meth:`plot` so the combined schematic+speed figure
        (:func:`vr_schematic_and_speed`) can draw onto an externally placed axes instead of a
        standalone figure.
        """
        panel_w = state["panel_aspect"]
        room_gap = state["room_gap"]
        track_height = state["track_height"]
        margin = state["margin"]
        fontsize = state["fontsize"]
        show_legend = state["show_legend"]
        reward_zone_alpha = state["reward_zone_alpha"]

        metrics = self.layout_metrics(state)
        track_w = metrics["track_w"]
        legend_y = metrics["legend_y"]
        y_bottom = metrics["y_bottom"]

        # No ticks, no spines -- just the rendered content, filling whatever box ``ax`` occupies.
        ax.set_xlim(-margin, track_w + margin)
        ax.set_ylim(y_bottom - margin, margin)
        ax.set_axis_off()

        for env, panel_top, arrow_y in zip(self.envs, metrics["panel_tops"], metrics["arrow_ys"]):
            color = self._env_color(env)

            for room, image in enumerate(self.images[env]):
                x0 = room * (panel_w + room_gap)
                ax.imshow(image, extent=(x0, x0 + panel_w, panel_top - 1.0, panel_top), aspect="auto", zorder=2)
                if state["panel_border"] > 0:
                    ax.add_patch(
                        Rectangle(
                            (x0, panel_top - 1.0),
                            panel_w,
                            1.0,
                            facecolor="none",
                            edgecolor="black",
                            linewidth=state["panel_border"],
                            zorder=3,
                        )
                    )

            # Track arrow: the mouse runs left to right over the full environment length.
            ax.annotate(
                "",
                xy=(track_w, arrow_y),
                xytext=(0.0, arrow_y),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.2, shrinkA=0, shrinkB=0, mutation_scale=fontsize * 1.4),
                zorder=4,
            )

            if state["show_reward_zones"]:
                # ENV_REWARD_MAP holds the zone start in cm on the 200 cm reference track.
                start = track_w * ENV_REWARD_MAP[env] / REFERENCE_ENV_LENGTH_CM
                zone_width = track_w * state["reward_zone_width_cm"] / REFERENCE_ENV_LENGTH_CM
                ax.add_patch(
                    Rectangle(
                        (start, arrow_y - track_height / 2),
                        zone_width,
                        track_height,
                        facecolor=color,
                        alpha=reward_zone_alpha,
                        edgecolor="none",
                        zorder=5,
                    )
                )

        if show_legend:
            # The zones are environment-colored, so the swatch is split into one segment per
            # environment rather than drawn in a neutral grey that matches nothing on the plot.
            swatch_w = 0.55 * panel_w / 1.6  # scales with panel width so the row stays balanced
            segment_w = swatch_w / len(self.envs)
            for segment, env in enumerate(self.envs):
                ax.add_patch(
                    Rectangle(
                        (segment * segment_w, legend_y - track_height / 2),
                        segment_w,
                        track_height,
                        facecolor=self._env_color(env),
                        alpha=reward_zone_alpha,
                        edgecolor="none",
                        zorder=5,
                    )
                )
            ax.text(swatch_w + 0.12, legend_y, VR_REWARD_LEGEND_LABEL, ha="left", va="center", fontsize=fontsize, zorder=5)

        if state["show_scalebar"]:
            ax.text(
                track_w,
                legend_y if show_legend else y_bottom,
                f"{REFERENCE_ENV_LENGTH_CM:g}cm",
                ha="right",
                va="center",
                fontsize=fontsize,
                zorder=5,
            )

        # Set last: imshow(aspect="auto") resets the axes aspect on every call, so an earlier
        # set_aspect would be silently undone. The box this axes occupies was sized from the same
        # layout (see plot / vr_schematic_and_speed), so this adds no padding -- it just makes the
        # units isotropic if that box's aspect were ever overridden.
        ax.set_aspect("equal")

    def plot(self, state):
        fitted = self.fitted_figure_layout(state)
        fig = self.new_figure(figsize=fitted["figsize"])
        self.draw(state, fig.add_axes(fitted["axes_bounds"]))
        return fig


def vr_schematic_and_speed(
    schematic: VREnvironmentSchematic,
    speed: MouseSpeedFocus,
    *,
    fig_width: float = 3.5,
    fig_height: float = 6.0,
    left_margin: float = 0.16,
    right_margin: float = 0.98,
    top_margin: float = 0.02,
    bottom_margin: float = 0.14,
    panel_gap: float = 0.03,
):
    """
    Stack the VR schematic over the speed panel on a shared physical x-axis.

    The two panels line up so the reward zones in the schematic sit above the reward zones drawn
    on the speed curves. Both are drawn by the viewers' own ``draw`` methods onto axes this
    function places, so each panel keeps whatever state its viewer is in -- construct them with
    the keywords you want, or tune them interactively first and pass them here.

    Sizing is driven entirely by ``fig_width`` and ``fig_height``. The schematic has a fixed
    aspect ratio (:meth:`VREnvironmentSchematic.layout_metrics`), so its shared
    :meth:`VREnvironmentSchematic.fitted_figure_layout` calculation is run in width-forced mode:
    given the shared axes width (``fig_width * (right_margin - left_margin)``), its height is
    fully determined. That height becomes the top row's share of ``fig_height``, and the speed
    panel takes what's left. Both
    axes share ``left_margin``/``right_margin`` so their plotted x-range spans identical
    figure-fraction width -- necessary for x-alignment, since the schematic has no y-axis gutter
    of its own but the speed panel needs one for its ticks and label. The schematic's standalone
    ``fig_width``, ``fig_height``, and ``force_width`` controls are intentionally ignored because
    this composed figure owns the outer canvas.

    Parameters
    ----------
    schematic : VREnvironmentSchematic
        Viewer drawn on the top row.
    speed : MouseSpeedFocus
        Viewer drawn on the bottom row.
    fig_width, fig_height : float
        Figure size in inches. Both panels are forced to fit inside it.
    left_margin, right_margin : float
        Shared horizontal extent of both axes, as a fraction of ``fig_width``. Sized to leave room
        for the speed panel's y-axis label and ticks; the schematic donates the same margin on
        either side even though it doesn't need it, so the two axes' x=0 and
        x=``REFERENCE_ENV_LENGTH_CM`` line up.
    top_margin, bottom_margin : float
        Figure-fraction space left above the schematic and below the speed panel's x-axis label
        and ticks.
    panel_gap : float
        Figure-fraction vertical gap between the two panels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    schem_state = schematic.state
    speed_state = speed.state

    metrics = schematic.layout_metrics(schem_state)
    axes_width_frac = right_margin - left_margin
    # The combined panel must fill the shared x extent for its track to align with the speed
    # axis. Reuse the standalone sizing engine in width-forced mode instead of duplicating its
    # relative-units-to-inches conversion here.
    fitted = schematic.fitted_figure_layout(
        schem_state,
        figsize=(axes_width_frac * fig_width, fig_height),
        force_width=True,
    )
    schem_height_in = fitted["content_height_in"]
    schem_height_frac = schem_height_in / fig_height

    avail_frac = 1.0 - top_margin - bottom_margin - panel_gap
    if schem_height_frac >= avail_frac:
        raise ValueError(
            f"Schematic needs {schem_height_in:.2f} in of height ({schem_height_frac:.1%} of "
            f"fig_height={fig_height}), leaving no room for the speed panel ({avail_frac:.1%} "
            "available). Increase fig_height or fig_width, or shrink the schematic (e.g. a "
            "smaller panel_aspect or margin)."
        )
    speed_height_frac = avail_frac - schem_height_frac

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax_schem = fig.add_axes([left_margin, 1.0 - top_margin - schem_height_frac, axes_width_frac, schem_height_frac])
    ax_speed = fig.add_axes([left_margin, bottom_margin, axes_width_frac, speed_height_frac])

    schematic.draw(schem_state, ax_schem)
    speed.draw(speed_state, ax_speed)

    # The schematic's track (arrow + reward zones) spans data units [0, track_w] for a physical
    # length of REFERENCE_ENV_LENGTH_CM; converting its margin to the same cm scale and applying
    # it to the speed panel's xlim keeps the two axes' x=0 and x=REFERENCE_ENV_LENGTH_CM aligned
    # regardless of margin, since both axes occupy the same figure-fraction width.
    cm_per_unit = REFERENCE_ENV_LENGTH_CM / metrics["track_w"]
    margin_cm = schem_state["margin"] * cm_per_unit
    ax_speed.set_xlim(-margin_cm, REFERENCE_ENV_LENGTH_CM + margin_cm)

    return fig
