"""Rounded-box schematics of the external / internal / neural model zoo."""

from dataclasses import dataclass, replace

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from dimensionality_manuscript.figure_scripts.panels import FigureViewer

from .performance import STRUCTURED_ADDITIVE_MODEL_COLOR


@dataclass
class ModelZooSchematicConfig:
    """Fully tunable layout/style for the model-zoo schematics.

    Coordinates are in abstract "data units"; the axis uses an equal aspect ratio, so a
    unit is the same horizontally and vertically. Every box shares ``box_width`` (except the
    gain box, which uses ``gain_width``); the source/target rows share ``box_height`` while
    the latent row is scaled by ``latent_height_scale``.

    Corner radii behave like CSS ``border-radius`` (in data units) and are clamped so they
    never exceed half of a box's smaller side.
    """

    # --- Shared box geometry -----------------------------------------------------------
    box_width: float = 2.0  # shared width of every standard box
    box_height: float = 1.0  # shared height of the source and target rows
    latent_height_scale: float = 1.65  # latent (middle) row height as a multiple of box_height
    gain_width: float = 1.2  # the single box with a unique width

    # --- Spacing (data units) ----------------------------------------------------------
    col_gap: float = 0.25  # horizontal gap between columns
    row_gap: float = 0.5  # vertical gap between rows (leaves room for arrows)
    panel_gap: float = 0.3  # gap between the three containers (defaults to col_gap)
    container_pad: float = 0.3  # padding between boxes and their container edge
    label_pad: float = 0.75  # extra room on the labelled side for the container title (sets container height)
    label_gap: float = 0.45  # gap between the boxes and the container label (the pad beneath the label)
    gain_separator_gap: float = 0.2  # extra horizontal space left of the gain model for its dotted divider
    neural_label_offset: float = 0.18  # horizontal nudge of each neural label off the arrow, as a fraction of box_width

    # --- Rounded corners (data units, like CSS border-radius) --------------------------
    box_corner_radius: float = 0.18
    container_corner_radius: float = 0.4

    # --- Container fill colors ---------------------------------------------------------
    external_container_color: str = "#d9d9d9"
    internal_container_color: str = "#f4b0ab"
    neural_container_color: str = "#b8bbe8"

    # --- Box fill colors ---------------------------------------------------------------
    black_box_color: str = "#000000"
    red_box_color: str = "#c00000"
    blue_box_color: str = "#0000cd"
    box_text_color: str = "#ffffff"
    container_label_color: str = "#000000"

    # --- Arrow colors (one per panel) --------------------------------------------------
    external_arrow_color: str = "#000000"
    internal_arrow_color: str = "#c00000"
    neural_arrow_color: str = "#0000cd"
    gain_second_arrow_color: str = "#000000"  # the "other" model in the two-model gain column

    # --- Line / arrow styling ----------------------------------------------------------
    arrow_linewidth: float = 2.2
    arrow_mutation_scale: float = 10.0  # arrowhead size
    junction_dot_size: float = 4.0  # multiplicative-gain junction marker
    junction_dot_offset: float = 0.15  # rightward shift of the red junction dot off the arrow, as a fraction of box_width
    gain_two_model_offset: float = 0.1  # half-separation of the paired gain arrows, as a fraction of box_width
    gain_route_hoffset: float = 0.10  # horizontal separation of the black/red gain routes, as a fraction of box_width
    gain_route_voffset: float = 0.10  # vertical separation of the black/red gain routes, as a fraction of box_height
    gain_separator_linewidth: float = 1.6  # dotted divider left of the gain model

    # --- Condensed schematic geometry -------------------------------------------------
    condensed_lane_gap: float = 0.3  # gap between boxes within one condensed model
    condensed_model_gap: float = 0.8  # gap between complete condensed models
    structured_bypass_offset: float = 0.45  # clearance right of PF for the source -> target bypass

    # --- Fonts -------------------------------------------------------------------------
    box_fontsize: float = 13.0
    container_label_fontsize: float = 15.0
    arrow_label_fontsize: float = 13.0

    # --- Figure ------------------------------------------------------------------------
    figsize: tuple[float, float] = (18.0, 5.0)
    background_color: str = "#ffffff"


@dataclass
class ModelZooUltraCondensedConfig:
    """Layout and style for :class:`ModelZooUltraCondensed`.

    The ultra-condensed schematic overlays the placefield, global-gain, and shared-residual
    models into one group, then places the peer prediction beside it. Coordinates use equal-
    aspect data units, just like :class:`ModelZooSchematicConfig`.
    """

    # --- Box geometry -----------------------------------------------------------------
    box_width: float = 2.0
    box_height: float = 1.0
    latent_height_scale: float = 1.65
    gain_width: float = 1.2

    # --- Layout -----------------------------------------------------------------------
    row_gap: float = 0.5
    lane_gap: float = 0.3
    group_gap: float = 1.1
    structured_bypass_offset: float = 0.45
    label_column_width: float = 3.5
    label_gap: float = 0.6
    label_line_spacing: float = 0.62

    # --- Colors -----------------------------------------------------------------------
    placefield_color: str = "#000000"
    gain_color: str = "#c00000"
    structured_additive_color: str = STRUCTURED_ADDITIVE_MODEL_COLOR
    peer_color: str = "#0000cd"
    box_text_color: str = "#ffffff"
    background_color: str = "#ffffff"

    # --- Lines and corners ------------------------------------------------------------
    box_corner_radius: float = 0.18
    arrow_linewidth: float = 2.2
    arrow_mutation_scale: float = 10.0
    junction_dot_size: float = 4.0
    junction_dot_offset: float = 0.15  # leftward from PF -> target, as a fraction of box_width
    horizontal_connections_as_lines: bool = False  # pos-PF and the orange residual route
    vertical_connections_as_lines: bool = False  # PF backbone, source-gain, and peer source-target

    # --- Typography / figure ----------------------------------------------------------
    box_fontsize: float = 13.0
    label_fontsize: float = 13.0
    figsize: tuple[float, float] = (12.0, 5.0)


# Text as it appears on the source slide. The four model boxes are shared verbatim by the
# external and internal panels; explicit line breaks reproduce the slide's wrapping.
_ZOO_MODEL_LABELS = ["placefield", "high-d\nposition", "high-d pos\n+speed", "high-d pos\n+speed\n+reward"]

# Slider ranges for every tunable ModelZooSchematicConfig field, shared by the schematic viewers.
_ZOO_TUNABLE_LIMITS = {
    "box_width": (0.5, 4.0),
    "box_height": (0.5, 3.0),
    "latent_height_scale": (1.0, 2.5),
    "gain_width": (0.3, 3.0),
    "col_gap": (0.0, 2.0),
    "row_gap": (0.1, 3.0),
    "panel_gap": (0.0, 4.0),
    "container_pad": (0.0, 2.0),
    "label_pad": (0.0, 2.5),
    "label_gap": (0.0, 2.0),
    "gain_separator_gap": (0.0, 2.0),
    "neural_label_offset": (0.0, 0.5),
    "box_corner_radius": (0.0, 0.5),
    "container_corner_radius": (0.0, 1.0),
    "arrow_linewidth": (0.5, 6.0),
    "arrow_mutation_scale": (5.0, 40.0),
    "junction_dot_size": (2.0, 24.0),
    "junction_dot_offset": (0.0, 0.5),
    "gain_two_model_offset": (0.0, 0.4),
    "gain_route_hoffset": (0.0, 0.3),
    "gain_route_voffset": (0.0, 0.5),
    "condensed_lane_gap": (0.0, 2.0),
    "condensed_model_gap": (0.0, 3.0),
    "structured_bypass_offset": (0.05, 2.0),
    "box_fontsize": (6.0, 28.0),
    "container_label_fontsize": (6.0, 30.0),
}

# ModelZooSchematicConfig fields exposed as live Syd controls by the full schematic.
_ZOO_TUNABLES = [
    "box_width",
    "box_height",
    "latent_height_scale",
    "gain_width",
    "row_gap",
    "panel_gap",
    "container_pad",
    "label_pad",
    "label_gap",
    "gain_separator_gap",
    "neural_label_offset",
    "box_corner_radius",
    "container_corner_radius",
    "arrow_linewidth",
    "arrow_mutation_scale",
    "junction_dot_size",
    "gain_two_model_offset",
    "junction_dot_offset",
    "gain_route_hoffset",
    "gain_route_voffset",
    "box_fontsize",
    "container_label_fontsize",
]


def _zoo_box(ax, cx, cy, w, h, text, facecolor, cfg: ModelZooSchematicConfig):
    """Draw a rounded box of total size ``w`` x ``h`` centered at ``(cx, cy)`` with centered text."""
    r = min(cfg.box_corner_radius, 0.5 * min(w, h) - 1e-6)
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2 + r, cy - h / 2 + r),
            w - 2 * r,
            h - 2 * r,
            boxstyle=f"round,pad={r},rounding_size={r}",
            mutation_scale=1.0,
            facecolor=facecolor,
            edgecolor="none",
            zorder=2,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", color=cfg.box_text_color, fontsize=cfg.box_fontsize, zorder=3)


def _zoo_container(ax, x0, y0, x1, y1, color, cfg: ModelZooSchematicConfig):
    """Draw a rounded container spanning ``[x0, x1] x [y0, y1]`` behind the boxes."""
    r = min(cfg.container_corner_radius, 0.5 * min(x1 - x0, y1 - y0) - 1e-6)
    ax.add_patch(
        FancyBboxPatch(
            (x0 + r, y0 + r),
            (x1 - x0) - 2 * r,
            (y1 - y0) - 2 * r,
            boxstyle=f"round,pad={r},rounding_size={r}",
            mutation_scale=1.0,
            facecolor=color,
            edgecolor="none",
            zorder=0,
        )
    )


def _zoo_arrow(ax, x0, y0, x1, y1, color, cfg: ModelZooSchematicConfig):
    """Draw a straight arrow from ``(x0, y0)`` to ``(x1, y1)``."""
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=cfg.arrow_mutation_scale,
            color=color,
            lw=cfg.arrow_linewidth,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def _zoo_line(ax, xs, ys, color, cfg: ModelZooSchematicConfig):
    """Draw a (possibly elbowed) plain connector through the given points."""
    ax.plot(xs, ys, color=color, lw=cfg.arrow_linewidth, solid_capstyle="round", solid_joinstyle="round", zorder=1)


class ModelZooSchematic(FigureViewer):
    """Interactive rounded-box schematic of the external / internal / neural model zoo.

    The three panels reproduce the source slide: an external panel (model -> target), an
    internal panel adding a source row and a multiplicatively-wired gain box, and a neural
    reduced-rank-regression panel. All geometry, colors, corner radii, and fonts come from
    a :class:`ModelZooSchematicConfig`; the numeric fields in ``_ZOO_TUNABLES`` are exposed
    as live sliders.

    Parameters
    ----------
    config : ModelZooSchematicConfig or None
        Full style/layout config. A default one is created when None.
    """

    _TUNABLES = _ZOO_TUNABLES

    def __init__(self, config: ModelZooSchematicConfig | None = None):
        self.cfg = self.add_controls(self, config)

    @classmethod
    def add_controls(cls, viewer, config: ModelZooSchematicConfig | None = None) -> ModelZooSchematicConfig:
        """Add this schematic's controls to ``viewer`` and return the base config."""
        config = config or ModelZooSchematicConfig()
        for name in cls._TUNABLES:
            lo, hi = _ZOO_TUNABLE_LIMITS[name]
            viewer.add_float(name, value=float(getattr(config, name)), min=lo, max=hi)
        return config

    @classmethod
    def config_from_state(cls, config: ModelZooSchematicConfig, state) -> ModelZooSchematicConfig:
        """Apply this schematic's live control values to ``config``."""
        return replace(config, **{name: state[name] for name in cls._TUNABLES})

    def plot(self, state):
        cfg = self.config_from_state(self.cfg, state)
        bw = cfg.box_width
        h = cfg.box_height  # source / target rows
        hm = cfg.box_height * cfg.latent_height_scale  # taller latent (middle) row
        cg, rg, pad = cfg.col_gap, cfg.row_gap, cfg.container_pad

        # Row centers (target row bottom sits at y = 0). The latent row is taller, so
        # neighbours are spaced by the appropriate half-heights. External uses only the
        # latent + target rows; internal/neural add a source row on top, so all panels align.
        y_target = h / 2
        y_model = y_target + h / 2 + rg + hm / 2
        y_source = y_model + hm / 2 + rg + h / 2

        # Row edges reused throughout.
        src_top, src_bot = y_source + h / 2, y_source - h / 2
        mid_top, mid_bot = y_model + hm / 2, y_model - hm / 2
        tgt_top, tgt_bot = y_target + h / 2, y_target - h / 2

        fig, ax = self.new_subplots(figsize=cfg.figsize, layout="constrained")
        fig.patch.set_facecolor(cfg.background_color)
        ax.set_aspect("equal")
        ax.axis("off")

        black, red, blue = cfg.black_box_color, cfg.red_box_color, cfg.blue_box_color

        # ---- External panel (4 columns, latent -> target) ----------------------------
        ext_left = 0.0
        ext_cx = [ext_left + bw / 2 + i * (bw + cg) for i in range(4)]
        ext_right = ext_cx[-1] + bw / 2
        for cx, label in zip(ext_cx, _ZOO_MODEL_LABELS):
            _zoo_box(ax, cx, y_model, bw, hm, label, black, cfg)
            _zoo_box(ax, cx, y_target, bw, h, "target", black, cfg)
            _zoo_arrow(ax, cx, mid_bot, cx, tgt_top, cfg.external_arrow_color, cfg)
        ext_outer = (ext_left - pad, tgt_bot - pad, ext_right + pad, mid_top + pad + cfg.label_pad)
        _zoo_container(ax, *ext_outer, cfg.external_container_color, cfg)
        ax.text(
            (ext_outer[0] + ext_outer[2]) / 2,
            mid_top + cfg.label_gap,
            "external models",
            ha="center",
            va="center",
            color=cfg.container_label_color,
            fontsize=cfg.container_label_fontsize,
            zorder=3,
        )

        # ---- Internal panel (5 columns, source -> latent -> target, + gain) ----------
        int_content_left = ext_outer[2] + cfg.panel_gap + pad
        # Columns 0-3 are evenly spaced; an extra gap before column 4 sets off the gain model.
        int_cx = []
        x = int_content_left + bw / 2
        for i in range(5):
            int_cx.append(x)
            x += bw + cg + (cfg.gain_separator_gap if i == 3 else 0.0)
        gain_cx = int_cx[-1] + bw / 2 + cg + cfg.gain_width / 2
        int_right = gain_cx + cfg.gain_width / 2
        ai = cfg.internal_arrow_color
        cx5 = int_cx[-1]
        off = cfg.gain_two_model_offset * bw
        for i, cx in enumerate(int_cx):
            is_last = i == 4
            model_color = black if is_last else red  # last column is the neural/gain model (black boxes)
            model_label = "placefield" if is_last else _ZOO_MODEL_LABELS[i]
            _zoo_box(ax, cx, y_source, bw, h, "source", red, cfg)  # source is always red
            _zoo_box(ax, cx, y_model, bw, hm, model_label, model_color, cfg)
            _zoo_box(ax, cx, y_target, bw, h, "target", model_color, cfg)
            # In the gain column the source -> latent arrow shares the vertical slice of the
            # (offset) red latent -> target arrow below it.
            src_x = cx + off if is_last else cx
            _zoo_arrow(ax, src_x, src_bot, src_x, mid_top, ai, cfg)
            if not is_last:
                _zoo_arrow(ax, cx, mid_bot, cx, tgt_top, ai, cfg)

        # Dotted divider setting the gain model off as its own sub-group (spans source..target).
        x_sep = (int_cx[3] + bw / 2 + cx5 - bw / 2) / 2
        ax.plot([x_sep, x_sep], [tgt_bot, src_top], linestyle=":", color=black, lw=cfg.gain_separator_linewidth, zorder=1)

        # Gain box wiring. Two paired latent -> target arrows carry two models: a plain
        # placefield model (black, left) and a gain-modulated one (red, right). The gain is
        # fed from the source and modulates BOTH: each colored route runs source -> gain ->
        # its own arrow. Horizontal/vertical route separation are independent knobs.
        black_arrow = cfg.gain_second_arrow_color
        hgr = cfg.gain_route_hoffset * bw
        vgr = cfg.gain_route_voffset * h
        _zoo_box(ax, gain_cx, y_model, cfg.gain_width, hm, "gain", red, cfg)  # gain is always red
        _zoo_arrow(ax, cx5 - off, mid_bot, cx5 - off, tgt_top, black_arrow, cfg)
        _zoo_arrow(ax, cx5 + off, mid_bot, cx5 + off, tgt_top, ai, cfg)

        # Source -> gain, one route per model (red high/right, black low/left).
        _zoo_line(ax, [cx5 + bw / 2, gain_cx + hgr], [y_source + vgr, y_source + vgr], ai, cfg)
        _zoo_arrow(ax, gain_cx + hgr, y_source + vgr, gain_cx + hgr, mid_top, ai, cfg)
        _zoo_line(ax, [cx5 + bw / 2, gain_cx - hgr], [y_source - vgr, y_source - vgr], black_arrow, cfg)
        _zoo_arrow(ax, gain_cx - hgr, y_source - vgr, gain_cx - hgr, mid_top, black_arrow, cfg)

        # Gain -> junction dots. Both dots share the same leftmost x; black sits upper, red
        # lower. To nest without crossing, the upper (black) route is the inner one (exits the
        # gain's left, turns higher) and the lower (red) route is the outer one (exits the
        # right, turns lower) -- also keeping each color on one side of the gain throughout.
        y_junction = (mid_bot + tgt_top) / 2
        dot_x = cx5 + off + cfg.junction_dot_offset * bw
        y_black, y_red = y_junction + vgr, y_junction - vgr
        _zoo_line(ax, [gain_cx - hgr, gain_cx - hgr, dot_x], [mid_bot, y_black, y_black], black_arrow, cfg)
        ax.plot(dot_x, y_black, marker="o", markersize=cfg.junction_dot_size, color=black_arrow, zorder=4)
        _zoo_line(ax, [gain_cx + hgr, gain_cx + hgr, dot_x], [mid_bot, y_red, y_red], ai, cfg)
        ax.plot(dot_x, y_red, marker="o", markersize=cfg.junction_dot_size, color=ai, zorder=4)

        int_outer = (int_content_left - pad, tgt_bot - pad, int_right + pad, src_top + pad + cfg.label_pad)
        _zoo_container(ax, *int_outer, cfg.internal_container_color, cfg)
        ax.text(
            (int_outer[0] + int_outer[2]) / 2,
            src_top + cfg.label_gap,
            "internal models",
            ha="center",
            va="center",
            color=cfg.container_label_color,
            fontsize=cfg.container_label_fontsize,
            zorder=3,
        )

        # ---- Neural panel (source -> target; label superimposed inside the column) ----
        # No extra horizontal gutter: the box column uses the same padding as the others,
        # and the rotated label is drawn over the source->target gap, nudged right of the arrow.
        neu_content_left = int_outer[2] + cfg.panel_gap + pad
        neu_cx = neu_content_left + bw / 2
        an = cfg.neural_arrow_color
        _zoo_box(ax, neu_cx, y_source, bw, h, "source", blue, cfg)
        _zoo_box(ax, neu_cx, y_target, bw, h, "target", black, cfg)
        _zoo_arrow(ax, neu_cx, src_bot, neu_cx, tgt_top, an, cfg)
        # Two independent labels flanking the arrow: "reduced rank" left, "regression" right.
        y_mid = (y_source + y_target) / 2
        label_kwargs = dict(
            ha="center",
            va="center",
            rotation=90,
            color=cfg.container_label_color,
            fontsize=cfg.arrow_label_fontsize,
            zorder=3,
        )
        ax.text(neu_cx - cfg.neural_label_offset * bw, y_mid, "reduced rank", **label_kwargs)
        ax.text(neu_cx + cfg.neural_label_offset * bw, y_mid, "regression", **label_kwargs)
        neu_outer = (neu_content_left - pad, tgt_bot - pad, neu_cx + bw / 2 + pad, src_top + pad + cfg.label_pad)
        _zoo_container(ax, *neu_outer, cfg.neural_container_color, cfg)
        ax.text(
            (neu_outer[0] + neu_outer[2]) / 2,
            src_top + cfg.label_gap,
            "neural model",
            ha="center",
            va="center",
            color=cfg.container_label_color,
            fontsize=cfg.container_label_fontsize,
            zorder=3,
        )

        margin = 0.3
        ax.set_xlim(ext_outer[0] - margin, neu_outer[2] + margin)
        ax.set_ylim(
            min(ext_outer[1], int_outer[1], neu_outer[1]) - margin,
            max(int_outer[3], neu_outer[3]) + margin,
        )
        return fig


# ======================================================================================
# Condensed model-zoo schematic (one column per model type)
# ======================================================================================

# Color identifies each model throughout its complete diagram.
_ZOO_CONDENSED_TITLES = ("PF", "PF+Gain", "Structured\nAdditive", "Peer")

# ModelZooSchematicConfig fields exposed as live Syd controls. The condensed layout has no
# containers and a single gain route, so the container/panel and paired-route fields drop out.
_ZOO_CONDENSED_TUNABLES = [
    "box_width",
    "box_height",
    "latent_height_scale",
    "gain_width",
    "col_gap",
    "row_gap",
    "condensed_lane_gap",
    "condensed_model_gap",
    "structured_bypass_offset",
    "box_corner_radius",
    "arrow_linewidth",
    "arrow_mutation_scale",
    "junction_dot_size",
    "junction_dot_offset",
    "box_fontsize",
]


class ModelZooCondensed(ModelZooSchematic):
    """Condensed PF / PF+gain / structured-additive / peer model schematic.

    PF models decode source activity into position on the left, use position to select a
    placefield prediction on the right, and pass that prediction to the target. PF+gain adds
    a left-hand gain branch that modulates the PF-to-target arrow. Structured additive adds
    a direct source-to-target bypass around the right of the standard internal-PF path.

    Style comes from the same :class:`ModelZooSchematicConfig` as :class:`ModelZooSchematic`,
    whose container and panel fields are unused here; ``_ZOO_CONDENSED_TUNABLES`` are exposed
    as live sliders.
    """

    _TUNABLES = _ZOO_CONDENSED_TUNABLES

    def plot(self, state):
        cfg = self.config_from_state(self.cfg, state)
        fig, ax = self.new_subplots(figsize=cfg.figsize, layout="constrained")
        fig.patch.set_facecolor(cfg.background_color)
        self.draw(ax, cfg)
        return fig

    @staticmethod
    def draw(ax, cfg: ModelZooSchematicConfig):
        """Draw the condensed schematic on an existing Matplotlib axis."""
        bw = cfg.box_width
        h = cfg.box_height  # source / target rows
        hm = cfg.box_height * cfg.latent_height_scale  # taller latent (middle) row
        lane_gap = cfg.condensed_lane_gap
        model_gap = cfg.condensed_model_gap
        rg = cfg.row_gap

        # Row centers (target row bottom sits at y = 0), matching ModelZooSchematic.
        y_target = h / 2
        y_model = y_target + h / 2 + rg + hm / 2
        y_source = y_model + hm / 2 + rg + h / 2
        src_top, src_bot = y_source + h / 2, y_source - h / 2
        mid_top, mid_bot = y_model + hm / 2, y_model - hm / 2
        tgt_top, tgt_bot = y_target + h / 2, y_target - h / 2

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        model_colors = (cfg.black_box_color, cfg.red_box_color, cfg.red_box_color, cfg.blue_box_color)

        def draw_internal_pf(left: float, color: str) -> tuple[float, float]:
            """Draw source/position left of PF/target and return its horizontal bounds."""
            left_cx = left + bw / 2
            right_cx = left_cx + bw + lane_gap
            _zoo_box(ax, left_cx, y_source, bw, h, "source", color, cfg)
            _zoo_box(ax, left_cx, y_model, bw, hm, "pos", color, cfg)
            _zoo_box(ax, right_cx, y_model, bw, hm, "PF", color, cfg)
            _zoo_box(ax, right_cx, y_target, bw, h, "target", color, cfg)
            _zoo_arrow(ax, left_cx, src_bot, left_cx, mid_top, color, cfg)
            _zoo_arrow(ax, left_cx + bw / 2, y_model, right_cx - bw / 2, y_model, color, cfg)
            _zoo_arrow(ax, right_cx, mid_bot, right_cx, tgt_top, color, cfg)
            return left, right_cx + bw / 2

        # PF: source -> position on the left, then position -> PF -> target on the right.
        pf_left = 0.0
        pf_bounds = draw_internal_pf(pf_left, model_colors[0])

        # PF+Gain: source branches down to gain (leftmost) and position (middle); position
        # drives PF on the right, while gain comes down and right onto PF -> target.
        gain_left = pf_bounds[1] + model_gap
        gain_cx = gain_left + cfg.gain_width / 2
        pos_cx = gain_left + cfg.gain_width + lane_gap + bw / 2
        pf_gain_cx = pos_cx + bw + lane_gap
        gain_color = model_colors[1]
        _zoo_box(ax, pos_cx, y_source, bw, h, "source", gain_color, cfg)
        _zoo_box(ax, gain_cx, y_model, cfg.gain_width, hm, "gain", gain_color, cfg)
        _zoo_box(ax, pos_cx, y_model, bw, hm, "pos", gain_color, cfg)
        _zoo_box(ax, pf_gain_cx, y_model, bw, hm, "PF", gain_color, cfg)
        _zoo_box(ax, pf_gain_cx, y_target, bw, h, "target", gain_color, cfg)
        _zoo_arrow(ax, pos_cx, src_bot, pos_cx, mid_top, gain_color, cfg)
        branch_y = (src_bot + mid_top) / 2
        _zoo_line(ax, [pos_cx, pos_cx, gain_cx], [src_bot, branch_y, branch_y], gain_color, cfg)
        _zoo_arrow(ax, gain_cx, branch_y, gain_cx, mid_top, gain_color, cfg)
        _zoo_arrow(ax, pos_cx + bw / 2, y_model, pf_gain_cx - bw / 2, y_model, gain_color, cfg)
        _zoo_arrow(ax, pf_gain_cx, mid_bot, pf_gain_cx, tgt_top, gain_color, cfg)

        # Gain -> junction: the dot marks multiplicative influence on the PF -> target arrow.
        y_junction = (mid_bot + tgt_top) / 2
        dot_x = pf_gain_cx + cfg.junction_dot_offset * bw
        _zoo_line(ax, [gain_cx, gain_cx, dot_x], [mid_bot, y_junction, y_junction], gain_color, cfg)
        ax.plot(
            dot_x,
            y_junction,
            marker="o",
            markersize=cfg.junction_dot_size,
            color=gain_color,
            zorder=4,
        )

        gain_bounds = (gain_left, pf_gain_cx + bw / 2)

        # Structured additive: standard internal PF plus a direct source -> target route
        # wrapping around its right-hand side as a sideways U.
        structured_left = gain_bounds[1] + model_gap
        structured_bounds = draw_internal_pf(structured_left, model_colors[2])
        structured_source_cx = structured_left + bw / 2
        structured_target_cx = structured_source_cx + bw + lane_gap
        bypass_x = structured_bounds[1] + cfg.structured_bypass_offset
        _zoo_line(
            ax,
            [structured_source_cx + bw / 2, bypass_x, bypass_x],
            [y_source, y_source, y_target],
            model_colors[2],
            cfg,
        )
        _zoo_arrow(
            ax,
            bypass_x,
            y_target,
            structured_target_cx + bw / 2,
            y_target,
            model_colors[2],
            cfg,
        )

        # Peer model retains the direct neural latent stack.
        peer_left = bypass_x + model_gap
        peer_cx = peer_left + bw / 2
        peer_color = model_colors[3]
        _zoo_box(ax, peer_cx, y_source, bw, h, "source", peer_color, cfg)
        _zoo_box(ax, peer_cx, y_model, bw, hm, "neural", peer_color, cfg)
        _zoo_box(ax, peer_cx, y_target, bw, h, "target", peer_color, cfg)
        _zoo_arrow(ax, peer_cx, src_bot, peer_cx, mid_top, peer_color, cfg)
        _zoo_arrow(ax, peer_cx, mid_bot, peer_cx, tgt_top, peer_color, cfg)

        margin = 0.3
        title_centers = (
            sum(pf_bounds) / 2,
            sum(gain_bounds) / 2,
            sum(structured_bounds) / 2,
            peer_cx,
        )
        title_y = src_top + 0.35 * h
        for cx, title, color in zip(title_centers, _ZOO_CONDENSED_TITLES, model_colors):
            ax.text(
                cx,
                title_y,
                title,
                ha="center",
                va="bottom",
                color=color,
                fontsize=cfg.box_fontsize,
                zorder=3,
            )

        ax.set_xlim(pf_left - margin, peer_cx + bw / 2 + margin)
        ax.set_ylim(tgt_bot - margin, title_y + 0.4 * h)
        return ax


# ======================================================================================
# Ultra-condensed model zoo (overlaid PF family + direct peer prediction)
# ======================================================================================

_ZOO_ULTRA_CONDENSED_LIMITS = {
    "box_width": (0.5, 4.0),
    "box_height": (0.5, 3.0),
    "latent_height_scale": (1.0, 2.5),
    "gain_width": (0.3, 3.0),
    "row_gap": (0.1, 3.0),
    "lane_gap": (0.0, 2.0),
    "group_gap": (0.0, 3.0),
    "structured_bypass_offset": (0.05, 2.0),
    "label_column_width": (0.5, 5.0),
    "label_gap": (0.0, 2.0),
    "label_line_spacing": (0.2, 1.5),
    "box_corner_radius": (0.0, 0.5),
    "arrow_linewidth": (0.5, 6.0),
    "arrow_mutation_scale": (5.0, 40.0),
    "junction_dot_size": (2.0, 24.0),
    "junction_dot_offset": (0.0, 0.5),
    "box_fontsize": (6.0, 28.0),
    "label_fontsize": (6.0, 28.0),
}

_ZOO_ULTRA_CONDENSED_TUNABLES = tuple(_ZOO_ULTRA_CONDENSED_LIMITS)

_ZOO_ULTRA_CONDENSED_BOOLEANS = (
    "horizontal_connections_as_lines",
    "vertical_connections_as_lines",
)

_ZOO_ULTRA_CONDENSED_LABELS = (
    "Placefield",
    "Gain",
    "Residual",
    "Peer Pred.",
)


class ModelZooUltraCondensed(FigureViewer):
    """Two-group schematic overlaying the three PF-family models beside peer prediction.

    The first group has a black source -> position -> PF -> target backbone. A red gain box
    branches from source and modulates PF -> target; an orange residual box above PF carries
    the shared-residual route from source around to target. The peer group is a direct blue
    source -> target path.
    """

    _TUNABLES = _ZOO_ULTRA_CONDENSED_TUNABLES

    def __init__(self, config: ModelZooUltraCondensedConfig | None = None):
        self.cfg = self.add_controls(self, config)

    @classmethod
    def add_controls(
        cls,
        viewer,
        config: ModelZooUltraCondensedConfig | None = None,
    ) -> ModelZooUltraCondensedConfig:
        """Add ultra-condensed geometry and typography controls to ``viewer``."""
        config = config or ModelZooUltraCondensedConfig()
        for name in cls._TUNABLES:
            lo, hi = _ZOO_ULTRA_CONDENSED_LIMITS[name]
            viewer.add_float(name, value=float(getattr(config, name)), min=lo, max=hi)
        for name in _ZOO_ULTRA_CONDENSED_BOOLEANS:
            viewer.add_boolean(name, value=bool(getattr(config, name)))
        return config

    @classmethod
    def config_from_state(
        cls,
        config: ModelZooUltraCondensedConfig,
        state,
    ) -> ModelZooUltraCondensedConfig:
        """Apply live control values without changing non-widget style fields."""
        control_names = (*cls._TUNABLES, *_ZOO_ULTRA_CONDENSED_BOOLEANS)
        return replace(config, **{name: state[name] for name in control_names})

    def plot(self, state):
        cfg = self.config_from_state(self.cfg, state)
        fig, ax = self.new_subplots(figsize=cfg.figsize, layout="constrained")
        fig.patch.set_facecolor(cfg.background_color)
        self.draw(ax, cfg)
        return fig

    @staticmethod
    def draw(ax, cfg: ModelZooUltraCondensedConfig):
        """Draw the ultra-condensed schematic on an existing Matplotlib axis."""
        bw = cfg.box_width
        h = cfg.box_height
        hm = h * cfg.latent_height_scale

        y_target = h / 2
        y_model = y_target + h / 2 + cfg.row_gap + hm / 2
        y_source = y_model + hm / 2 + cfg.row_gap + h / 2
        src_top, src_bot = y_source + h / 2, y_source - h / 2
        mid_top, mid_bot = y_model + hm / 2, y_model - hm / 2
        tgt_top, tgt_bot = y_target + h / 2, y_target - h / 2

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        def connection(x0, y0, x1, y1, color, *, as_line: bool):
            if as_line:
                _zoo_line(ax, [x0, x1], [y0, y1], color, cfg)
            else:
                _zoo_arrow(ax, x0, y0, x1, y1, color, cfg)

        # Four color-coded labels occupy a dedicated column to the left of both diagrams.
        label_x = 0.0
        label_step = cfg.label_line_spacing * h
        label_center_y = (y_source + y_target) / 2
        label_ys = [label_center_y + (1.5 - i) * label_step for i in range(4)]
        label_colors = (
            cfg.placefield_color,
            cfg.gain_color,
            cfg.structured_additive_color,
            cfg.peer_color,
        )
        for y, label, color in zip(label_ys, _ZOO_ULTRA_CONDENSED_LABELS, label_colors):
            ax.text(
                label_x,
                y,
                label,
                ha="left",
                va="center",
                color=color,
                fontsize=cfg.label_fontsize,
                zorder=3,
            )

        # Black placefield backbone: source/position left, PF/target right.
        family_left = cfg.label_column_width + cfg.label_gap
        gain_cx = family_left + cfg.gain_width / 2
        pos_cx = family_left + cfg.gain_width + cfg.lane_gap + bw / 2
        pf_cx = pos_cx + bw + cfg.lane_gap
        _zoo_box(ax, pos_cx, y_source, bw, h, "source", cfg.placefield_color, cfg)
        _zoo_box(ax, pos_cx, y_model, bw, hm, "pos", cfg.placefield_color, cfg)
        _zoo_box(ax, pf_cx, y_model, bw, hm, "PF", cfg.placefield_color, cfg)
        _zoo_box(ax, pf_cx, y_target, bw, h, "target", cfg.placefield_color, cfg)
        connection(
            pos_cx,
            src_bot,
            pos_cx,
            mid_top,
            cfg.placefield_color,
            as_line=cfg.vertical_connections_as_lines,
        )
        connection(
            pos_cx + bw / 2,
            y_model,
            pf_cx - bw / 2,
            y_model,
            cfg.placefield_color,
            as_line=cfg.horizontal_connections_as_lines,
        )
        connection(
            pf_cx,
            mid_bot,
            pf_cx,
            tgt_top,
            cfg.placefield_color,
            as_line=cfg.vertical_connections_as_lines,
        )

        # Red global-gain overlay: leave the source to the left, then turn down into gain
        # with a single bend. Gain comes down and right to the left side of PF -> target.
        _zoo_box(ax, gain_cx, y_model, cfg.gain_width, hm, "gain", cfg.gain_color, cfg)
        _zoo_line(ax, [pos_cx - bw / 2, gain_cx], [y_source, y_source], cfg.gain_color, cfg)
        connection(
            gain_cx,
            y_source,
            gain_cx,
            mid_top,
            cfg.gain_color,
            as_line=cfg.vertical_connections_as_lines,
        )
        y_junction = (mid_bot + tgt_top) / 2
        dot_x = pf_cx - cfg.junction_dot_offset * bw
        _zoo_line(ax, [gain_cx, gain_cx, dot_x], [mid_bot, y_junction, y_junction], cfg.gain_color, cfg)
        ax.plot(dot_x, y_junction, marker="o", markersize=cfg.junction_dot_size, color=cfg.gain_color, zorder=4)

        # Orange shared-residual overlay: source drives an explicit residual box above PF;
        # its output continues around the right side and back into target.
        _zoo_box(
            ax,
            pf_cx,
            y_source,
            bw,
            h,
            "residual",
            cfg.structured_additive_color,
            cfg,
        )
        connection(
            pos_cx + bw / 2,
            y_source,
            pf_cx - bw / 2,
            y_source,
            cfg.structured_additive_color,
            as_line=cfg.horizontal_connections_as_lines,
        )
        family_right = pf_cx + bw / 2
        bypass_x = family_right + cfg.structured_bypass_offset
        _zoo_line(
            ax,
            [family_right, bypass_x, bypass_x],
            [y_source, y_source, y_target],
            cfg.structured_additive_color,
            cfg,
        )
        connection(
            bypass_x,
            y_target,
            pf_cx + bw / 2,
            y_target,
            cfg.structured_additive_color,
            as_line=cfg.horizontal_connections_as_lines,
        )

        # Peer prediction is deliberately direct: no intermediate neural box.
        peer_cx = bypass_x + cfg.group_gap + bw / 2
        _zoo_box(ax, peer_cx, y_source, bw, h, "source", cfg.peer_color, cfg)
        _zoo_box(ax, peer_cx, y_target, bw, h, "target", cfg.peer_color, cfg)
        connection(
            peer_cx,
            src_bot,
            peer_cx,
            tgt_top,
            cfg.peer_color,
            as_line=cfg.vertical_connections_as_lines,
        )

        margin = 0.3
        ax.set_xlim(label_x - margin, peer_cx + bw / 2 + margin)
        ax.set_ylim(tgt_bot - margin, src_top + margin)
        return ax
