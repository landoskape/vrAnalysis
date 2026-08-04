"""Rounded-box schematics of the external / internal / neural model zoo."""

from dataclasses import dataclass, replace

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from dimensionality_manuscript.figure_scripts.panels import FigureViewer


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

    # --- Fonts -------------------------------------------------------------------------
    box_fontsize: float = 13.0
    container_label_fontsize: float = 15.0
    arrow_label_fontsize: float = 13.0

    # --- Figure ------------------------------------------------------------------------
    figsize: tuple[float, float] = (18.0, 5.0)
    background_color: str = "#ffffff"


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
    "gain_two_model_offset": (0.0, 0.4),
    "junction_dot_offset": (0.0, 0.5),
    "gain_route_hoffset": (0.0, 0.3),
    "gain_route_voffset": (0.0, 0.5),
    "box_fontsize": (6.0, 28.0),
    "container_label_fontsize": (6.0, 30.0),
}

# ModelZooSchematicConfig fields exposed as live Syd controls by the full schematic.
_ZOO_TUNABLES = [
    "box_width",
    "box_height",
    "latent_height_scale",
    "gain_width",
    "col_gap",
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

# One column per model. Color identifies the model throughout its complete stack.
_ZOO_CONDENSED_MODEL_LABELS = ("PF", "PF", "neural")

# The column whose latent is multiplicatively modulated by a gain box drawn to its right.
_ZOO_CONDENSED_GAIN_COLUMN = 1

# ModelZooSchematicConfig fields exposed as live Syd controls. The condensed layout has no
# containers and a single gain route, so the container/panel and paired-route fields drop out.
_ZOO_CONDENSED_TUNABLES = [
    "box_width",
    "box_height",
    "latent_height_scale",
    "gain_width",
    "col_gap",
    "row_gap",
    "box_corner_radius",
    "arrow_linewidth",
    "arrow_mutation_scale",
    "junction_dot_size",
    "junction_dot_offset",
    "box_fontsize",
]


class ModelZooCondensed(ModelZooSchematic):
    """Three-column PF / PF+gain / neural model schematic.

    Each source -> latent -> target stack has one color: PF is black, PF+gain is red, and
    neural is blue. The gain box and its wiring share the second column's red encoding.

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
        cg, rg = cfg.col_gap, cfg.row_gap

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

        model_colors = (cfg.black_box_color, cfg.red_box_color, cfg.blue_box_color)

        # Column centers; the gain column reserves room for its gain box before the next column.
        cxs = []
        x = bw / 2
        for i in range(len(_ZOO_CONDENSED_MODEL_LABELS)):
            cxs.append(x)
            x += bw + cg + (cfg.gain_width + cg if i == _ZOO_CONDENSED_GAIN_COLUMN else 0.0)

        for cx, label, color in zip(cxs, _ZOO_CONDENSED_MODEL_LABELS, model_colors):
            _zoo_box(ax, cx, y_source, bw, h, "source", color, cfg)
            _zoo_box(ax, cx, y_model, bw, hm, label, color, cfg)
            _zoo_box(ax, cx, y_target, bw, h, "target", color, cfg)
            _zoo_arrow(ax, cx, src_bot, cx, mid_top, color, cfg)
            _zoo_arrow(ax, cx, mid_bot, cx, tgt_top, color, cfg)

        # Gain wiring: source -> gain along the source row, then gain down and back left onto
        # its column's latent -> target arrow, where the dot marks the multiplicative junction.
        # junction_dot_offset slides that dot rightward off the arrow so both stay visible.
        gain_col_cx = cxs[_ZOO_CONDENSED_GAIN_COLUMN]
        gain_color = model_colors[_ZOO_CONDENSED_GAIN_COLUMN]
        gain_cx = gain_col_cx + bw / 2 + cg + cfg.gain_width / 2
        y_junction = (mid_bot + tgt_top) / 2
        dot_x = gain_col_cx + cfg.junction_dot_offset * bw
        _zoo_box(ax, gain_cx, y_model, cfg.gain_width, hm, "gain", gain_color, cfg)
        _zoo_line(ax, [gain_col_cx + bw / 2, gain_cx], [y_source, y_source], gain_color, cfg)
        _zoo_arrow(ax, gain_cx, y_source, gain_cx, mid_top, gain_color, cfg)
        _zoo_line(ax, [gain_cx, gain_cx, dot_x], [mid_bot, y_junction, y_junction], gain_color, cfg)
        ax.plot(
            dot_x,
            y_junction,
            marker="o",
            markersize=cfg.junction_dot_size,
            color=gain_color,
            zorder=4,
        )

        margin = 0.3
        gain_group_left = gain_col_cx - bw / 2
        gain_group_right = gain_cx + cfg.gain_width / 2
        title_centers = (cxs[0], (gain_group_left + gain_group_right) / 2, cxs[-1])
        title_y = src_top + 0.35 * h
        for cx, title, color in zip(title_centers, ("PF", "PF+Gain", "Peer"), model_colors):
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

        ax.set_xlim(cxs[0] - bw / 2 - margin, cxs[-1] + bw / 2 + margin)
        ax.set_ylim(tgt_bot - margin, title_y + 0.4 * h)
        return ax
