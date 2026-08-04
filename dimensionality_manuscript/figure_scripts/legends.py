"""Shared Syd widgets for styling a panel's matplotlib legend.

A figure that wants its legends tunable from the viewer registers one widget per knob with
:func:`add_legend_widgets`, seeds any non-default starting values with
:func:`update_legend_widgets`, and draws with :func:`apply_legend`. Every widget name is
namespaced by a ``prefix``, so a multi-panel figure can carry independent controls for each
panel's legend.
"""

from syd import Viewer

# Matplotlib legend placements, plus "auto" (the panel's own default placement, which may be no legend
# at all) and "none" (never draw one). See add_legend_widgets / apply_legend.
LEGEND_LOCS = (
    "auto",
    "none",
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "center left",
    "center right",
    "upper center",
    "lower center",
    "center",
)
# Legend knob -> (widget kind, default, min, max, step). ``fontsize_scale`` multiplies the viewer's
# base fontsize; the rest are the matplotlib :meth:`~matplotlib.axes.Axes.legend` kwargs of the same
# name, at their matplotlib defaults (except ``frameon``, off throughout these figures).
LEGEND_KNOBS = {
    "ncols": ("integer", 1, 1, 6, None),
    "fontsize_scale": ("float", 1.0, 0.2, 2.0, 0.05),
    "handlelength": ("float", 2.0, 0.0, 6.0, 0.1),
    "handletextpad": ("float", 0.8, 0.0, 4.0, 0.1),
    "labelspacing": ("float", 0.5, 0.0, 3.0, 0.05),
    "borderpad": ("float", 0.4, 0.0, 3.0, 0.05),
    "borderaxespad": ("float", 0.5, 0.0, 3.0, 0.05),
    "markerfirst": ("boolean", True, None, None, None),
    "frameon": ("boolean", False, None, None, None),
    "visible": ("boolean", True, None, None, None),
}


def add_legend_widgets(viewer: Viewer, prefix: str = "legend") -> None:
    """Add one widget per legend knob (``{prefix}_loc`` plus :data:`LEGEND_KNOBS`) to ``viewer``.

    ``prefix`` namespaces the widgets so several panels can carry independent legend controls.
    """
    viewer.add_selection(f"{prefix}_loc", options=list(LEGEND_LOCS), value="auto")
    for knob, (kind, default, low, high, step) in LEGEND_KNOBS.items():
        name = f"{prefix}_{knob}"
        if kind == "integer":
            viewer.add_integer(name, value=default, min=low, max=high)
        elif kind == "float":
            viewer.add_float(name, value=default, min=low, max=high, step=step)
        else:
            viewer.add_boolean(name, value=default)


def update_legend_widgets(viewer: Viewer, options: dict, prefix: str = "legend") -> None:
    """Seed the ``{prefix}_*`` legend widgets from a ``{knob: value}`` mapping.

    Keys are ``"loc"`` or any :data:`LEGEND_KNOBS` name; unknown keys raise.
    """
    casts = {"integer": int, "float": float, "boolean": bool}
    for knob, value in options.items():
        if knob != "loc" and knob not in LEGEND_KNOBS:
            raise ValueError(f"Unknown legend option {knob!r}. Options: {['loc'] + list(LEGEND_KNOBS)}")
        cast = casts.get(LEGEND_KNOBS[knob][0]) if knob != "loc" else None
        # set_parameter_value is type-agnostic, so one call covers every widget kind.
        viewer.set_parameter_value(f"{prefix}_{knob}", cast(value) if cast else value)


def apply_legend(ax, state: dict, fontsize: float, prefix: str = "legend", handles=None, labels=None, auto_loc: str | None = None) -> None:
    """Draw ``ax``'s legend under the ``{prefix}_*`` widget settings.

    ``loc="auto"`` falls back to ``auto_loc`` -- the panel's own default placement, or None for panels
    that show no legend unless asked; ``loc="none"`` never draws one. ``handles``/``labels`` are passed
    through when given, for panels whose artists carry no labels (e.g. a beeswarm).

    Calling this on a panel that already drew its own legend restyles it: matplotlib replaces the
    previous legend, and the "don't draw one" cases clear it.
    """
    if state["legend_visible"] is False:
        existing = ax.get_legend()
        if existing is not None:
            existing.remove()
        return

    loc = state[f"{prefix}_loc"]
    if loc == "auto":
        loc = auto_loc
    if loc is None or loc == "none":
        existing = ax.get_legend()
        if existing is not None:
            existing.remove()
        return
    kwargs = {knob: state[f"{prefix}_{knob}"] for knob in LEGEND_KNOBS if knob != "fontsize_scale"}
    kwargs["loc"] = loc
    kwargs["fontsize"] = fontsize * state[f"{prefix}_fontsize_scale"]
    if "visible" in kwargs:
        kwargs.pop("visible")  # matplotlib doesn't accept a visible kwarg, but we use it to hide the legend
    if handles is not None:
        ax.legend(handles, labels, **kwargs)
    else:
        ax.legend(**kwargs)
