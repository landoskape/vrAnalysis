"""Data-selection defaults and widgets shared by figure 3's panels.

Two things live here. First, the starting values the figure wants where they differ from what a
config declares -- every panel draws the deconvolved ``"default"`` activity variant, while the
configs all ship ``"raw"``.

Second, the widget encoding for ``StimSpaceSpectraConfig``'s two tuple-valued param axes
(``smooth_widths``, ``reliability_fraction_active_thresholds``), which a Syd selection can't hold
directly -- the same idea as ``figure2/latents.py``'s ``MODEL_PAIR_LABELS``. Everything else goes
through the shared
:func:`~dimensionality_manuscript.figure_scripts.panels.add_data_selection_widgets`.

``include_iti`` is deliberately not a widget: every panel built on those results reads both values
of it (the Behaving / w-ITIs / w-Spont split, or the per-env scope), so it is passed explicitly at
``sel`` time instead.
"""

from syd import Viewer

from dimensionality_manuscript.pipeline import ResultsAggregator
from dimensionality_manuscript.figure_scripts.panels import add_data_selection_widgets

# Axes each stimspace panel fixes at ``sel`` time rather than offering as a widget.
SKIP_AXES = ("include_iti",)

# Starting values for the scalar-axis configs (``SubspaceConfig``, ``PlaceFieldStructureConfig``).
# Both declare ``activity_parameters_name="raw"``, and their ``smooth_width=None`` is
# indistinguishable from "unset" to the config-value lookup, so both axes are pinned here.
ACTIVITY_SELECTION_DEFAULTS = {"smooth_width": None, "activity_parameters_name": "default"}

# The same idea for ``StimSpaceSpectraConfig``, which sweeps tuple-valued smoothing/threshold axes
# instead of a single ``smooth_width``. A caller can still override any of these.
STIMSPACE_SELECTION_DEFAULTS = {
    "smooth_widths": (5.0, None),
    "activity_parameters_name": "default",
    "reliability_fraction_active_thresholds": (None, None),
}


def tuple_label(value: tuple) -> str:
    """Render a tuple param value (elements are float or None) as a widget-safe string label."""
    return "-".join("None" if v is None else str(v) for v in value)


def add_stimspace_selection_widgets(
    viewer: Viewer, results: ResultsAggregator, defaults: dict | None = None
) -> tuple[tuple[str, ...], dict[str, dict[str, tuple]]]:
    """Add one selection widget per ``StimSpaceSpectraConfig`` param axis except ``include_iti``.

    Scalar axes are handled by :func:`add_data_selection_widgets`; tuple-valued axes (which it
    skips automatically) get a label-encoded selection here.

    Parameters
    ----------
    viewer : Viewer
        Viewer to register the widgets on.
    results : ResultsAggregator
        Aggregator whose ``param_axes`` define the axes and their options.
    defaults : dict or None
        Starting values overriding :data:`STIMSPACE_SELECTION_DEFAULTS`. Tuple axes accept either
        a native tuple or an already-encoded label.

    Returns
    -------
    tuple
        ``(names, tuple_labels)``: the widget names, and ``{axis: {label: tuple}}`` for decoding
        the tuple axes. Pass both to :func:`stimspace_selection` and the names to ``on_change``.
    """
    defaults = {**STIMSPACE_SELECTION_DEFAULTS, **(defaults or {})}
    tuple_axes = [
        axis
        for axis, options in results.param_axes.items()
        if axis not in SKIP_AXES and any(isinstance(option, tuple) for option in options)
    ]

    names = list(add_data_selection_widgets(viewer, results, skip=(*SKIP_AXES, *tuple_axes), defaults=defaults))

    tuple_labels: dict[str, dict[str, tuple]] = {}
    for axis in tuple_axes:
        label_map = {tuple_label(option): option for option in results.param_axes[axis]}
        tuple_labels[axis] = label_map
        value = defaults.get(axis)
        label = tuple_label(value) if isinstance(value, tuple) else value
        # Syd validates the value against the options, so a stale default fails loudly here.
        viewer.add_selection(axis, value=label if label is not None else next(iter(label_map)), options=list(label_map))
        names.append(axis)
    return tuple(names), tuple_labels


def stimspace_selection(state: dict, names, tuple_labels: dict[str, dict[str, tuple]], **extra) -> dict:
    """``results.sel`` kwargs for the stimspace widgets, decoding tuple labels back to tuples."""
    selection = {name: tuple_labels[name][state[name]] if name in tuple_labels else state[name] for name in names}
    selection.update(extra)
    return selection
