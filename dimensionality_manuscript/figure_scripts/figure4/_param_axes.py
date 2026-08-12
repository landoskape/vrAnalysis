"""Shared spectrum-key constants and the merged/tuple-encoded param-axis widget bundle.

Several figure4 panels read more than one :class:`~dimensionality_manuscript.pipeline.ResultsAggregator`
at once (StimSpaceSpectra, CVPCAConfig, TilburyFitConfig, ...) and need one widget per param-axis name
shared across all of them, with tuple-valued axes (``smooth_widths``, ...) encoded as string labels
since a Syd selection widget takes only scalars. :func:`add_merged_param_axis_widgets` /
:func:`sel_params` are that pair -- the panel-specific analogue of
:func:`~dimensionality_manuscript.figure_scripts.panels.add_data_selection_widgets` /
:func:`~dimensionality_manuscript.figure_scripts.panels.data_selection`, which does not support
tuple-valued axes (see its docstring: a panel with those supplies its own encoding).

This module also holds spectrum-key constants (which aggregator a key comes from, its fixed plot
color) shared by every spectrum-diagnostics panel.
"""

from syd import Viewer

from ._spectrum_math import _tuple_label

# Selectable spectrum keys and which aggregator each one comes from. StimSpace keys resolve
# against the StimSpaceSpectra aggregator; CVPCA keys against the CVPCAConfig aggregator.
STIMSPACE_KEYS = ["ss_cv", "ss_direct", "ss_cvpca", "sf_cv", "sf_direct"]
CVPCA_KEYS = ["reg_covariances_fixed"]
# The full (functional) spectrum key, also from the StimSpaceSpectra aggregator.
FF_KEY = "ff"

# Tilbury-fit eigenvalue spectra selectable as extra ax[0] overlays in SpectrumFigureViewer, with
# fixed colors. These come from the TilburyFitConfig aggregator, which fits only reliable/active
# neurons at a single fixed reliability/fraction-active threshold (no selection axis).
FIT_KEYS = [
    "eig_tilbury",
    "eig_control",
    "eig_shrinkage",
    "eig_better",
]
FIT_KEY_COLORS = {
    "eig_tilbury": "blue",
    "eig_control": "green",
    "eig_shrinkage": "purple",
    "eig_better": "red",
}
FIT_KEY_LABELS = {
    "eig_tilbury": "Generalized",
    "eig_control": "Gaussian",
    "eig_shrinkage": "Generalized (shrinkage)",
    "eig_better": "Best (Single)",
}
# Which per-neuron fit ``params*`` arrays gate each fit_key's neuron count (matching the
# ``ok_*`` masks TilburyFitConfig.process used to build that key's underlying curve matrix):
# eig_tilbury/eig_control/eig_better all come from ``mat_tilbury``/``mat_control``/``mat_better``,
# which TilburyFitConfig built from ``ok_both`` (params & params_control both finite);
# eig_shrinkage comes from ``ok_s`` (params_shrinkage finite).
FIT_KEY_PARAM_KEYS = {
    "eig_tilbury": ["params", "params_control"],
    "eig_control": ["params", "params_control"],
    "eig_shrinkage": ["params_shrinkage"],
    "eig_better": ["params", "params_control"],
}
TILBURY_REL_FA = (0.3, 0.1)

# Population alpha-comparison panel (ax[-1] of PlacefieldPopulationViewer): the selected source_key
# spectrum plus the four Tilbury-fit eigenvalue spectra, each a per-mouse power-law-exponent
# beeswarm, in plotting order (shrinkage sits between the composite and the unregularized
# generalized fit). "eig_gaussian" in the request is the plain-Gaussian control key ``eig_control``.
POP_EIG_KEYS = ["eig_better", "eig_shrinkage", "eig_tilbury", "eig_control"]
POP_ALPHA_COLORS = {
    "source_key": "darkorange",
    "eig_better": "red",
    "eig_tilbury": "blue",
    "eig_shrinkage": "purple",
    "eig_control": "black",
}
POP_ALPHA_LABELS = {
    "eig_better": "Better",
    "eig_tilbury": "Generalized",
    "eig_shrinkage": "Shrinkage",
    "eig_control": "Gaussian",
}
# key -> "stimspace" | "cvpca"
SOURCE_OF_KEY = {
    **{k: "stimspace" for k in STIMSPACE_KEYS},
    **{k: "cvpca" for k in CVPCA_KEYS},
    FF_KEY: "stimspace",
    "ffres": "stimspace",
    "ffres_white": "stimspace",
}

# Fixed color per curve option, used for the per-mouse alpha scatter and the local-exponent curves
# so a given curve reads the same across panels.
KEY_COLORS = {
    "ss_cv": "black",
    "ss_direct": "blue",
    "ss_cvpca": "red",
    "sf_cv": "orange",
    "sf_direct": "cyan",
    "reg_covariances_fixed": "green",
    "ff": "purple",
}

# "Reliable CA1" (full/functional) curve keys backed by the subspace aggregator.
SVCA_FULL_KEYS = {
    "SVCA": "variance_activity",
    "SVCA_RES": "variance_activity_residual",
}
# Per-environment analogues of SVCA_FULL_KEYS, used by DimensionalityFamiliarityViewer and
# SpectrumCurvesByFamiliarityViewer's "avg_env"/"by_env" plot modes.
SVCA_FULL_ENV_KEYS = {
    "SVCA": "variance_activity_env",
    "SVCA_RES": "variance_activity_residual_env",
}
ASE_SVCA_KEY = "svca"

# Preferred default values for shared param widgets, keyed by raw param-axis name. A widget is
# seeded with this value when the axis exists (in any source) and the value is among its options.
PREFERRED_DEFAULTS = {
    "activity_parameters_name": "default",
    "include_iti": False,
    "spks_type": "sigrebase",
    "center": True,
    "use_fast_sampling": True,
}


def _short_mouse_name(name: str) -> str:
    """Shorten ``CR_Hippocannula*`` mouse names to ``CR*``; other names unchanged."""
    prefix = "CR_Hippocannula"
    if name.startswith(prefix):
        return "CR" + name[len(prefix) :]
    return name


def add_merged_param_axis_widgets(
    viewer: Viewer,
    *aggregators,
    extra_axes: dict[str, list] | None = None,
    preferred_defaults: dict | None = None,
) -> dict[str, dict[str, tuple]]:
    """Add one selection widget per param axis across ``aggregators``, encoding tuple-valued options.

    Axes with the same name across several aggregators (e.g. ``activity_parameters_name`` shared by
    a StimSpaceSpectra and a CVPCAConfig aggregator) get one widget whose options are the union, in
    first-seen order. ``extra_axes`` adds axis names that belong only to a source without its own
    widgets here (e.g. TilburyFitConfig axes not already covered by ``aggregators``) -- names already
    covered are left alone.

    Tuple-valued options (e.g. ``smooth_widths``) are shown as string labels since Syd selections
    take only scalars; the returned map decodes a widget value back to its tuple (see
    :func:`sel_params`). A widget is seeded from ``preferred_defaults`` (falling back to
    :data:`PREFERRED_DEFAULTS`) when the axis has a preferred value among its options.

    Parameters
    ----------
    viewer : Viewer
    *aggregators : ResultsAggregator or None
        None entries are skipped, so an optional aggregator can be passed through unconditionally.
    extra_axes : dict or None
        Additional ``{axis: options}`` entries for axes not already covered by ``aggregators``.
    preferred_defaults : dict or None
        ``{axis: value}`` preferred starting values; defaults to :data:`PREFERRED_DEFAULTS`.

    Returns
    -------
    dict
        ``{axis: {label: tuple}}`` for every tuple-valued axis; scalar axes are absent.
    """
    preferred_defaults = PREFERRED_DEFAULTS if preferred_defaults is None else preferred_defaults
    merged: dict[str, list] = {}
    for agg in aggregators:
        if agg is None:
            continue
        for name, options in agg.param_axes.items():
            existing = merged.setdefault(name, [])
            existing.extend(opt for opt in options if opt not in existing)
    if extra_axes:
        for name, options in extra_axes.items():
            merged.setdefault(name, list(options))

    tuple_labels: dict[str, dict[str, tuple]] = {}
    for name, options in merged.items():
        if any(isinstance(opt, tuple) for opt in options):
            label_map = {_tuple_label(opt): opt for opt in options}
            tuple_labels[name] = label_map
            widget_options = list(label_map)
        else:
            widget_options = options
        viewer.add_selection(name, options=widget_options)
        if name in preferred_defaults:
            default = preferred_defaults[name]
            default = _tuple_label(default) if name in tuple_labels and isinstance(default, tuple) else default
            if default in widget_options:
                viewer.update_selection(name, value=default)
    return tuple_labels


def encode_param(tuple_labels: dict[str, dict[str, tuple]], name: str, value):
    """Map a raw param value to its widget value (tuple -> string label; else unchanged)."""
    if name in tuple_labels and isinstance(value, tuple):
        return _tuple_label(value)
    return value


def sel_params(state: dict, tuple_labels: dict[str, dict[str, tuple]], axis_names) -> dict:
    """``results.sel``-ready kwargs for ``axis_names``, decoding tuple labels back to tuples.

    ``axis_names`` is typically ``aggregator.param_axes`` (or a fixed list, for an aggregator whose
    axes are pinned independently of the shared widgets, e.g. the Tilbury-fit aggregator). Names not
    present in ``state`` are skipped, so the same call works whether or not every axis got a widget.
    """
    params = {}
    for name in axis_names:
        if name not in state:
            continue
        value = state[name]
        if name in tuple_labels:
            value = tuple_labels[name][value]
        params[name] = value
    return params


def merged_axis_names(*aggregators, extra_axes: dict | None = None) -> tuple[str, ...]:
    """Every axis name :func:`add_merged_param_axis_widgets` would register a widget for.

    That function returns only the tuple-valued axes (needed to decode a widget value back to
    its tuple); ``on_change`` wiring needs every axis name, scalar or tuple, that actually got a
    widget. Mirrors :func:`add_merged_param_axis_widgets`'s own axis collection exactly (union of
    every aggregator's ``param_axes``, first-seen order, plus ``extra_axes``).
    """
    names: list[str] = []
    for agg in aggregators:
        if agg is None:
            continue
        for name in agg.param_axes:
            if name not in names:
                names.append(name)
    if extra_axes:
        for name in extra_axes:
            if name not in names:
                names.append(name)
    return tuple(names)


# --- Per-environment "full" (functional) spectrum options, shared by DimensionalityFamiliarityViewer
# and SpectrumCurvesByFamiliarityViewer/SpectrumDimFamiliarityViewer ------------------------------
PER_ENV_PF_KEYS: list[str] = ["ss_cv", "ss_direct", "ss_cvpca", "sf_cv", "sf_direct"]
PER_ENV_PF_RESULT_KEYS: dict[str, str] = {
    "ss_cv": "ss_cv_env",
    "ss_direct": "ss_direct_env",
    "ss_cvpca": "ss_cvpca_env",
}
# Public per-environment residual-Full choices and their stored keys, keyed the same way as
# ``SpectrumDimFamiliarityViewer``'s ``full_key`` options.
PER_ENV_RESIDUAL_RESULT_KEYS: dict[str, str] = {
    "svd_res_full1": "ffres_env_full1",
    "svd_res_fullall": "ffres_env_full1_fullall",
}
PER_ENV_SESSION_ALIGNMENTS: list[str] = ["within_env", "overall"]


def full_key_options(results, results_subspace=None) -> list[str]:
    """Full-spectrum estimators discoverable in either overall or per-environment results."""
    options = ["SVD"]
    if "ffres" in results.arrays or any(key in results.arrays for key in PER_ENV_RESIDUAL_RESULT_KEYS.values()):
        options.append("SVD_RES")
    if results_subspace is not None:
        options.append("SVCA")
        if "variance_activity_residual" in results_subspace.arrays:
            options.append("SVCA_RES")
    return options


def per_env_full_options(results, results_subspace=None) -> list[str]:
    """Return Full choices whose stored per-environment spectra are discoverable."""
    options = ["svd_full1", "svd_fullall"]
    options.extend(public_key for public_key, stored_key in PER_ENV_RESIDUAL_RESULT_KEYS.items() if stored_key in results.arrays)
    if results_subspace is not None:
        options.append("svca")
    return options
