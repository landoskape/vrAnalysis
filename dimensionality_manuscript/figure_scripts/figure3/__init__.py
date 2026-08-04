"""Figure 3: how much of CA1's activity subspace the placefield code spans, and how that changes.

Each panel is a :class:`syd.Viewer` subclass whose ``__init__`` keyword arguments *are* the
widget defaults -- there is no separate caller function. In a notebook::

    viewer = SubspaceCurvesRatiosViewer(results, figsize=(5.5, 3.0))
    viewer.show()                    # interactive
    fig = viewer.plot(viewer.state)  # the static figure, for save_figure

Data-selection widgets are built from each aggregator's ``param_axes``, so any stored variation
can be seeded by name (``activity_parameters_name=...``, ``smooth_widths=(5.0, None)``, ...).
Every font size on every panel is a widget; nothing here touches ``plt.rcParams``.

The panels read three different aggregators: ``SubspaceConfig`` (the cross-spectrum panels),
``StimSpaceSpectraConfig`` (the ratios, familiarity and composite panels) and
``PlaceFieldStructureConfig`` (the per-cell feature panel).
"""

from ._curves import DISTRIBUTION_METRICS, SMOOTH_KINDS, gini, smooth_fraction, weighted_fraction
from ._familiarity import CONDITION_COLORS, ENV_FULL_SCOPES, FAMILIARITY_STYLES, familiarity_curves
from ._ratios import ratios_arrays
from ._selection import ACTIVITY_SELECTION_DEFAULTS, STIMSPACE_SELECTION_DEFAULTS, tuple_label
from ._slopes import ENV_SLOPE_STYLES, env_slope_stats, env_slope_table
from .complete_spectrum import CompleteSpectrumViewer
from .crossspace import CROSS_METRIC_COLORS, SubspaceCrossspaceViewer
from .crossspace_example import SubspaceCrossspaceExampleViewer
from .crossspace_per_mouse import SubspaceCrossPerMouseViewer
from .familiarity import FAMILIARITY_MODES, SubspaceFamiliarityViewer
from .pf_structure import PF_FEATURE_KEYS, PlaceFieldStructureOverTimeViewer, pf_env_curves
from .ratios import SubspaceCurvesRatiosViewer

__all__ = [
    "ACTIVITY_SELECTION_DEFAULTS",
    "CONDITION_COLORS",
    "CROSS_METRIC_COLORS",
    "CompleteSpectrumViewer",
    "DISTRIBUTION_METRICS",
    "ENV_FULL_SCOPES",
    "ENV_SLOPE_STYLES",
    "FAMILIARITY_MODES",
    "FAMILIARITY_STYLES",
    "PF_FEATURE_KEYS",
    "PlaceFieldStructureOverTimeViewer",
    "SMOOTH_KINDS",
    "STIMSPACE_SELECTION_DEFAULTS",
    "SubspaceCrossPerMouseViewer",
    "SubspaceCrossspaceExampleViewer",
    "SubspaceCrossspaceViewer",
    "SubspaceCurvesRatiosViewer",
    "SubspaceFamiliarityViewer",
    "env_slope_stats",
    "env_slope_table",
    "familiarity_curves",
    "gini",
    "pf_env_curves",
    "ratios_arrays",
    "smooth_fraction",
    "tuple_label",
    "weighted_fraction",
]
