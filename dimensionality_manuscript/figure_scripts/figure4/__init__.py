"""Figure 4: shared-variance/placefield spectra, power-law decay, and dimensionality over familiarity.

Each panel is a :class:`syd.Viewer` subclass (via :class:`~..panels.FigureViewer`) whose
``__init__`` keyword arguments *are* the widget defaults -- there is no separate caller function.
In a notebook::

    viewer = SpectrumFigureViewer(results, results_cvpca=results_cvpca, fontsize=9)
    viewer.show()                    # interactive
    fig = viewer.plot(viewer.state)  # the static figure, for save_figure

Panels that read a :class:`~dimensionality_manuscript.pipeline.ResultsAggregator` build their
data-selection widgets from its ``param_axes`` (see ``_param_axes.add_merged_param_axis_widgets``),
so any stored variation can also be seeded by name. Every font size on every panel is a widget;
nothing here touches ``plt.rcParams``.
"""

from ._alpha_config import (
    ADAPTIVE_ALPHA_CONFIG_NAMES,
    ADAPTIVE_ALPHA_CONFIG_REGISTRY,
    AdaptiveAlphaConfig,
    SpectrumSmoothing,
    SpectrumSmoothingConfig,
    get_adaptive_alpha_config,
)
from .familiarity import DimensionalityFamiliarityViewer
from .placefield_fits import PlacefieldExampleFitViewer, PlacefieldFitFigureViewer
from .placefield_population import PlacefieldPopulationViewer
from .placefield_sessions_mse import PlacefieldParameterSessionsViewer, PlacefieldSpectrumMSEViewer
from .spectra_diagnostics import (
    AdaptiveSpectraEstimationViewer,
    PlacefieldSpectraViewer,
    SessionSpectraViewer,
)
from .spectrum_by_familiarity import SpectrumCurvesByFamiliarityViewer, SpectrumDimFamiliarityViewer
from .spectrum_figure import SpectrumAlphaFigureViewer, SpectrumFigureViewer

__all__ = [
    "ADAPTIVE_ALPHA_CONFIG_NAMES",
    "ADAPTIVE_ALPHA_CONFIG_REGISTRY",
    "AdaptiveAlphaConfig",
    "AdaptiveSpectraEstimationViewer",
    "DimensionalityFamiliarityViewer",
    "PlacefieldExampleFitViewer",
    "PlacefieldFitFigureViewer",
    "PlacefieldParameterSessionsViewer",
    "PlacefieldPopulationViewer",
    "PlacefieldSpectraViewer",
    "PlacefieldSpectrumMSEViewer",
    "SessionSpectraViewer",
    "SpectrumAlphaFigureViewer",
    "SpectrumCurvesByFamiliarityViewer",
    "SpectrumDimFamiliarityViewer",
    "SpectrumFigureViewer",
    "SpectrumSmoothing",
    "SpectrumSmoothingConfig",
    "get_adaptive_alpha_config",
]
