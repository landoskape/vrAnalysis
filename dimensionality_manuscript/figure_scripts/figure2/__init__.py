"""Figure 2: what regression models of external, internal, and neural latents explain.

Each panel is a :class:`syd.Viewer` subclass whose ``__init__`` keyword arguments *are* the
widget defaults -- there is no separate caller function. In a notebook::

    viewer = ModelPerformanceViewer(results, activity_parameters_name="std", fontsize=12)
    viewer.show()                    # interactive
    fig = viewer.plot(viewer.state)  # the static figure, for save_figure

Panels that read a :class:`ResultsAggregator` build their data-selection widgets from its
``param_axes``, so any stored variation can also be seeded by name (``spks_type=...``).
Every font size on every panel is a widget; nothing here touches ``plt.rcParams``.
"""

from ._predictions import clear_prediction_cache, get_model_predictions
from .dim_sweep import DIM_SWEEP_MODEL_NAMES, RankModelsSweepViewer, RegressionDimSweepViewer
from .familiarity import RegressionFamiliarityByEnvViewer, RegressionFamiliarityViewer

# Retired with the rrr_to_external_latents analysis. Both viewers in .latents read that config's
# results, so the module is left unimported rather than partially revived.
# from .latents import DimSweepLatentsViewer, RRRExternalLatentsViewer
from .model_style import ARCH_LABELS, ARCH_LINESTYLE, MODEL_STYLE, ROLE_COLOR, ROLE_LABELS
from .performance import PERFORMANCE_MODEL_NAMES, ModelPerformanceViewer
from .placefield_gain import PlacefieldGainViewer
from .pf_residual import (
    PF_RESIDUAL_MODEL_NAMES,
    ModelPlacefieldResidualExplorer,
    ModelPlacefieldResidualFamiliarityViewer,
    ModelPlacefieldResidualInsetViewer,
    ModelPlacefieldResidualViewer,
)
from .prediction import RegressionPredictionViewer
from .rasters import PREDICTION_FIGURE_MODELS, ModelPredictionsFigureViewer, ModelRasterFocus
from .residual_structure import RegressionResidualStructureViewer
from .zoo import (
    ModelZooCondensed,
    ModelZooSchematic,
    ModelZooSchematicConfig,
    ModelZooUltraCondensed,
    ModelZooUltraCondensedConfig,
)

__all__ = [
    "ARCH_LABELS",
    "ARCH_LINESTYLE",
    "DIM_SWEEP_MODEL_NAMES",
    "MODEL_STYLE",
    "ModelPerformanceViewer",
    "PlacefieldGainViewer",
    "ModelPlacefieldResidualExplorer",
    "ModelPlacefieldResidualFamiliarityViewer",
    "ModelPlacefieldResidualInsetViewer",
    "ModelPlacefieldResidualViewer",
    "ModelPredictionsFigureViewer",
    "ModelRasterFocus",
    "PREDICTION_FIGURE_MODELS",
    "ModelZooCondensed",
    "ModelZooSchematic",
    "ModelZooSchematicConfig",
    "ModelZooUltraCondensed",
    "ModelZooUltraCondensedConfig",
    "PERFORMANCE_MODEL_NAMES",
    "PF_RESIDUAL_MODEL_NAMES",
    "ROLE_COLOR",
    "ROLE_LABELS",
    "RegressionDimSweepViewer",
    "RankModelsSweepViewer",
    "RegressionFamiliarityByEnvViewer",
    "RegressionFamiliarityViewer",
    "RegressionPredictionViewer",
    "RegressionResidualStructureViewer",
    "clear_prediction_cache",
    "get_model_predictions",
]
