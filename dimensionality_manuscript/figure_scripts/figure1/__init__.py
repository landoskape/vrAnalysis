"""Figure 1: what the mice did, and how well a place field describes a hippocampal cell.

Each panel is a :class:`syd.Viewer` subclass whose ``__init__`` keyword arguments *are* the
widget defaults -- there is no separate caller function. In a notebook::

    viewer = StackedRasterFocus(pred_results, mouse="ATL027", vmax=5.0)
    viewer.show()                    # interactive
    fig = viewer.plot(viewer.state)  # the static figure, for save_figure

    viewer = PlacefieldActivityFocus(
        pred_results, mouse="ATL027", all_environments=False, env_slot=0,
        roi_sort="environment", sample_sort="chronological", transpose_pf=False,
        vmax_pf=5.0, vmax_activity=5.0, vmax_covariance=2.0,
    )
    viewer.show()

Panels that read a :class:`ResultsAggregator` build their data-selection widgets from its
``param_axes``, so any stored variation can also be seeded by name. Every font size on every
panel is a widget; nothing here touches ``plt.rcParams``.

The one function is :func:`vr_schematic_and_speed`, which composes two of the viewers into a
single aligned figure rather than drawing a panel of its own.
"""

from .behavior_decoding import SpeedEnvironmentDecoding
from .crossval import CrossValidatedPlacefieldFocus, CrossValidatedPlacefields
from .familiarity import R2Familiarity, ReliabilityPredictionSummary
from .holistic_placefield_familiarity import HolisticPlacefieldFamiliarityViewer
from .pf_amplitude import PlacefieldPeakAmplitude
from .placefield import (
    PlaceFieldFocus,
    PlaceFieldPopulationDistributions,
    PlaceFieldPredictionFocus,
    PlaceFieldReliabilityRelationships,
    SpatialAmplitudeReliability,
)
from .r2_placefield import R2PlacefieldFocus, r2_placefield_arrays
from .rasters import StackedRasterFocus
from .placefield_activity import PlacefieldActivityFocus
from .reliability import DataOverviewViewer, ReliabilityHistogramViewer
from .reliability_prediction import ReliabilityPredictionFamiliarity, ReliabilityPredictionFocus
from .schematic import (
    VR_REWARD_ZONE_WIDTH_CM,
    VR_SCHEMATIC_ENVS,
    VREnvironmentSchematic,
    vr_schematic_and_speed,
)
from .speed import MouseSpeedFocus, env_speed_mixedlm, format_speed_stats
from .timeline import ExperimentTimeline
from .tracked_placefield import TrackedPlaceFieldPredictionFocus
from .traversal import TraversalFocus

__all__ = [
    "CrossValidatedPlacefields",
    "CrossValidatedPlacefieldFocus",
    "DataOverviewViewer",
    "ExperimentTimeline",
    "HolisticPlacefieldFamiliarityViewer",
    "MouseSpeedFocus",
    "PlaceFieldFocus",
    "PlaceFieldPopulationDistributions",
    "PlaceFieldPredictionFocus",
    "PlaceFieldReliabilityRelationships",
    "SpatialAmplitudeReliability",
    "PlacefieldPeakAmplitude",
    "R2Familiarity",
    "R2PlacefieldFocus",
    "ReliabilityHistogramViewer",
    "ReliabilityPredictionFamiliarity",
    "ReliabilityPredictionFocus",
    "ReliabilityPredictionSummary",
    "SpeedEnvironmentDecoding",
    "StackedRasterFocus",
    "PlacefieldActivityFocus",
    "TraversalFocus",
    "TrackedPlaceFieldPredictionFocus",
    "VREnvironmentSchematic",
    "VR_REWARD_ZONE_WIDTH_CM",
    "VR_SCHEMATIC_ENVS",
    "env_speed_mixedlm",
    "format_speed_stats",
    "r2_placefield_arrays",
    "vr_schematic_and_speed",
]
