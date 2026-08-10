"""Shared visual identity for the figure-2 regression models.

Color encodes where a model's latent lives and line style encodes its architecture, matching
the model-zoo schematic: matching external/internal models share a dash pattern and differ only
in black/red, while RRR keeps the schematic's blue. Every figure-2 panel that draws models
reads its colors from here, so the schematic, the dimensionality sweep, and the score panels
stay in agreement.
"""

import numpy as np

from dimensionality_manuscript.registry import ModelName

ROLE_COLOR = {
    "external": "#000000",
    "internal": "#c00000",
    "neural": "#0000cd",
}

ROLE_LABELS = {
    "external": "External Latent",
    "internal": "Internal Latent",
    "neural": "Neural Latent",
}

ARCH_LINESTYLE = {
    "placefield": "-",
    "highd_pos": (0, (6, 2)),
    "highd_pos_speed": (0, (1, 1.7)),
    "highd_pos_speed_reward": "-.",
    "placefield_gain": (0, (3, 1, 1, 1, 1, 1)),
    "placefield_structured_gain": (0, (5, 1, 1, 1)),
}

# Legend text for each architecture, in the order the entries are listed.
ARCH_LABELS = {
    "placefield": "placefield",
    "highd_pos": "rbf-pos",
    "highd_pos_speed": "+speed",
    "highd_pos_speed_reward": "+reward",
    "placefield_gain": "+gain",
    "placefield_structured_gain": "+struct. gain",
}

MODEL_STYLE: dict[ModelName, dict] = {
    "external_placefield_1d": dict(role="external", arch="placefield"),
    "rbfpos_decoder_only": dict(role="external", arch="highd_pos"),
    "pos_speed_decoder_only_1dspeed": dict(role="external", arch="highd_pos_speed"),
    "fullregressor_decoder_only_1dspeed": dict(role="external", arch="highd_pos_speed_reward"),
    "internal_placefield_1d": dict(role="internal", arch="placefield"),
    "rbfpos": dict(role="internal", arch="highd_pos"),
    "pos_speed_1dspeed": dict(role="internal", arch="highd_pos_speed"),
    "fullregressor_1dspeed": dict(role="internal", arch="highd_pos_speed_reward"),
    "external_placefield_1d_gain": dict(role="external", arch="placefield_gain"),
    "internal_placefield_1d_gain": dict(role="internal", arch="placefield_gain"),
    "external_placefield_1d_structured_gain": dict(role="external", arch="placefield_structured_gain"),
    "internal_placefield_1d_structured_gain": dict(role="internal", arch="placefield_structured_gain"),
    "rrr": dict(role="neural", arch="placefield"),
    "rrr_no_intercept": dict(role="neural", arch="placefield"),
    "fullregressor_decoder_only_1dspeed_predreward": dict(role="external", arch="highd_pos_speed_reward"),
    "fullregressor_1dspeed_predreward": dict(role="internal", arch="highd_pos_speed_reward"),
    "fullregressor_decoder_only_1dspeed_predreward_no_intercept": dict(role="external", arch="highd_pos_speed_reward"),
    "fullregressor_1dspeed_predreward_no_intercept": dict(role="internal", arch="highd_pos_speed_reward"),
}


def roles_for(model_names) -> np.ndarray:
    """Role ("external" / "internal" / "neural") of each model, as a string array."""
    return np.array([MODEL_STYLE[name]["role"] for name in model_names])
