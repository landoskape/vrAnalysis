from __future__ import annotations

from types import SimpleNamespace

import torch

from dimilibi import Population
from dimensionality_manuscript.configs.regression import (
    KEY_FIGURE_MODELS,
    VALID_ACTIVITY_PARAMETERS,
    RegressionConfig,
    RegressionDimensionalitySweepConfig,
    RegressionPlacefieldResidualConfig,
)
from dimensionality_manuscript.configs.rrr_to_external_latents import VALID_MODEL_EXT_INT_PAIRS
from dimensionality_manuscript.registry import get_activity_parameters, get_model


EXPECTED_KEY_FIGURE_MODELS = [
    "external_placefield_1d",
    "internal_placefield_1d",
    "external_placefield_1d_gain",
    "internal_placefield_1d_gain",
    "rrr",
    "rrr_no_intercept",
    "fullregressor_decoder_only_1dspeed_predreward",
    "fullregressor_1dspeed_predreward",
    "fullregressor_decoder_only_1dspeed_predreward_no_intercept",
    "fullregressor_1dspeed_predreward_no_intercept",
]


def test_std_activity_parameters_scale_each_roi_by_session_std():
    data = torch.tensor(
        [
            [1.0, 2.0, 4.0, 8.0],
            [3.0, 3.0, 3.0, 3.0],
        ]
    )
    population = Population(data, generate_splits=False)
    parameters = get_activity_parameters("std")
    assert parameters.center is False
    assert parameters.scale is True
    assert parameters.presplit is True

    scaled = population.apply_split(
        data,
        center=parameters.center,
        scale=parameters.scale,
        pre_split=parameters.presplit,
        scale_type=parameters.scale_type,
    )

    expected_scale = data.std(dim=1, keepdim=True)
    expected_scale[expected_scale == 0] = 1
    assert torch.equal(scaled, data / expected_scale)


def test_std_activity_parameters_have_distinct_joblib_cache_key():
    registry = SimpleNamespace(registry_params={"test": "registry"})
    session = SimpleNamespace(
        session_name=("mouse", "date", "session"),
        params=SimpleNamespace(spks_type="sigrebase"),
    )
    default_model = get_model("external_placefield_1d", registry, activity_parameters="default")
    std_model = get_model("external_placefield_1d", registry, activity_parameters="std")

    default_key = default_model._get_hyperparameter_cache_key(session, method="optuna")
    std_key = std_model._get_hyperparameter_cache_key(session, method="optuna")

    assert default_key != std_key


def test_regression_grids_use_only_key_figure_models_and_include_std():
    assert KEY_FIGURE_MODELS == EXPECTED_KEY_FIGURE_MODELS
    assert "std" in VALID_ACTIVITY_PARAMETERS

    regression_grid = RegressionConfig._param_grid()
    residual_grid = RegressionPlacefieldResidualConfig._param_grid()
    dimensionality_grid = RegressionDimensionalitySweepConfig._param_grid()
    assert regression_grid["model_name"] == EXPECTED_KEY_FIGURE_MODELS
    assert residual_grid["model_name"] == EXPECTED_KEY_FIGURE_MODELS
    assert regression_grid["activity_parameters_name"] == ["default", "preserved", "std"]
    assert dimensionality_grid["model_name"] == EXPECTED_KEY_FIGURE_MODELS


def test_rrr_external_latent_pairs_are_derived_from_key_figure_models():
    assert VALID_MODEL_EXT_INT_PAIRS == [
        (
            "fullregressor_decoder_only_1dspeed_predreward",
            "fullregressor_1dspeed_predreward",
        ),
        (
            "fullregressor_decoder_only_1dspeed_predreward_no_intercept",
            "fullregressor_1dspeed_predreward_no_intercept",
        ),
    ]
    assert all(model_name in KEY_FIGURE_MODELS for pair in VALID_MODEL_EXT_INT_PAIRS for model_name in pair)
