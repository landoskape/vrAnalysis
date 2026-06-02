"""Pytest fixtures that avoid importing the full ``dimensionality_manuscript`` package."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# Register package stub so submodule imports skip configs/__init__.py (vrAnalysis deps).
_PKG_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PKG_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if "dimensionality_manuscript" not in sys.modules:
    _pkg = types.ModuleType("dimensionality_manuscript")
    _pkg.__path__ = [str(_PKG_ROOT)]
    sys.modules["dimensionality_manuscript"] = _pkg

from dimensionality_manuscript.model_metadata_io import load_model_metadata

# Canonical order from registry.MODEL_NAMES (kept here to avoid importing registry).
MODEL_NAMES: tuple[str, ...] = (
    "external_placefield_1d",
    "internal_placefield_1d",
    "external_placefield_1d_gain",
    "internal_placefield_1d_gain",
    "external_placefield_1d_vector_gain",
    "internal_placefield_1d_vector_gain",
    "rbfpos_decoder_only",
    "rbfpos",
    "rbfpos_leak",
    "pos_speed_decoder_only",
    "pos_speed",
    "pos_speed_leak",
    "fullregressor_decoder_only",
    "fullregressor",
    "fullregressor_leak",
    "pos_speed_decoder_only_1dspeed",
    "pos_speed_1dspeed",
    "pos_speed_leak_1dspeed",
    "fullregressor_decoder_only_1dspeed",
    "fullregressor_1dspeed",
    "fullregressor_leak_1dspeed",
    "rbfpos_decoder_only_no_intercept",
    "rbfpos_no_intercept",
    "rbfpos_leak_no_intercept",
    "pos_speed_decoder_only_no_intercept",
    "pos_speed_no_intercept",
    "pos_speed_leak_no_intercept",
    "fullregressor_decoder_only_no_intercept",
    "fullregressor_no_intercept",
    "fullregressor_leak_no_intercept",
    "rrr",
    "rrr_no_intercept",
)


@pytest.fixture
def model_names() -> tuple[str, ...]:
    """Model names in registry order."""
    return MODEL_NAMES


@pytest.fixture
def model_metadata(model_names: tuple[str, ...]):
    """Metadata loaded from ``model_metadata.csv``."""
    return load_model_metadata(model_names)
