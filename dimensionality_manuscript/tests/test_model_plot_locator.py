"""Tests for :mod:`dimensionality_manuscript.model_plot_locator`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dimensionality_manuscript.model_metadata_io import ModelMetadata
from dimensionality_manuscript.model_plot_locator import ModelPlotLocator, select_models

FULL_MAIN_IDX = [0, 1, 2, 3, 6, 7, 8, 15, 16, 17, 18, 19, 20, 30]
FULL_HIGH_D_SPEED_IDX = [9, 10, 11, 12, 13, 14]
FULL_HIGH_D_SPEED_ANCHOR_IDX = [15, 16, 17, 18, 19, 20]
FULL_NO_INTERCEPT_IDX = [21, 22, 23, 24, 25, 26, 27, 28, 29, 31]
FULL_NO_INTERCEPT_ANCHOR_IDX = [6, 7, 8, 15, 16, 17, 18, 19, 20, 30]
FULL_VECTOR_GAIN_IDX = [4, 5]
FULL_VECTOR_GAIN_ANCHOR_IDX = [2, 3]

REDUCED_MAIN_IDX = [0, 1, 2, 3, 6, 15, 18, 30]
REDUCED_HIGH_D_SPEED_IDX = [9, 12]
REDUCED_HIGH_D_SPEED_ANCHOR_IDX = [15, 18]
REDUCED_NO_INTERCEPT_IDX = [21, 24, 27, 31]
REDUCED_NO_INTERCEPT_ANCHOR_IDX = [6, 15, 18, 30]
REDUCED_VECTOR_GAIN_IDX = [4, 5]
REDUCED_VECTOR_GAIN_ANCHOR_IDX = [2, 3]


@pytest.mark.parametrize(
    (
        "preset",
        "main_idx",
        "hds_idx",
        "hds_anchor_idx",
        "ni_idx",
        "ni_anchor_idx",
        "vg_idx",
        "vg_anchor_idx",
    ),
    [
        (
            "full",
            FULL_MAIN_IDX,
            FULL_HIGH_D_SPEED_IDX,
            FULL_HIGH_D_SPEED_ANCHOR_IDX,
            FULL_NO_INTERCEPT_IDX,
            FULL_NO_INTERCEPT_ANCHOR_IDX,
            FULL_VECTOR_GAIN_IDX,
            FULL_VECTOR_GAIN_ANCHOR_IDX,
        ),
        (
            "reduced",
            REDUCED_MAIN_IDX,
            REDUCED_HIGH_D_SPEED_IDX,
            REDUCED_HIGH_D_SPEED_ANCHOR_IDX,
            REDUCED_NO_INTERCEPT_IDX,
            REDUCED_NO_INTERCEPT_ANCHOR_IDX,
            REDUCED_VECTOR_GAIN_IDX,
            REDUCED_VECTOR_GAIN_ANCHOR_IDX,
        ),
    ],
)
def test_preset_golden_indices(
    preset: str,
    main_idx: list[int],
    hds_idx: list[int],
    hds_anchor_idx: list[int],
    ni_idx: list[int],
    ni_anchor_idx: list[int],
    vg_idx: list[int],
    vg_anchor_idx: list[int],
    model_names: tuple[str, ...],
    model_metadata: dict[str, ModelMetadata],
) -> None:
    """Layer indices match dim_regression_figures.ipynb."""
    loc = ModelPlotLocator(preset=preset, names=model_names, metadata=model_metadata)
    d = loc.as_dict()
    assert d["main_idx"] == main_idx
    assert d["high_d_speed_idx"] == hds_idx
    assert d["high_d_speed_anchor_idx"] == hds_anchor_idx
    assert d["no_intercept_idx"] == ni_idx
    assert d["no_intercept_anchor_idx"] == ni_anchor_idx
    assert d["vector_gain_idx"] == vg_idx
    assert d["vector_gain_anchor_idx"] == vg_anchor_idx


@pytest.mark.parametrize("preset", ["full", "reduced"])
def test_x_values_match_notebook_anchors(
    preset: str,
    model_names: tuple[str, ...],
    model_metadata: dict[str, ModelMetadata],
) -> None:
    """Overlay x positions align with notebook xvals_map logic."""
    loc = ModelPlotLocator(preset=preset, names=model_names, metadata=model_metadata)
    d = loc.as_dict()
    x_map = loc.x_positions()
    hds_x = loc.x_values_for_layer("high_d_speed", as_dict=False)
    assert hds_x == [x_map[name] for name in d["high_d_speed_anchor_name"]]
    ni_x = loc.x_values_for_layer("no_intercept", as_dict=False)
    assert ni_x == [x_map[name] for name in d["no_intercept_anchor_name"]]
    vg_x = loc.x_values_for_layer("vector_gain", as_dict=False)
    assert vg_x == [x_map[name] for name in d["vector_gain_anchor_name"]]


def test_select_models_main_excludes_vector_gain(
    model_names: tuple[str, ...],
    model_metadata: dict[str, ModelMetadata],
) -> None:
    """Full main selection excludes vector-gain models."""
    names = select_models(
        include={"main": True},
        exclude={"vector_gain": True},
        names=model_names,
        metadata=model_metadata,
    )
    assert "external_placefield_1d_vector_gain" not in names
    assert "external_placefield_1d" in names


def test_layers_are_disjoint(
    model_names: tuple[str, ...],
    model_metadata: dict[str, ModelMetadata],
) -> None:
    """No model appears in two layers for default presets."""
    for preset in ("full", "reduced"):
        layers = ModelPlotLocator(preset=preset, names=model_names, metadata=model_metadata).layers()
        all_names = [name for layer in layers.values() for name in layer.names]
        assert len(all_names) == len(set(all_names))


def test_overlay_without_reference_raises(
    model_names: tuple[str, ...],
    model_metadata: dict[str, ModelMetadata],
) -> None:
    """Overlays must have a reference in metadata."""
    bad_meta = dict(model_metadata)
    bad_meta["pos_speed_decoder_only"] = SimpleNamespace(reference=None)
    with pytest.raises(ValueError, match="no reference"):
        ModelPlotLocator(preset="full", names=model_names, metadata=bad_meta)


def test_overlay_reference_not_on_main_raises(
    model_names: tuple[str, ...],
    model_metadata: dict[str, ModelMetadata],
) -> None:
    """Overlay reference must be a main-layer model."""
    bad_meta = dict(model_metadata)
    bad_meta["rbfpos_decoder_only_no_intercept"] = ModelMetadata(
        main=False,
        high_d_pos=True,
        no_intercept=True,
        reference="rbfpos",
    )
    with pytest.raises(ValueError, match="not on the main layer"):
        ModelPlotLocator(preset="reduced", names=model_names, metadata=bad_meta)


def test_duplicate_across_layers_raises(
    model_names: tuple[str, ...],
    model_metadata: dict[str, ModelMetadata],
) -> None:
    """Constructing overlapping layers manually is rejected."""
    bad_meta = dict(model_metadata)
    name = "rrr"
    meta = bad_meta[name]
    bad_meta[name] = ModelMetadata(
        main=True,
        unconstrained=meta.unconstrained,
        high_d_speed=True,
    )
    with pytest.raises(ValueError, match="appears in layers"):
        ModelPlotLocator(
            preset="full",
            names=model_names,
            metadata=bad_meta,
            layers=("main", "high_d_speed"),
        )
