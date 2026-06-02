"""Build plot layer index/name lists from model metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .model_metadata_io import ModelMetadata

PlotPreset = Literal["full", "reduced"]
LayerKey = Literal["main", "high_d_speed", "no_intercept", "vector_gain"]
DEFAULT_LAYERS: tuple[LayerKey, ...] = ("main", "high_d_speed", "no_intercept", "vector_gain")


def select_models(
    *,
    include: dict[str, bool] | None = None,
    exclude: dict[str, bool] | None = None,
    names: tuple[str, ...],
    metadata: dict[str, ModelMetadata],
) -> list[str]:
    """Return model names matching metadata flags in registry order.

    Parameters
    ----------
    include
        Field names on :class:`ModelMetadata` that must be ``True``.
    exclude
        Field names on :class:`ModelMetadata` that must be ``False``.
    names
        Model sequence to scan (defaults to :data:`MODEL_NAMES`).
    metadata
        Metadata lookup (defaults to :data:`MODEL_METADATA`).

    Returns
    -------
    list[str]
        Matching model names, preserving ``names`` order.
    """
    include = include or {}
    exclude = exclude or {}
    selected: list[str] = []
    for name in names:
        meta = metadata[name]
        if any(not getattr(meta, field) for field in include):
            continue
        if any(getattr(meta, field) for field in exclude):
            continue
        selected.append(name)
    return selected


def _is_reduced_main(name: str, meta: ModelMetadata) -> bool:
    """Whether ``name`` belongs on the reduced plot main axis."""
    if meta.vector_gain:
        return False
    if meta.one_d_pos and meta.main:
        return True
    if name == "rrr":
        return True
    if "_no_intercept" in name:
        return False
    if name.endswith("_decoder_only_1dspeed"):
        return True
    if name.endswith("_decoder_only") and not meta.high_d_speed:
        return True
    return False


def _is_reduced_high_d_speed(name: str, meta: ModelMetadata) -> bool:
    """Whether ``name`` is a reduced high-D-speed overlay."""
    return meta.high_d_speed and "decoder_only" in name and not meta.internal and not meta.leak


def _is_reduced_no_intercept(name: str, meta: ModelMetadata) -> bool:
    """Whether ``name`` is a reduced no-intercept overlay."""
    return meta.no_intercept and not meta.internal and not meta.leak and ("decoder_only" in name or name == "rrr_no_intercept")


def _resolve_layer_names(
    preset: PlotPreset,
    layer_key: LayerKey,
    *,
    include_internal: bool,
    include_leak: bool,
    names: tuple[str, ...],
    metadata: dict[str, ModelMetadata],
) -> list[str]:
    """Resolve model names for one plot layer under a preset."""
    exclude: dict[str, bool] = {}
    if not include_internal:
        exclude["internal"] = True
    if not include_leak:
        exclude["leak"] = True
    if layer_key == "high_d_speed":
        exclude["no_intercept"] = True

    if preset == "full":
        if layer_key == "main":
            return select_models(
                include={"main": True},
                exclude={"vector_gain": True, **exclude},
                names=names,
                metadata=metadata,
            )
        if layer_key == "high_d_speed":
            return select_models(
                include={"high_d_speed": True},
                exclude=exclude or None,
                names=names,
                metadata=metadata,
            )
        if layer_key == "no_intercept":
            return select_models(
                include={"no_intercept": True},
                exclude=exclude or None,
                names=names,
                metadata=metadata,
            )
        if layer_key == "vector_gain":
            return select_models(
                include={"vector_gain": True},
                exclude=exclude or None,
                names=names,
                metadata=metadata,
            )

    selected: list[str] = []
    for name in names:
        meta = metadata[name]
        if any(getattr(meta, field) for field in exclude):
            continue
        if layer_key == "main" and _is_reduced_main(name, meta):
            selected.append(name)
        elif layer_key == "high_d_speed" and _is_reduced_high_d_speed(name, meta):
            selected.append(name)
        elif layer_key == "no_intercept" and _is_reduced_no_intercept(name, meta):
            selected.append(name)
        elif layer_key == "vector_gain" and meta.vector_gain:
            selected.append(name)
    return selected


def _assert_disjoint_layers(layers: dict[str, PlotLayer]) -> None:
    """Raise if any model name appears in more than one layer."""
    seen: dict[str, str] = {}
    for key, layer in layers.items():
        for name in layer.names:
            if name in seen:
                raise ValueError(f"Model {name!r} appears in layers {seen[name]!r} and {key!r}; " "each model may belong to only one layer.")
            seen[name] = key


def _resolve_overlay_anchors(
    overlay_names: tuple[str, ...],
    main_names: tuple[str, ...],
    metadata: dict[str, ModelMetadata],
) -> tuple[str, ...]:
    """Return parallel anchor names for overlay models."""
    main_set = set(main_names)
    anchors: list[str] = []
    for name in overlay_names:
        reference = metadata[name].reference
        if reference is None:
            raise ValueError(f"Overlay model {name!r} has no reference; cannot place on plot.")
        if reference not in main_set:
            raise ValueError(f"Overlay model {name!r} references {reference!r}, which is not on the main layer.")
        anchors.append(reference)
    return tuple(anchors)


@dataclass(frozen=True)
class PlotLayer:
    """One sequence of models (and optional overlay anchors) for plotting."""

    key: str
    names: tuple[str, ...]
    anchor_names: tuple[str, ...] | None = None


class ModelPlotLocator:
    """Resolve main and overlay model lists for regression comparison plots."""

    def __init__(
        self,
        preset: PlotPreset = "reduced",
        layers: tuple[LayerKey, ...] = DEFAULT_LAYERS,
        *,
        include_internal: bool = True,
        include_leak: bool = True,
        names: tuple[str, ...] | None = None,
        metadata: dict[str, ModelMetadata] | None = None,
        disjoint_layers: bool = True,
    ) -> None:
        if names is None or metadata is None:
            from .registry import MODEL_METADATA, MODEL_NAMES

            names = MODEL_NAMES if names is None else names
            metadata = MODEL_METADATA if metadata is None else metadata
        self.preset = preset
        self.layer_keys = layers
        self.include_internal = include_internal
        self.include_leak = include_leak
        self.names = names
        self.metadata = metadata
        self.disjoint_layers = disjoint_layers
        self._layers = self._build_layers()

    def _build_layers(self) -> dict[str, PlotLayer]:
        """Construct validated plot layers."""
        overlay_keys = [key for key in self.layer_keys if key != "main"]
        if overlay_keys and "main" not in self.layer_keys:
            raise ValueError(f"Overlay layers {overlay_keys!r} require 'main' in layers.")

        built: dict[str, PlotLayer] = {}
        main_names: tuple[str, ...] = ()

        if "main" in self.layer_keys:
            main_list = _resolve_layer_names(
                self.preset,
                "main",
                include_internal=self.include_internal,
                include_leak=self.include_leak,
                names=self.names,
                metadata=self.metadata,
            )
            main_names = tuple(main_list)
            built["main"] = PlotLayer(key="main", names=main_names)

        for layer_key in self.layer_keys:
            if layer_key == "main":
                continue
            overlay_list = _resolve_layer_names(
                self.preset,
                layer_key,
                include_internal=self.include_internal,
                include_leak=self.include_leak,
                names=self.names,
                metadata=self.metadata,
            )
            overlay_names = tuple(overlay_list)
            anchor_names = _resolve_overlay_anchors(overlay_names, main_names, self.metadata)
            built[layer_key] = PlotLayer(key=layer_key, names=overlay_names, anchor_names=anchor_names)

        if self.disjoint_layers:
            _assert_disjoint_layers(built)
        return built

    def layers(self) -> dict[str, PlotLayer]:
        """Return plot layers keyed by layer name."""
        return dict(self._layers)

    def _name_to_idx(self, names: tuple[str, ...]) -> list[int]:
        return [self.names.index(name) for name in names]

    def x_positions(self) -> dict[int, int]:
        """Map registry model index to x-tick index on the main axis.

        Returns
        -------
        dict[int, int]
            Keys are indices into :data:`MODEL_NAMES`; values are positions on
            the main-axis tick sequence (``0 .. len(main)-1``).
        """
        main_layer = self._layers["main"]
        return {idx: x for x, idx in enumerate(main_layer.names)}

    def x_values_for_layer(self, layer_key: str, as_dict: bool = True) -> list[int] | dict[str, int]:
        """Return x-tick positions for plotting one overlay layer.

        Parameters
        ----------
        layer_key
            Layer name (e.g. ``"high_d_speed"``).
        as_dict : bool, default True
            If True, return a dictionary mapping model names to x-tick indices.
            If False, return a list of x-tick indices.

        Returns
        -------
        list[int]
            Parallel to that layer's model list; each entry is an x-tick index.
        dict[str, int]
            Keys are model names in that layer; values are x-tick indices.
        """
        x_map_main = self.x_positions()
        if layer_key == "main":
            if as_dict:
                return x_map_main
            return list(x_map_main.values())

        layer = self._layers[layer_key]
        if layer.anchor_names is None:
            raise ValueError(f"Layer {layer_key!r} has no anchors (not an overlay layer).")
        if as_dict:
            return {name: x_map_main[anchor] for name, anchor in zip(layer.names, layer.anchor_names)}
        return [x_map_main[anchor] for anchor in layer.anchor_names]

    def as_dict(self) -> dict[str, list[int] | list[str]]:
        """Flatten layers into parallel name and index lists.

        Returns
        -------
        dict[str, list[int] | list[str]]
            Keys like ``main_name``, ``main_idx``, ``high_d_speed_anchor_idx``, etc.
        """
        flat: dict[str, list[int] | list[str]] = {}
        for key, layer in self._layers.items():
            flat[f"{key}_name"] = list(layer.names)
            flat[f"{key}_idx"] = self._name_to_idx(layer.names)
            if layer.anchor_names is not None:
                flat[f"{key}_anchor_name"] = list(layer.anchor_names)
                flat[f"{key}_anchor_idx"] = self._name_to_idx(layer.anchor_names)
        return flat

    @property
    def x_values(self) -> dict[str, list[int] | dict[int, int] | dict[str, int]]:
        return dict(
            main=self.x_values_for_layer("main"),
            high_d_speed=self.x_values_for_layer("high_d_speed"),
            no_intercept=self.x_values_for_layer("no_intercept"),
            vector_gain=self.x_values_for_layer("vector_gain"),
        )
