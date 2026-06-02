"""Aggregate stored results from a ResultsStore into padded ndarrays."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class _CellRef:
    """Index entry for one (session, param-variation) result."""

    sess_idx: int
    var_idx: tuple[int, ...]
    row: dict


def _value_to_array(key: str, value: Any, session_id: str) -> np.ndarray | None:
    """Convert a single result dict value to ndarray, or None if unsupported."""
    if isinstance(value, np.ndarray):
        return value
    if np.isscalar(value):
        return np.asarray(value)
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    warnings.warn(
        f"Skipping result key {key!r} (session={session_id}): " f"type {type(value).__name__!r} is not an ndarray or scalar.",
        stacklevel=3,
    )
    return None


class _LazyArrayDict(Mapping[str, np.ndarray]):
    """Lazy ``results.arrays`` view: loads pad/ragged keys on access."""

    def __init__(self, agg: ResultsAggregator) -> None:
        self._agg = agg

    def __getitem__(self, key: str) -> np.ndarray:
        backend = self._agg._arrays_backend
        if key not in backend:
            load_ragged = self._agg.load_ragged or key in self._agg._ragged_keys
            self._agg._materialize_keys([key], scope="full_grid", load_ragged=load_ragged)
        return backend[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self):
        return self._agg._known_pad_keys()

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.keys()

    def items(self):
        for key in list(self.keys()):
            yield key, self[key]


class _LazyObjectDict(Mapping[str, np.ndarray]):
    """Lazy ``results.objects`` view: loads skip keys on access."""

    def __init__(self, agg: ResultsAggregator) -> None:
        self._agg = agg

    def __getitem__(self, key: str) -> np.ndarray:
        backend = self._agg._objects_backend
        if key not in backend:
            self._agg._materialize_keys([key], scope="full_grid", load_objects=True)
        return backend[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self):
        return self._agg._known_skip_keys()

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.keys()

    def items(self):
        for key in list(self.keys()):
            yield key, self[key]


class _LazyShapeDict(Mapping[str, np.ndarray]):
    """Lazy ``results.result_shapes`` view (follows pad keys in ``arrays``)."""

    def __init__(self, agg: ResultsAggregator) -> None:
        self._agg = agg

    def __getitem__(self, key: str) -> np.ndarray:
        _ = self._agg.arrays[key]
        return self._agg._result_shapes_backend[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._agg._result_shapes_backend)

    def __len__(self) -> int:
        return len(self._agg._result_shapes_backend)

    def keys(self):
        return self._agg._result_shapes_backend.keys()

    def __contains__(self, key: object) -> bool:
        return key in self._agg._result_shapes_backend

    def items(self):
        return self._agg._result_shapes_backend.items()


class ResultsAggregator:
    """Load stored results for a config class into padded ndarrays.

    By default (``lazy=True``), only metadata and a cell index are built at
    init; blobs are unpickled when :meth:`sel`, :meth:`sel_objects`, or dict
    access requests data.

    Parameters
    ----------
    config_class : type[AnalysisConfigBase]
        The analysis config class whose results to load.
    store : ResultsStore
        The results store to load from.
    sessions : list
        Session objects (B2Session) to include.  Order determines axis 0.
    lazy : bool
        If True (default), defer blob loading until accessed.
    keys : list of str, optional
        If given, only these pad keys are loaded by :meth:`sel` / materialize
        (unless a call passes an explicit ``keys``).
    load_ragged : bool
        If True, include ``"ragged"`` keys in pad loading.

    Attributes
    ----------
    arrays : dict[str, np.ndarray]
        {result_key: array of shape (n_sess, *param_dims, *max_result_shape)}.
    objects : dict[str, np.ndarray]
        Skip-keyed object arrays (``"skip"`` in ``_result_handling``).
    param_axes : dict[str, list]
        {param_name: [value0, value1, ...]}.
    session_ids : list[str]
        session.session_uid for each session in axis 0.
    mouse_names : np.ndarray
        Shape (n_sess,).
    result_shapes : dict[str, np.ndarray]
        Un-padded shapes for pad keys with ndim >= 1.
    """

    def __init__(
        self,
        config_class,
        store,
        sessions,
        *,
        lazy: bool = True,
        keys: list[str] | None = None,
        load_ragged: bool = False,
    ):
        self.config_class = config_class
        self.store = store
        self.sessions = list(sessions)
        self.lazy = lazy
        self._default_keys = keys
        self.load_ragged = load_ragged

        self._result_handling: dict[str, str] = getattr(config_class, "_result_handling", {})
        self._ragged_keys = {k for k, h in self._result_handling.items() if h == "ragged"}
        self._skip_keys = {k for k, h in self._result_handling.items() if h == "skip"}
        self._discovered_keys: set[str] = set()
        self._key_to_config: dict[str, Any] = {}

        self._arrays_backend: dict[str, np.ndarray] = {}
        self._objects_backend: dict[str, np.ndarray] = {}
        self._result_shapes_backend: dict[str, np.ndarray] = {}
        self._filled_pad: set[tuple[str, tuple[int, ...]]] = set()
        self._filled_objects: set[tuple[str, tuple[int, ...]]] = set()

        self._cells: dict[tuple[int, tuple[int, ...]], _CellRef] = {}
        self.variation_index: dict[str, tuple[int, ...]] = {}
        self._session_index: dict[str, int] = {}
        self._param_names: list[str] = []
        self._param_shape: tuple[int, ...] = ()

        self._build_metadata()
        self._build_index()

        if lazy:
            self.arrays = _LazyArrayDict(self)
            self.objects = _LazyObjectDict(self)
            self.result_shapes = _LazyShapeDict(self)
        else:
            self._load_eager()
            self.arrays = self._arrays_backend
            self.objects = self._objects_backend
            self.result_shapes = self._result_shapes_backend

    @classmethod
    def _from_data(
        cls,
        arrays: dict[str, np.ndarray],
        param_axes: dict[str, list],
        session_ids: list[str],
        mouse_names: np.ndarray,
        result_shapes: dict[str, np.ndarray] | None = None,
        objects: dict[str, np.ndarray] | None = None,
    ) -> ResultsAggregator:
        """Construct directly from pre-loaded data (used by average_by_mouse)."""
        obj = object.__new__(cls)
        obj.config_class = None
        obj.store = None
        obj.sessions = []
        obj.lazy = False
        obj._default_keys = None
        obj.load_ragged = False
        obj._result_handling = {}
        obj._ragged_keys = set()
        obj._skip_keys = set()
        obj._discovered_keys = set()
        obj._arrays_backend = arrays
        obj._objects_backend = objects if objects is not None else {}
        obj._result_shapes_backend = result_shapes if result_shapes is not None else {}
        obj._filled_pad = set()
        obj._filled_objects = set()
        obj._cells = {}
        obj.variation_index = {}
        obj._session_index = {}
        obj._key_to_config = {}
        obj._param_names = list(param_axes.keys())
        obj._param_shape = tuple(len(v) for v in param_axes.values())
        obj.arrays = arrays
        obj.objects = obj._objects_backend
        obj.result_shapes = obj._result_shapes_backend
        obj.param_axes = param_axes
        obj.session_ids = session_ids
        obj.mouse_names = mouse_names
        return obj

    def _build_metadata(self) -> None:
        config_class = self.config_class
        self.param_axes = {k: list(v) for k, v in config_class._param_grid().items()}
        self._param_names = list(self.param_axes.keys())
        self._param_shape = tuple(len(v) for v in self.param_axes.values())

        variations = config_class.generate_variations()
        self.variation_index = {}
        self._key_to_config = {}
        for var in variations:
            k = var.key()
            idx = tuple(self.param_axes[name].index(getattr(var, name)) for name in self._param_names)
            self.variation_index[k] = idx
            self._key_to_config[k] = var

        self._session_index = {s.session_uid: i for i, s in enumerate(self.sessions)}
        self.session_ids = [s.session_uid for s in self.sessions]
        self.mouse_names = np.array([s.mouse_name for s in self.sessions])

    def _build_index(self) -> None:
        session_ids = list(self._session_index.keys())
        records = self.store.summary_table(
            analysis_type=self.config_class.display_name,
            session_ids=session_ids,
        )
        for row in records:
            session_id = row["session_id"]
            analysis_key = row["analysis_key"]
            if session_id not in self._session_index:
                continue
            if analysis_key not in self.variation_index:
                continue
            sess_idx = self._session_index[session_id]
            var_idx = self.variation_index[analysis_key]
            self._cells[(sess_idx, var_idx)] = _CellRef(sess_idx, var_idx, row)

    def _known_pad_keys(self) -> set[str]:
        self._peek_discover_keys()
        keys = {k for k, h in self._result_handling.items() if h in ("pad", "ragged")}
        keys |= self._discovered_keys - self._skip_keys
        if not self.load_ragged:
            keys -= self._ragged_keys
        keys |= set(self._arrays_backend)
        return keys

    def _known_skip_keys(self) -> set[str]:
        return self._skip_keys | {k for k in self._discovered_keys if self._result_handling.get(k, "pad") == "skip"} | set(self._objects_backend)

    def _peek_discover_keys(self) -> None:
        if self._discovered_keys or not self._cells:
            return
        ref = next(iter(self._cells.values()))
        result = self._fetch_result(ref)
        if result is not None:
            self._discovered_keys.update(result.keys())

    def _fetch_result(self, ref: _CellRef) -> dict | None:
        try:
            analysis_key = ref.row["analysis_key"]
            cfg = self._key_to_config.get(analysis_key) or self.config_class.from_key(analysis_key)
            return cfg.get_result(self.store, ref.row)
        except Exception as exc:
            warnings.warn(
                f"get_result failed for session={ref.row['session_id']} " f"key={ref.row['analysis_key']}: {exc}",
                stacklevel=2,
            )
            return None

    def _select_cells(
        self,
        mouse: str | None = None,
        param_indices: dict[str, int] | None = None,
    ) -> list[_CellRef]:
        """Return cell refs matching optional session mouse filter and fixed param indices."""
        param_indices = param_indices or {}
        name_to_dim = {name: i for i, name in enumerate(self._param_names)}
        out: list[_CellRef] = []
        for ref in self._cells.values():
            if mouse is not None and self.mouse_names[ref.sess_idx] != mouse:
                continue
            matched = True
            for name, pidx in param_indices.items():
                dim = name_to_dim[name]
                if ref.var_idx[dim] != pidx:
                    matched = False
                    break
            if matched:
                out.append(ref)
        return out

    def _extract_from_result(
        self,
        result: dict,
        session_id: str,
        *,
        keys: set[str] | None,
        load_pad: bool,
        load_ragged: bool,
        load_objects: bool,
    ) -> tuple[dict[str, np.ndarray], dict]:
        pad_part: dict[str, np.ndarray] = {}
        skip_part: dict = {}
        for k, v in result.items():
            self._discovered_keys.add(k)
            if keys is not None and k not in keys:
                continue
            handling = self._result_handling.get(k, "pad")
            if handling == "skip":
                if load_objects:
                    skip_part[k] = v
                continue
            if handling == "ragged":
                if not load_ragged:
                    continue
            elif not load_pad:
                continue
            arr = _value_to_array(k, v, session_id)
            if arr is not None:
                pad_part[k] = arr
        return pad_part, skip_part

    def _cell_needs_materialize(
        self,
        ref: _CellRef,
        key_set: set[str] | None,
        *,
        load_pad: bool,
        load_ragged: bool,
        load_objects: bool,
    ) -> bool:
        """Return True if any requested key for this cell is not yet filled."""
        full_idx = (ref.sess_idx,) + ref.var_idx
        if load_objects:
            skip_keys = key_set if key_set is not None else self._known_skip_keys()
            if any((k, full_idx) not in self._filled_objects for k in skip_keys):
                return True
        if load_pad or load_ragged:
            if key_set is not None:
                pad_keys = key_set
            else:
                self._peek_discover_keys()
                pad_keys = self._known_pad_keys()
            if load_ragged:
                pad_keys = set(pad_keys) | self._ragged_keys
            elif not self.load_ragged and not load_ragged:
                pad_keys = {k for k in pad_keys if k not in self._ragged_keys}
            if any((k, full_idx) not in self._filled_pad for k in pad_keys):
                return True
        return False

    def _materialize_keys(
        self,
        keys: list[str] | None,
        *,
        scope: str,
        load_pad: bool = True,
        load_ragged: bool = False,
        load_objects: bool = False,
        mouse: str | None = None,
        param_indices: dict[str, int] | None = None,
    ) -> None:
        if scope == "full_grid":
            cells = list(self._cells.values())
        else:
            cells = self._select_cells(mouse=mouse, param_indices=param_indices)

        key_set = set(keys) if keys is not None else None
        pad_triples: list[tuple[int, tuple[int, ...], dict[str, np.ndarray]]] = []
        obj_triples: list[tuple[int, tuple[int, ...], dict]] = []

        cells_to_fetch = [
            ref
            for ref in cells
            if self._cell_needs_materialize(
                ref,
                key_set,
                load_pad=load_pad,
                load_ragged=load_ragged,
                load_objects=load_objects,
            )
        ]

        n_workers = min(8, len(cells_to_fetch))
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(self._fetch_result, ref): ref for ref in cells_to_fetch}
                raw_results = [(futures[f], f.result()) for f in as_completed(futures)]
        else:
            raw_results = [(ref, self._fetch_result(ref)) for ref in cells_to_fetch]

        for ref, result in raw_results:
            if result is None:
                continue
            pad_part, skip_part = self._extract_from_result(
                result,
                ref.row["session_id"],
                keys=key_set,
                load_pad=True,
                load_ragged=load_ragged,
                load_objects=load_objects,
            )
            if pad_part:
                pad_triples.append((ref.sess_idx, ref.var_idx, pad_part))
            if skip_part:
                obj_triples.append((ref.sess_idx, ref.var_idx, skip_part))

        if pad_triples:
            self._merge_pad_triples(pad_triples, load_ragged=load_ragged)
        if obj_triples:
            self._merge_object_triples(obj_triples)

    def _merge_pad_triples(
        self,
        triples: list[tuple[int, tuple[int, ...], dict[str, np.ndarray]]],
        *,
        load_ragged: bool,
    ) -> None:
        n_sess = len(self.sessions)
        param_shape = self._param_shape
        result_handling = self._result_handling

        new_keys: set[str] = set()
        for _, _, result in triples:
            new_keys.update(result.keys())

        for key in new_keys:
            handling = result_handling.get(key, "pad")
            is_ragged = handling == "ragged"

            shapes = [result[key].shape for _, _, result in triples if key in result]
            if not shapes:
                continue
            ndims = {len(s) for s in shapes}
            if len(ndims) > 1:
                warnings.warn(
                    f"Result key {key!r} has inconsistent ndim across results: {ndims}. Skipping.",
                    stacklevel=2,
                )
                continue
            ndim = next(iter(ndims))
            max_shape = () if ndim == 0 else tuple(max(s[i] for s in shapes) for i in range(ndim))

            if key not in self._arrays_backend:
                if is_ragged:
                    self._arrays_backend[key] = np.empty((n_sess,) + param_shape, dtype=object)
                else:
                    self._arrays_backend[key] = np.full((n_sess,) + param_shape + max_shape, np.nan)
                    if ndim >= 1:
                        self._result_shapes_backend[key] = np.zeros(
                            (n_sess,) + param_shape + (ndim,),
                            dtype=np.intp,
                        )
            else:
                self._grow_pad_array(key, max_shape, ndim, is_ragged)

            for sess_idx, var_idx, result in triples:
                if key not in result:
                    continue
                cell_key = (key, (sess_idx,) + var_idx)
                if cell_key in self._filled_pad:
                    continue
                val = result[key]
                full_idx = (sess_idx,) + var_idx
                if is_ragged:
                    self._arrays_backend[key][full_idx] = val
                elif ndim == 0:
                    self._arrays_backend[key][full_idx] = val
                else:
                    target = self._arrays_backend[key][full_idx]
                    target[tuple(slice(0, s) for s in val.shape)] = val
                    self._result_shapes_backend[key][full_idx] = val.shape
                self._filled_pad.add(cell_key)

    def _grow_pad_array(self, key: str, max_shape: tuple[int, ...], ndim: int, is_ragged: bool) -> None:
        if is_ragged:
            return
        arr = self._arrays_backend[key]
        trailing = arr.shape[len(self._param_shape) + 1 :]
        if trailing == max_shape:
            return
        if len(trailing) != len(max_shape):
            warnings.warn(f"Cannot grow result key {key!r}: ndim mismatch.", stacklevel=2)
            return
        if all(t >= m for t, m in zip(trailing, max_shape)):
            return
        new_trailing = tuple(max(t, m) for t, m in zip(trailing, max_shape))
        n_sess = len(self.sessions)
        new_arr = np.full((n_sess,) + self._param_shape + new_trailing, np.nan)
        slices = (slice(None),) * (1 + len(self._param_shape)) + tuple(slice(0, s) for s in trailing)
        new_arr[slices] = arr[slices]
        self._arrays_backend[key] = new_arr
        if key in self._result_shapes_backend:
            old_shapes = self._result_shapes_backend[key]
            new_shapes = np.zeros((n_sess,) + self._param_shape + (ndim,), dtype=np.intp)
            shape_slices = (slice(None),) * (1 + len(self._param_shape)) + (slice(None),)
            new_shapes[shape_slices] = old_shapes[shape_slices]
            self._result_shapes_backend[key] = new_shapes

    def _merge_object_triples(self, triples: list[tuple[int, tuple[int, ...], dict]]) -> None:
        n_sess = len(self.sessions)
        param_shape = self._param_shape
        for _, _, d in triples:
            for key in d:
                if key not in self._objects_backend:
                    self._objects_backend[key] = np.empty((n_sess,) + param_shape, dtype=object)
        for sess_idx, var_idx, d in triples:
            full_idx = (sess_idx,) + var_idx
            for key, val in d.items():
                cell_key = (key, (sess_idx,) + var_idx)
                if cell_key in self._filled_objects:
                    continue
                self._objects_backend[key][full_idx] = val
                self._filled_objects.add(cell_key)

    def load_all(self, *, load_ragged: bool | None = None, load_objects: bool = True) -> None:
        """Materialize all pad keys (and optionally skip/ragged keys) on the full grid.

        Parameters
        ----------
        load_ragged : bool or None
            If None, uses the aggregator's ``load_ragged`` setting.
        load_objects : bool
            Whether to load skip-keyed ``objects``.
        """
        if load_ragged is None:
            load_ragged = self.load_ragged
        self._peek_discover_keys()
        pad_keys = list(self._known_pad_keys())
        if pad_keys:
            self._materialize_keys(pad_keys, scope="full_grid", load_ragged=load_ragged)
        if load_objects:
            skip_keys = list(self._known_skip_keys())
            if skip_keys:
                self._materialize_keys(skip_keys, scope="full_grid", load_objects=True)

    def _load_eager(self) -> None:
        """Eagerly load all pad and skip keys (legacy ``lazy=False`` path)."""
        self.load_all(load_ragged=True, load_objects=True)

    def _sel_index(
        self,
        mouse: str | None,
        params: dict[str, Any],
    ) -> tuple[tuple, list[str], dict[str, int]]:
        param_names = self._param_names
        axis_names = ["session"]
        idx: list = [slice(None)]
        if mouse is not None:
            idx[0] = self.mouse_names == mouse
        param_indices: dict[str, int] = {}
        for name in param_names:
            if name in params:
                param_indices[name] = self.param_axes[name].index(params[name])
                idx.append(param_indices[name])
            else:
                idx.append(slice(None))
                axis_names.append(name)
        return tuple(idx), axis_names, param_indices

    def sel(
        self,
        mouse: str = None,
        squeeze_ones: bool = True,
        return_param_sizes: bool = False,
        keys: list[str] | None = None,
        load_ragged: bool | None = None,
        **params,
    ) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, list]]:
        """Return arrays sliced to specific param values, with those dims squeezed."""
        if load_ragged is None:
            load_ragged = self.load_ragged
        effective_keys = keys if keys is not None else self._default_keys
        idx_tuple, axis_names, param_indices = self._sel_index(mouse, params)

        if self.lazy:
            self._materialize_keys(
                effective_keys,
                scope="slice",
                load_ragged=load_ragged,
                load_objects=False,
                mouse=mouse,
                param_indices=param_indices,
            )

        out = {}
        for k, v in self._arrays_backend.items():
            if k in self._ragged_keys and not load_ragged:
                continue
            out[k] = v[idx_tuple]

        if squeeze_ones and out:
            _example = next(iter(out.values()))
            axis_names = [name for i, name in enumerate(axis_names) if _example.shape[i] > 1]
            out = {k: v.squeeze() for k, v in out.items()}
        if return_param_sizes:
            _example = next(iter(out.values()))
            param_sizes = {name: _example.shape[i] for i, name in enumerate(axis_names)}
            return out, param_sizes
        return out

    def sel_objects(
        self,
        mouse: str = None,
        keys: list[str] | None = None,
        **params,
    ) -> dict[str, np.ndarray]:
        """Return skip-keyed object arrays sliced to specific param values."""
        effective_keys = keys if keys is not None else self._default_keys
        idx_tuple, _, param_indices = self._sel_index(mouse, params)

        if self.lazy:
            self._materialize_keys(
                effective_keys,
                scope="slice",
                load_pad=False,
                load_objects=True,
                mouse=mouse,
                param_indices=param_indices,
            )

        return {k: v[idx_tuple] for k, v in self._objects_backend.items()}

    def average_by_mouse(self) -> ResultsAggregator:
        """Average arrays across sessions with the same mouse_name."""
        if self.lazy:
            self._peek_discover_keys()
            pad_keys = list(self._known_pad_keys())
            self._materialize_keys(pad_keys, scope="full_grid", load_ragged=self.load_ragged)

        unique_mice: list[str] = list(dict.fromkeys(self.mouse_names.tolist()))
        n_mice = len(unique_mice)

        new_arrays: dict[str, np.ndarray] = {}
        for key, arr in self._arrays_backend.items():
            if arr.dtype == object:
                warnings.warn(
                    f"Skipping ragged key {key!r} in average_by_mouse() — cannot nanmean object arrays.",
                    stacklevel=2,
                )
                continue
            out = np.full((n_mice,) + arr.shape[1:], np.nan)
            for i, mouse in enumerate(unique_mice):
                mask = self.mouse_names == mouse
                out[i] = np.nanmean(arr[mask], axis=0)
            new_arrays[key] = out

        new_shapes: dict[str, np.ndarray] = {}
        for key, shape_arr in self._result_shapes_backend.items():
            out = np.zeros((n_mice,) + shape_arr.shape[1:], dtype=shape_arr.dtype)
            for i, mouse in enumerate(unique_mice):
                mask = self.mouse_names == mouse
                out[i] = np.max(shape_arr[mask], axis=0)
            new_shapes[key] = out

        return ResultsAggregator._from_data(
            arrays=new_arrays,
            param_axes=self.param_axes,
            session_ids=unique_mice,
            mouse_names=np.array(unique_mice),
            result_shapes=new_shapes,
            objects={},
        )
