"""Aggregate stored results from a ResultsStore into padded ndarrays."""

from __future__ import annotations

import warnings
import numpy as np
import torch


class ResultsAggregator:
    """Load stored results for a config class into padded ndarrays.

    Parameters
    ----------
    config_class : type[AnalysisConfigBase]
        The analysis config class whose results to load.
    store : ResultsStore
        The results store to load from.
    sessions : list
        Session objects (B2Session) to include.  Order determines axis 0.

    Attributes
    ----------
    arrays : dict[str, np.ndarray]
        {result_key: array of shape (n_sess, *param_dims, *max_result_shape)}.
        Scalar (0-D) results have shape (n_sess, *param_dims).
    param_axes : dict[str, list]
        {param_name: [value0, value1, ...]} — ordered axis values for each
        parameter dimension.
    session_ids : list[str]
        session.session_uid for each session in axis 0.
    mouse_names : np.ndarray
        Shape (n_sess,) — session.mouse_name for each session.
    result_shapes : dict[str, np.ndarray]
        {result_key: array of shape (n_sess, *param_dims, ndim)} storing the
        un-padded shape of each (session, variation) result.  Only present for
        results with ndim >= 1; absent for scalars.
    """

    def __init__(self, config_class, store, sessions):
        self.config_class = config_class
        self.store = store
        self.sessions = list(sessions)
        self._load()

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
        obj.arrays = arrays
        obj.param_axes = param_axes
        obj.session_ids = session_ids
        obj.mouse_names = mouse_names
        obj.result_shapes = result_shapes if result_shapes is not None else {}
        obj.objects = objects if objects is not None else {}
        return obj

    def _load(self) -> None:
        config_class = self.config_class

        # --- 1. param_axes and param shape ---
        param_axes: dict[str, list] = {k: list(v) for k, v in config_class._param_grid().items()}
        param_names = list(param_axes.keys())
        param_shape = tuple(len(v) for v in param_axes.values())

        # --- 2. variation_index: key -> position tuple in param grid ---
        variations = config_class.generate_variations()
        variation_index: dict[str, tuple] = {}
        for var in variations:
            idx = tuple(param_axes[name].index(getattr(var, name)) for name in param_names)
            variation_index[var.key()] = idx

        # --- 3. session index and metadata ---
        n_sess = len(self.sessions)
        session_index: dict[str, int] = {s.session_uid: i for i, s in enumerate(self.sessions)}
        session_map: dict[str, object] = {s.session_uid: s for s in self.sessions}
        session_ids = [s.session_uid for s in self.sessions]
        mouse_names = np.array([s.mouse_name for s in self.sessions])

        def _empty():
            self.arrays = {}
            self.objects = {}
            self.param_axes = param_axes
            self.session_ids = session_ids
            self.mouse_names = mouse_names
            self.result_shapes = {}

        # --- 4. filter summary table ---
        df = self.store.summary_table(as_dataframe=True)
        if df.empty:
            return _empty()

        mask = df["analysis_type"] == config_class.display_name
        df = df[mask]
        if df.empty:
            return _empty()

        # --- 5. Pass 1: collect (sess_idx, var_idx, result_dict) triples ---
        result_handling: dict[str, str] = getattr(config_class, "_result_handling", {})
        triples: list[tuple[int, tuple, dict]] = []
        objects_triples: list[tuple[int, tuple, dict]] = []
        for row in df.to_dict("records"):
            session_id = row["session_id"]
            if session_id not in session_index:
                continue
            analysis_key = row["analysis_key"]
            if analysis_key not in variation_index:
                continue

            sess_idx = session_index[session_id]
            var_idx = variation_index[analysis_key]

            try:
                cfg = config_class.from_key(analysis_key)
                result = cfg.get_result(self.store, row)
            except Exception as exc:
                warnings.warn(
                    f"get_result failed for session={session_id} key={analysis_key}: {exc}",
                    stacklevel=2,
                )
                continue

            if result is None:
                continue

            filtered: dict[str, np.ndarray] = {}
            skip_filtered: dict = {}
            for k, v in result.items():
                if result_handling.get(k, "pad") == "skip":
                    skip_filtered[k] = v
                    continue
                if isinstance(v, np.ndarray):
                    filtered[k] = v
                elif np.isscalar(v):
                    filtered[k] = np.asarray(v)
                elif isinstance(v, torch.Tensor):
                    filtered[k] = v.cpu().numpy()
                else:
                    warnings.warn(
                        f"Skipping result key {k!r} (session={session_id}): " f"type {type(v).__name__!r} is not an ndarray or scalar.",
                        stacklevel=2,
                    )

            if filtered:
                triples.append((sess_idx, var_idx, filtered))
            if skip_filtered:
                objects_triples.append((sess_idx, var_idx, skip_filtered))

        if not triples:
            return _empty()

        # --- 6. Pass 2: determine max shape per key, allocate, fill ---
        all_keys: set[str] = set()
        for _, _, result in triples:
            all_keys.update(result.keys())

        ragged_keys = {k for k in all_keys if result_handling.get(k, "pad") == "ragged"}
        pad_keys = all_keys - ragged_keys

        max_shapes: dict[str, tuple[int, ...]] = {}
        result_ndims: dict[str, int] = {}
        for key in pad_keys:
            shapes = [result[key].shape for _, _, result in triples if key in result]
            ndims = {len(s) for s in shapes}
            if len(ndims) > 1:
                warnings.warn(
                    f"Result key {key!r} has inconsistent ndim across results: {ndims}. Skipping.",
                    stacklevel=2,
                )
                continue
            ndim = next(iter(ndims))
            result_ndims[key] = ndim
            max_shapes[key] = () if ndim == 0 else tuple(max(s[i] for s in shapes) for i in range(ndim))

        arrays: dict[str, np.ndarray] = {}
        shape_arrays: dict[str, np.ndarray] = {}
        for key, max_shape in max_shapes.items():
            arrays[key] = np.full((n_sess,) + param_shape + max_shape, np.nan)
            ndim = result_ndims[key]
            if ndim >= 1:
                shape_arrays[key] = np.zeros((n_sess,) + param_shape + (ndim,), dtype=np.intp)
        for key in ragged_keys:
            arrays[key] = np.empty((n_sess,) + param_shape, dtype=object)

        for sess_idx, var_idx, result in triples:
            full_idx = (sess_idx,) + var_idx
            for key, val in result.items():
                if key not in arrays:
                    continue
                if key in ragged_keys:
                    arrays[key][full_idx] = val
                else:
                    ndim = result_ndims[key]
                    if ndim == 0:
                        arrays[key][full_idx] = val
                    else:
                        target = arrays[key][full_idx]
                        target[tuple(slice(0, s) for s in val.shape)] = val
                        shape_arrays[key][full_idx] = val.shape

        self.arrays = arrays
        self.param_axes = param_axes
        self.session_ids = session_ids
        self.mouse_names = mouse_names
        self.result_shapes = shape_arrays

        # --- 7. Build self.objects from skip-keyed values ---
        skip_keys: set[str] = set()
        for _, _, d in objects_triples:
            skip_keys.update(d.keys())
        objects: dict[str, np.ndarray] = {k: np.empty((n_sess,) + param_shape, dtype=object) for k in skip_keys}
        for sess_idx, var_idx, d in objects_triples:
            full_idx = (sess_idx,) + var_idx
            for key, val in d.items():
                objects[key][full_idx] = val
        self.objects = objects

    def sel(
        self,
        mouse: str = None,
        squeeze_ones: bool = True,
        return_param_sizes: bool = False,
        **params,
    ) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, list]]:
        """Return arrays sliced to specific param values, with those dims squeezed.

        Parameters
        ----------
        mouse
            The mouse name to select.  If None, includes all mice.
        squeeze_ones
            Whether to squeeze dimensions of size 1.
        return_param_sizes
            Whether to return the updated parameter axes.
        **params
            {param_name: value} pairs.  Each name must be in ``param_axes``.

        Returns
        -------
        dict[str, np.ndarray]
            Arrays with each specified param dimension removed.
        param_sizes : dict[str, int], optional
            Returned if ``return_param_sizes`` is True.

        Examples
        --------
        >>> sliced = results.sel(center=True, normalize=False)
        >>> sliced["reg_covariances"].shape  # (n_sess, remaining_params..., max_len)
        """
        param_names = list(self.param_axes.keys())
        param_axes = ["session"]

        idx: list = [slice(None)]  # session dim
        if mouse is not None:
            mask = self.mouse_names == mouse
            idx[0] = mask

        for param in params:
            if param not in self.param_axes:
                raise ValueError(f"Invalid parameter name {param!r} — must be one of {param_names}")
        for name in param_names:
            if name in params:
                idx.append(self.param_axes[name].index(params[name]))
            else:
                idx.append(slice(None))
                param_axes.append(name)

        idx_tuple = tuple(idx)
        out = {k: v[idx_tuple] for k, v in self.arrays.items()}
        if squeeze_ones:
            _example = next(iter(out.values()))
            param_axes = [name for i, name in enumerate(param_axes) if _example.shape[i] > 1]
            out = {k: v.squeeze() for k, v in out.items()}
        if return_param_sizes:
            _example = next(iter(out.values()))
            param_sizes = {name: _example.shape[i] for i, name in enumerate(param_axes)}
            return out, param_sizes
        return out

    def sel_objects(
        self,
        mouse: str = None,
        **params,
    ) -> dict[str, np.ndarray]:
        """Return skip-keyed object arrays sliced to specific param values.

        Same filtering interface as :meth:`sel` but operates on ``self.objects``
        (values marked ``"skip"`` in ``_result_handling``).  No squeezing is
        applied — the returned arrays always have dtype ``object`` and shape
        ``(n_sess_or_mouse, *remaining_param_dims)``.

        Parameters
        ----------
        mouse
            If given, restrict to sessions whose ``mouse_name`` matches.
        **params
            ``{param_name: value}`` pairs to index into the param grid.

        Returns
        -------
        dict[str, np.ndarray]
            Object arrays of shape ``(n_sel, *remaining_param_dims)``.
        """
        param_names = list(self.param_axes.keys())
        idx: list = [slice(None)]
        if mouse is not None:
            idx[0] = self.mouse_names == mouse
        for name in param_names:
            if name in params:
                idx.append(self.param_axes[name].index(params[name]))
            else:
                idx.append(slice(None))
        idx_tuple = tuple(idx)
        return {k: v[idx_tuple] for k, v in self.objects.items()}

    def average_by_mouse(self) -> ResultsAggregator:
        """Average arrays across sessions with the same mouse_name.

        NaN-safe: uses ``np.nanmean``.

        Returns
        -------
        ResultsAggregator
            Shape (n_mice, *param_dims, *max_result_shape).
        """
        unique_mice: list[str] = list(dict.fromkeys(self.mouse_names.tolist()))
        n_mice = len(unique_mice)

        new_arrays: dict[str, np.ndarray] = {}
        for key, arr in self.arrays.items():
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
        for key, shape_arr in self.result_shapes.items():
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
        )
