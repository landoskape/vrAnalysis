"""In-process cache of the expensive per-session products the figure scripts share.

Every figure function builds its own :class:`SpkmapProcessor` and rederives the same maps.
That is much more expensive than it looks: the figure scripts use ``SpkmapParams()``, whose
``autosave`` default is False, so nothing is ever written to the on-disk cache and
``get_env_maps`` walks the whole raw -> processed -> env pipeline on every call. A single
``stacked_raster_plot`` pays for that pipeline three times over — once inside
``get_placefield_prediction``, once inside ``get_reliability``, once inside
``sort_by_preferred_environment`` — and ``TraversalFocus`` pays for it again on every ROI
slider move, because ``get_traversals`` rederives the prediction and the maps each time.

:class:`CachedSpkmapProcessor` memoizes those methods on the instance, and :func:`get_smp`
hands the same instance to every caller for a given ``(session, spks_type, params)``. Because
it is the same object, code that takes an ``smp`` and calls back into it — most importantly
``sort_by_preferred_environment`` and ``SpkmapProcessor.get_traversals`` — hits the memo too,
without knowing this module exists.

Typical notebook use: pay the cost once in its own cell, then iterate on figures freely.

>>> from dimensionality_manuscript.figure_scripts import session_cache
>>> session_cache.preload(session)          # doctest: +SKIP
>>> fig = stacked_raster_plot(session)      # doctest: +SKIP

The cache lives for the life of the process and is keyed on session identity, ``spks_type``
and the spkmap params, so switching any of those is safe. It is not invalidated by edits to
the code that produced the values — call :func:`clear` after changing map processing.

Mutation
--------
:func:`get_env_maps` returns a shallow copy: the ``Maps`` container is fresh but the arrays
inside it are shared. That is safe for everything the figure scripts do, because
``filter_rois``, ``filter_positions``, ``filter_environments`` and ``average_trials`` all
rebind attributes with new arrays rather than writing into the existing ones. ``smooth_maps``
and ``raw_to_processed`` do write in place and must not be called on a cached ``Maps``.
Arrays returned by the other accessors are shared outright — treat them as read-only.
"""

import time
from copy import copy
from dataclasses import dataclass, field, fields
from typing import Any, Union

import numpy as np
from rastermap import Rastermap

from vrAnalysis import helpers
from vrAnalysis.helpers import sort_by_preferred_environment
from vrAnalysis.metrics import FractionActive
from vrAnalysis.processors import spkmaps as SMPs
from vrAnalysis.processors.placefields import get_frame_behavior, get_placefield, get_placefield_prediction
from vrAnalysis.processors.spkmaps import Maps, Reliability, SpkmapParams
from vrAnalysis.processors.support import median_zscore
from vrAnalysis.sessions import B2Session

#: How a frame-by-frame place-field prediction is built. ``spkmap`` predicts each frame from
#: the trial-averaged environment spkmaps (:meth:`SpkmapProcessor.get_placefield_prediction`);
#: ``placefield`` rebuilds the place field from the z-scored activity with
#: :func:`vrAnalysis.processors.placefields.get_placefield` first.
PREDICTION_MODES = ("spkmap", "placefield")

#: How :func:`get_roi_sort` orders ROIs. ``environment`` sorts by preferred environment then
#: place-field position; ``rastermap`` sorts by a Rastermap embedding of the activity, which
#: is position-agnostic.
SORT_METHODS = ("environment", "rastermap")

_PROCESSORS: dict[tuple, "CachedSpkmapProcessor"] = {}


def _detach(maps: Maps) -> Maps:
    """Shallow copy of ``maps`` sharing its arrays but not the containers holding them."""
    out = copy(maps)
    if maps.by_environment:
        out.occmap = list(maps.occmap)
        out.speedmap = list(maps.speedmap)
        out.spkmap = list(maps.spkmap)
        out.environments = list(maps.environments)
    return out


@dataclass
class CachedSpkmapProcessor(SMPs.SpkmapProcessor):
    """SpkmapProcessor that memoizes its expensive products for the life of the object.

    Only the default call is memoized: passing ``params`` or ``force_recompute`` bypasses the
    memo entirely and delegates straight to :class:`SpkmapProcessor`, so temporary-parameter
    calls keep their usual meaning and never poison the cache.

    ``memo`` doubles as the store for the figure-level products in this module (z-scored
    activity, fraction active, and so on), so dropping the processor drops everything.
    """

    memo: dict = field(default_factory=dict, repr=False, init=False)

    def get_env_maps(
        self,
        use_session_filters: bool = True,
        force_recompute: bool = False,
        clear_one_cache: bool = True,
        params: Union[SpkmapParams, dict, None] = None,
    ) -> Maps:
        if params is not None or force_recompute:
            return super().get_env_maps(
                use_session_filters=use_session_filters,
                force_recompute=force_recompute,
                clear_one_cache=clear_one_cache,
                params=params,
            )
        key = ("env_maps", use_session_filters)
        if key not in self.memo:
            self.memo[key] = super().get_env_maps(
                use_session_filters=use_session_filters,
                clear_one_cache=clear_one_cache,
            )
        return _detach(self.memo[key])

    def get_reliability(
        self,
        use_session_filters: bool = True,
        force_recompute: bool = False,
        clear_one_cache: bool = True,
        params: Union[SpkmapParams, dict, None] = None,
    ) -> Reliability:
        if params is not None or force_recompute:
            return super().get_reliability(
                use_session_filters=use_session_filters,
                force_recompute=force_recompute,
                clear_one_cache=clear_one_cache,
                params=params,
            )
        key = ("reliability", use_session_filters)
        if key not in self.memo:
            self.memo[key] = super().get_reliability(
                use_session_filters=use_session_filters,
                clear_one_cache=clear_one_cache,
            )
        # Shallow copy so a caller rebinding .values doesn't rewrite the cached object.
        return copy(self.memo[key])

    def get_frame_behavior(
        self,
        clear_one_cache: bool = True,
        params: Union[SpkmapParams, dict, None] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if params is not None:
            return super().get_frame_behavior(clear_one_cache, params)
        key = ("frame_behavior",)
        if key not in self.memo:
            self.memo[key] = super().get_frame_behavior(clear_one_cache)
        return self.memo[key]

    def get_placefield_prediction(
        self,
        use_session_filters: bool = True,
        spks_type: Union[str, None] = None,
        use_speed_threshold: bool = True,
        clear_one_cache: bool = True,
        params: Union[SpkmapParams, dict, None] = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if params is not None:
            return super().get_placefield_prediction(
                use_session_filters=use_session_filters,
                spks_type=spks_type,
                use_speed_threshold=use_speed_threshold,
                clear_one_cache=clear_one_cache,
                params=params,
            )
        # spks_type=None means "whatever the session is set to", which is part of the identity.
        effective_spks_type = spks_type if spks_type is not None else self.session.spks_type
        key = ("placefield_prediction", use_session_filters, effective_spks_type, use_speed_threshold)
        if key not in self.memo:
            self.memo[key] = super().get_placefield_prediction(
                use_session_filters=use_session_filters,
                spks_type=spks_type,
                use_speed_threshold=use_speed_threshold,
                clear_one_cache=clear_one_cache,
            )
        prediction, extras = self.memo[key]
        return prediction, dict(extras)

    def nbytes(self) -> int:
        """Total size of the arrays held in the memo, in bytes."""
        total = 0
        for value in self.memo.values():
            total += _nbytes(value)
        return total


def _nbytes(value: Any) -> int:
    """Recursive array-size accounting for the heterogeneous values stored in a memo."""
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, Maps):
        return value.nbytes()
    if isinstance(value, Reliability):
        return value.values.nbytes
    if isinstance(value, dict):
        return sum(_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nbytes(v) for v in value)
    return 0


def _resolve_params(params: Union[SpkmapParams, dict, None]) -> SpkmapParams:
    """Figure-script default params, or the caller's, as a :class:`SpkmapParams`."""
    if params is None:
        return SpkmapParams()
    return helpers.resolve_dataclass(params, SpkmapParams)


def _key(session: B2Session, params: SpkmapParams) -> tuple:
    """Cache identity: which session, read which way, processed with which parameters.

    ``B2Session`` hashes on ``session_name`` alone, so ``spks_type`` has to be added by hand —
    two sessions that compare equal can still produce completely different maps.
    """
    params_key = tuple((f.name, getattr(params, f.name)) for f in fields(params))
    return (session.session_name, session.spks_type, params_key)


def get_smp(session: B2Session, params: Union[SpkmapParams, dict, None] = None) -> CachedSpkmapProcessor:
    """
    The shared memoizing spkmap processor for a session.

    Pass this to anything that takes an ``smp`` — ``sort_by_preferred_environment``,
    ``get_traversals`` — so their internal ``get_env_maps`` / ``get_placefield_prediction``
    calls hit the memo instead of rebuilding the pipeline.

    Parameters
    ----------
    session : B2Session
    params : SpkmapParams, dict, or None
        Spkmap processing parameters. None uses the figure-script default, ``SpkmapParams()``.

    Returns
    -------
    CachedSpkmapProcessor
    """
    params = _resolve_params(params)
    key = _key(session, params)
    if key not in _PROCESSORS:
        _PROCESSORS[key] = CachedSpkmapProcessor(session, params=params)
    return _PROCESSORS[key]


def get_spks(session: B2Session, params: Union[SpkmapParams, dict, None] = None) -> np.ndarray:
    """
    Median z-scored, session-filtered activity, shape ``(frames, rois)``.

    Returns
    -------
    np.ndarray
        Shared read-only array.
    """
    smp = get_smp(session, params)
    key = ("figure_spks",)
    if key not in smp.memo:
        spks = session.spks[:, session.idx_rois]
        smp.memo[key] = median_zscore(spks, median_subtract=not session.zero_baseline_spks)
    return smp.memo[key]


def get_env_maps(session: B2Session, params: Union[SpkmapParams, dict, None] = None) -> Maps:
    """
    Environment-separated maps in the form every figure script wants.

    ``distcenters`` is set and NaN position bins are popped, which is what
    ``example_placefield``, ``example_traversal`` and ``example_placefield_prediction`` each
    did for themselves.

    Returns
    -------
    Maps
        Shallow copy — ``filter_rois``, ``filter_positions``, ``filter_environments`` and
        ``average_trials`` are safe on it; ``smooth_maps`` and ``raw_to_processed`` are not.
    """
    smp = get_smp(session, params)
    key = ("figure_env_maps",)
    if key not in smp.memo:
        env_maps = smp.get_env_maps()
        env_maps.distcenters = smp.dist_centers
        env_maps.pop_nan_positions()
        smp.memo[key] = env_maps
    return _detach(smp.memo[key])


def get_reliability(session: B2Session, params: Union[SpkmapParams, dict, None] = None) -> Reliability:
    """Spatial reliability per environment and ROI."""
    return get_smp(session, params).get_reliability()


def get_fraction_active(session: B2Session, params: Union[SpkmapParams, dict, None] = None) -> np.ndarray:
    """
    Fraction-active per environment and ROI, shape ``(envs, rois)``.

    Computed on the same NaN-popped maps :func:`get_env_maps` returns, so it matches what the
    figure functions computed inline.
    """
    smp = get_smp(session, params)
    key = ("fraction_active",)
    if key not in smp.memo:
        env_maps = get_env_maps(session, params)
        smp.memo[key] = np.stack([FractionActive.compute(spkmap, 2, 1) for spkmap in env_maps.spkmap])
    return smp.memo[key]


def get_prediction_extras(session: B2Session, params: Union[SpkmapParams, dict, None] = None) -> dict[str, np.ndarray]:
    """
    Frame bookkeeping that goes with a prediction: position bin, environment index, ``idx_valid``.

    Always comes from the ``spkmap`` pass, which is the frame grid both prediction modes are
    indexed on.
    """
    return get_smp(session, params).get_placefield_prediction()[1]


def get_prediction(
    session: B2Session,
    prediction_from: str = "spkmap",
    params: Union[SpkmapParams, dict, None] = None,
) -> np.ndarray:
    """
    Full-frame place-field prediction, shape ``(frames, rois)``.

    Parameters
    ----------
    session : B2Session
    prediction_from : {"spkmap", "placefield"}
        See :data:`PREDICTION_MODES`. Both are cached independently.
    params : SpkmapParams, dict, or None

    Returns
    -------
    np.ndarray
        Shared read-only array. NaN on frames where no prediction is possible; pair it with
        ``get_prediction_extras(...)["idx_valid"]``.
    """
    smp = get_smp(session, params)
    if prediction_from == "spkmap":
        return smp.get_placefield_prediction()[0]
    if prediction_from != "placefield":
        raise ValueError(f"prediction_from must be one of {PREDICTION_MODES}, got {prediction_from!r}")

    key = ("prediction", "placefield")
    if key not in smp.memo:
        frame_behavior = get_frame_behavior(session)
        placefield = get_placefield(
            get_spks(session, params),
            frame_behavior,
            smp.dist_edges,
            smp.params.speed_threshold,
            average=True,
            smooth_width=1.0,
        )
        smp.memo[key] = get_placefield_prediction(placefield, frame_behavior)[0]
    return smp.memo[key]


def get_valid_activity(
    session: B2Session,
    idx_rois: Union[np.ndarray, None] = None,
    params: Union[SpkmapParams, dict, None] = None,
) -> np.ndarray:
    """
    Activity on the predictable frames, transposed to raster orientation.

    Parameters
    ----------
    session : B2Session
    idx_rois : np.ndarray or None
        ROI subset, in session-filtered ROI coordinates. None keeps every ROI.
    params : SpkmapParams, dict, or None

    Returns
    -------
    np.ndarray
        ``(rois, valid_frames)``, restricted to ``get_prediction_extras(...)["idx_valid"]``.
    """
    spks = get_spks(session, params)
    spks_valid = spks[get_prediction_extras(session, params)["idx_valid"]]
    if idx_rois is not None:
        spks_valid = spks_valid[:, idx_rois]
    return spks_valid.T


def get_roi_sort(
    session: B2Session,
    method: str = "environment",
    idx_rois: Union[np.ndarray, None] = None,
    params: Union[SpkmapParams, dict, None] = None,
) -> np.ndarray:
    """
    Ordering of a subset of a session's ROIs, under one sort method.

    Both sorts are deterministic given the session and the subset, and ``rastermap`` costs
    tens of seconds, so they are cached here rather than on a viewer that gets rebuilt every
    time a figure function runs. The Rastermap fit uses :func:`get_valid_activity`, so the
    ordering does not depend on which prediction a figure happens to be showing.

    Parameters
    ----------
    session : B2Session
    method : {"environment", "rastermap"}
        See :data:`SORT_METHODS`.
    idx_rois : np.ndarray or None
        ROI subset to order, in session-filtered ROI coordinates. None orders every ROI.
    params : SpkmapParams, dict, or None

    Returns
    -------
    np.ndarray
        Permutation of ``range(len(idx_rois))``.
    """
    smp = get_smp(session, params)
    if idx_rois is None:
        idx_rois = np.arange(get_spks(session, params).shape[1])
    idx_rois = np.asarray(idx_rois)

    key = ("roi_sort", method, idx_rois.tobytes())
    if key not in smp.memo:
        if method == "environment":
            smp.memo[key] = sort_by_preferred_environment(smp, idx_rois=idx_rois)
        elif method == "rastermap":
            smp.memo[key] = Rastermap().fit(get_valid_activity(session, idx_rois, params)).isort
        else:
            raise ValueError(f"method must be one of {SORT_METHODS}, got {method!r}")
    return smp.memo[key]


def preload(
    session: B2Session,
    params: Union[SpkmapParams, dict, None] = None,
    prediction_modes: tuple[str, ...] = PREDICTION_MODES,
    verbose: bool = True,
) -> CachedSpkmapProcessor:
    """
    Compute and cache everything the figure1 functions need for a session.

    Run this in its own notebook cell. Every later figure call on the same session,
    ``spks_type`` and params then reads from memory.

    Parameters
    ----------
    session : B2Session
    params : SpkmapParams, dict, or None
    prediction_modes : tuple of str
        Which prediction modes to build. Both by default; drop ``"placefield"`` if no figure
        in the notebook uses it, since it is a second full pass over the activity.
    verbose : bool
        Print each step's elapsed time and the cache size when done.

    Returns
    -------
    CachedSpkmapProcessor
        The shared processor, so it can be passed straight to ``sort_by_preferred_environment``
        and friends.
    """
    smp = get_smp(session, params)

    steps: list[tuple[str, Any]] = [
        ("activity", lambda: get_spks(session, params)),
        ("env maps", lambda: get_env_maps(session, params)),
        ("reliability", lambda: get_reliability(session, params)),
        ("fraction active", lambda: get_fraction_active(session, params)),
    ]
    for mode in prediction_modes:
        steps.append((f"prediction ({mode})", lambda mode=mode: get_prediction(session, mode, params)))

    if verbose:
        print(f"Preloading {session.session_print()} [{session.spks_type}]")
    for name, step in steps:
        start = time.perf_counter()
        step()
        if verbose:
            print(f"  {name:<24s} {time.perf_counter() - start:6.2f} s")
    if verbose:
        print(f"  {'cached':<24s} {smp.nbytes() / 1e9:6.2f} GB")
    return smp


def clear(session: Union[B2Session, None] = None, params: Union[SpkmapParams, dict, None] = None) -> None:
    """
    Drop cached products so they are rebuilt on next use.

    Parameters
    ----------
    session : B2Session or None
        Session to drop. None drops every session. When given without ``params``, every
        params/``spks_type`` variant of that session is dropped.
    params : SpkmapParams, dict, or None
        Restrict the drop to one parameter set.
    """
    if session is None:
        _PROCESSORS.clear()
        return
    if params is not None:
        _PROCESSORS.pop(_key(session, _resolve_params(params)), None)
        return
    for key in [k for k in _PROCESSORS if k[0] == session.session_name]:
        del _PROCESSORS[key]


def cache_info() -> None:
    """Print what is currently cached, with the size of each entry."""
    if not _PROCESSORS:
        print("session_cache is empty")
        return
    for (session_name, spks_type, _), smp in _PROCESSORS.items():
        print(f"{'/'.join(session_name)} [{spks_type}] - {smp.nbytes() / 1e9:.2f} GB")
        for key, value in smp.memo.items():
            # Some keys carry raw index bytes to identify an ROI subset; those aren't printable.
            qualifiers = [str(part) for part in key[1:] if not isinstance(part, bytes)]
            label = f"{key[0]}[{', '.join(qualifiers)}]" if qualifiers else key[0]
            print(f"    {label:<44s} {_nbytes(value) / 1e9:6.3f} GB")
