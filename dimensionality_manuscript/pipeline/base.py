"""Abstract base class for analysis configurations."""

from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from dataclasses import asdict, dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from syd import Viewer
    from ..registry import PopulationRegistry
    from vrAnalysis.sessions import B2Session
    from .aggregate import ResultsAggregator
    from ..configs.data_config import DataConfig


@dataclass(frozen=True)
class AnalysisConfigBase:
    """Abstract frozen dataclass for analysis configurations.

    All analysis configs inherit from this. Provides content-addressed
    keying, parameter grid generation, and an abstract process method.
    """

    schema_version: str = "v1"
    data_config_name: str = "default"
    display_name: ClassVar[str]
    _result_handling: ClassVar[dict[str, str]] = {}
    """Per-key aggregation strategy.  Keys absent from this dict use ``"pad"``.

    Supported values
    ----------------
    ``"pad"`` (default)
        NaN-pad all results to the max observed shape.
    ``"ragged"``
        Store as an object array of shape ``(n_sess, *param_dims)``.
        Each cell holds the raw ndarray.  Use when dimensions carry
        semantic meaning that makes uniform padding nonsensical
        (e.g. position × position cross matrices that vary by environment).
    ``"skip"``
        Exclude this key from aggregation entirely.
    """

    def __post_init__(self):
        from ..configs.data_config import _NAMED_CONFIGS

        if self.data_config_name not in _NAMED_CONFIGS:
            available = ", ".join(sorted(_NAMED_CONFIGS))
            raise ValueError(f"Unknown data_config_name {self.data_config_name!r}. Available: {available}")
        self.validate()

    @property
    def data_config(self) -> DataConfig:
        """DataConfig corresponding to this config's data_config_name."""
        from ..configs.data_config import get_data_config

        return get_data_config(self.data_config_name)

    def key(self) -> str:
        """SHA256 of serialized config, truncated to 16 chars."""
        serialized = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def summary(self) -> str:
        """Human-readable summary string."""
        return f"{self.display_name}_{self.schema_version}"

    def validate(self):
        """Override to check invariants. Called in __post_init__."""

    @staticmethod
    @abstractmethod
    def _param_grid() -> dict:
        """Return ``{field: [options]}`` for Cartesian product generation."""

    @classmethod
    def generate_variations(cls, schema_version: str | None = None) -> list[AnalysisConfigBase]:
        """Cartesian product of ``_param_grid()``. Override to filter."""
        grid = cls._param_grid()
        fields = list(grid.keys())
        values = list(grid.values())
        if schema_version is not None:
            return [cls(**dict(zip(fields, combo)), schema_version=schema_version) for combo in product(*values)]
        return [cls(**dict(zip(fields, combo))) for combo in product(*values)]

    @classmethod
    def generate_variations_matching(cls, param_filters: dict[str, Any]) -> list[AnalysisConfigBase]:
        """Cartesian product of ``_param_grid()`` with fixed parameter values.

        Fields listed in ``param_filters`` are held constant; all other grid
        axes vary as in :meth:`generate_variations`. Fields not in
        ``_param_grid()`` (e.g. ``schema_version``) are passed through as
        fixed constructor kwargs.

        Parameters
        ----------
        param_filters : dict
            ``{field_name: value}`` constraints. Every returned config has
            ``getattr(cfg, name) == value`` for each entry.

        Returns
        -------
        list of AnalysisConfigBase
        """
        from dataclasses import fields

        if not param_filters:
            return cls.generate_variations()

        field_names = {f.name for f in fields(cls)}
        unknown = set(param_filters) - field_names
        if unknown:
            raise ValueError(f"Unknown field(s) for {cls.__name__}: {sorted(unknown)}")

        grid = dict(cls._param_grid())
        fixed: dict[str, Any] = {}
        for name, value in param_filters.items():
            if name in grid:
                grid[name] = [value]
            else:
                fixed[name] = value

        grid_fields = list(grid.keys())
        grid_values = list(grid.values())
        return [cls(**{**dict(zip(grid_fields, combo)), **fixed}) for combo in product(*grid_values)]

    @classmethod
    def from_key(cls, key: str) -> AnalysisConfigBase:
        """Look up a config by its content-hash key.

        Generates all variations and returns the one whose ``key()``
        matches. Raises ``KeyError`` if no match is found.

        Parameters
        ----------
        key : str
            The 16-char hex key to look up.
        """
        for cfg in cls.generate_variations():
            if cfg.key() == key:
                return cfg
        raise KeyError(f"No {cls.__name__} variation matches key {key!r}")

    def get_result(
        self,
        store,
        row: dict,
    ) -> dict | None:
        """Load the result for a summary-table row.

        Loads the blob from the store via ``get_by_uid``. Self-caching
        configs (where ``process`` returns None) need to handle getting
        results themselves!

        Parameters
        ----------
        store : ResultsStore
            The results store containing this row.
        row : dict
            A single row from ``ResultsStore.summary_table()``.  Always
            contains at least ``result_uid`` and ``session_id``.

        Returns
        -------
        dict or None
            The result dict, or None if not available.
        """
        return store.get_by_uid(row["result_uid"])

    def build_syd(
        self,
        viewer: Viewer,
        results: ResultsAggregator,
        include_get_results: bool = True,
        squeeze_ones: bool = True,
        no_filter: list[str] | None = None,
    ):
        """Add param-grid selections to a syd Viewer and attach a result getter.

        Adds one ``add_selection`` per ``_param_grid()`` axis (defaulting to this
        config's own values), plus ``view_by`` / ``session_id`` / ``mouse``
        selectors. Attaches ``viewer.get_result(state) -> dict[str, np.ndarray]``
        as a convenience method on the viewer.

        Parameters
        ----------
        viewer : syd.Viewer
            Viewer to configure in-place.
        results : ResultsAggregator
            Loaded results.
        include_get_results : bool
            Whether to attach the ``get_result`` method to the viewer.  Set to False if
            you want to implement your own result-getting logic (e.g. for self-caching configs).
        squeeze_ones : bool
            Whether to apply ``squeeze()`` to the returned arrays.  Set to False
            if you want to preserve singleton dimensions for easier indexing.
        no_filter : list[str] or None
            List of state keys to exclude from filtering by session/mouse.  By default, all keys are
            filtered, meaning that you can't compare across config combinations within a get_result call.

        Returns
        -------
        viewer
            The same viewer, for ``self = cfg.build_syd(self)`` syntax.

        Examples
        --------
        .. code-block:: python

            class CVPCAViewer(Viewer):
                def __init__(self, cfg, results):
                    self = cfg.build_syd(self, results)

                def plot(self, state):
                    data = self.get_result(state)
                    fig, ax = plt.subplots()
                    ax.plot(data["reg_covariances"])
                    return fig
        """
        unique_mice = list(dict.fromkeys(results.mouse_names.tolist()))

        param_axes = self._param_grid()
        filter_by = []
        for name, options in param_axes.items():
            if no_filter is None or name not in no_filter:
                viewer.add_selection(name, options=list(options), value=getattr(self, name))
                filter_by.append(name)

        viewer.add_selection("view_by", options=["session", "mouse_average"], value="mouse_average")
        viewer.add_selection("session_id", options=results.session_ids, value=results.session_ids[0])
        viewer.add_selection("mouse", options=unique_mice, value=unique_mice[0] if unique_mice else "")
        viewer.add_boolean("filter_by_ses_or_mouse", value=False)
        if include_get_results:

            def get_result(state: dict, load_ragged: bool = False) -> tuple[dict, list[str]]:
                param_kwargs = {k: state[k] for k in param_axes if k in filter_by}
                view_by = state["view_by"]
                sliced, axes_names = results.sel(
                    load_ragged=load_ragged,
                    squeeze_ones=squeeze_ones,
                    return_param_sizes=True,
                    avg_by_mouse=(view_by == "mouse_average"),
                    **param_kwargs,
                )
                if state["filter_by_ses_or_mouse"]:
                    if view_by == "mouse_average":
                        idx = results.unique_mice.index(state["mouse"])
                    else:
                        idx = results.session_ids.index(state["session_id"])
                    sliced = {k: v[idx] for k, v in sliced.items()}
                return sliced, axes_names

            viewer.get_result = get_result
        return viewer

    @abstractmethod
    def process(self, session: B2Session, registry: PopulationRegistry) -> dict | None:
        """Run analysis on a session.

        Parameters
        ----------
        session : B2Session
            Session to process.
        registry : PopulationRegistry
            Population registry for getting population data.

        Returns
        -------
        dict or None
            Result dict stored by ResultsStore, or None as a completion
            marker for self-caching workflows.
        """
