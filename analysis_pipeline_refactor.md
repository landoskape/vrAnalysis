# Analysis Pipeline Refactor Design

## Problem Diagnosis

The current system conflates three concerns that should be separate:

1. **What to compute** — the parameter space you want to cover
2. **How to compute it** — the pipeline logic
3. **Whether it's been computed** — the cache/registry

Symptoms: filenames encode parameters (breaks on addition), `required_keys` guards
validate result shape (breaks on new fields), session loops contain both orchestration
and processing logic, and there's no first-class concept of a "data config" separate
from an "analysis config."

---

## Core Architecture

```
AnalysisConfig      →  abstract base: keying, versioning, variation generation, processing
DataConfig          →  how to get data for a session (wraps RegistryParameters)
AnalysisPlan        →  generic orchestrator: config space × store → dispatched jobs
ResultsStore        →  SQLite-backed, content-addressed by (session, data_cfg, analysis_cfg)
```

The key shift from the old design: **each analysis config class carries its own
`process()` method**. `AnalysisPlan` dispatches to `acfg.process()`.
Adding a new analysis means writing a new config class — the plan and store are
never touched.

---

## Layer 1: AnalysisConfigBase

The abstract base class defines the full contract every analysis config must satisfy.
It provides concrete implementations where the logic is universal, and enforces
abstract methods where each analysis must supply its own.

```python
# base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from itertools import product
from typing import ClassVar
import hashlib, json


@dataclass(frozen=True)
class AnalysisConfigBase(ABC):
    """Abstract base for all analysis configs.

    Concrete subclasses must:
      - Define `display_name` as a ClassVar[str]
      - Implement `_param_grid()` returning {field: [options]}
      - Implement `process()` containing the analysis logic
      - Optionally override `summary()` for richer human-readable labels
      - Optionally override `validate()` for config-level assertions
      - Optionally override `generate_variations()` for non-product spaces
    """

    schema_version: str = "v1"

    # Class-level metadata — not an instance field, excluded from key() automatically
    display_name: ClassVar[str] = "base"

    def __post_init__(self):
        self.validate()

    # ------------------------------------------------------------------ #
    # Keying — do not override in subclasses                              #
    # ------------------------------------------------------------------ #

    def key(self) -> str:
        """Stable content-addressed key.

        Parameter changes → new key → old results untouched, not broken.
        Intentionally not abstract — the hash IS the identity.
        """
        canon = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(canon.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ #
    # Human-readable label                                                #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        """Human-readable label for display/logging. NOT used for cache lookup.

        Default: display_name + schema_version. Subclasses should override
        to include parameter detail — this string appears in summary_table()
        and you will read it often.
        """
        return f"{self.display_name}_{self.schema_version}"

    # ------------------------------------------------------------------ #
    # Validation hook                                                     #
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Assert config-level invariants. Called automatically in __post_init__.

        Default is a no-op. Override to catch contradictory parameter
        combinations early rather than at process time.

        Raises
        ------
        ValueError
            If the config is self-contradictory.
        """
        pass

    # ------------------------------------------------------------------ #
    # Variation generation                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    @abstractmethod
    def _param_grid() -> dict:
        """Return {field_name: [options]} for all swept parameters.

        Single source of truth for the parameter space. Edit here to
        add or change options — generate_variations() picks it up automatically.
        Fields not listed here take their dataclass default in all variations.
        """
        ...

    @classmethod
    def generate_variations(cls) -> list[AnalysisConfigBase]:
        """Generate all configs in the current parameter grid.

        Default: full Cartesian product of _param_grid(). Override if you
        need a constrained space (e.g. A x B but only C=True when D=False).
        """
        grid = cls._param_grid()
        keys, option_lists = zip(*grid.items())
        return [
            cls(**dict(zip(keys, combo)))
            for combo in product(*option_lists)
        ]

    # ------------------------------------------------------------------ #
    # Processing — the analysis logic                                     #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def process(
        self,
        session,
        data_cfg: DataConfig,
    ) -> dict:
        """Run the analysis for one (session, data_cfg, analysis_cfg) unit.

        Must be a pure function: no side effects, no persistent state.
        ResultsStore.put() is handled by AnalysisPlan — do not call it here.

        Parameters
        ----------
        session :
            Session object to process.
        data_cfg : DataConfig
            Describes how to load and split data for this session.

        Returns
        -------
        dict
            Result dict. Shape is analysis-specific — the store treats it
            as an opaque blob. No required keys, no validation needed.
        """
        ...


        ### NOTE: I changed this to be a normal method, not a classmethod. Other parts of this refactor document might be out of date.
```

### CVPCAConfig — example concrete implementation

```python
# cvpca_config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from .base import AnalysisConfigBase
from .data_config import DataConfig


@dataclass(frozen=True)
class CVPCAConfig(AnalysisConfigBase):

    display_name: ClassVar[str] = "cvpca"

    center: bool = True
    normalize: bool = True
    use_fast_sampling: bool = True
    reliability_threshold: float | None = None
    fraction_active_threshold: float | None = None
    fixed_smooth_widths: tuple[float, ...] = (3.0,)
    num_bins: int = 100

    @staticmethod
    def _param_grid() -> dict:
        return {
            "center":                    [True, False],
            "normalize":                 [True, False],
            "use_fast_sampling":         [True, False],
            "reliability_threshold":     [None, 0.2],
            "fraction_active_threshold": [None, 0.05],
            "fixed_smooth_widths":       [(3.0,), (3.0, 6.0)],
            "num_bins":                  [100],
        }

    def summary(self) -> str:
        parts = [self.display_name, f"bins{self.num_bins}"]
        if not self.center:                parts.append("nocenter")
        if not self.normalize:             parts.append("nonorm")
        if self.use_fast_sampling:         parts.append("fast")
        if self.reliability_threshold:     parts.append(f"rel{self.reliability_threshold}")
        if self.fraction_active_threshold: parts.append(f"frac{self.fraction_active_threshold}")
        parts.append(f"sw{'_'.join(str(w) for w in self.fixed_smooth_widths)}")
        parts.append(self.schema_version)
        return "_".join(parts)

    def validate(self) -> None:
        if self.reliability_threshold is not None and not self.normalize:
            raise ValueError("reliability_threshold requires normalize=True")

    def process(self, session, data_cfg: DataConfig) -> dict:
        """CVPCA analysis for one session."""
        registry = PopulationRegistry(registry_params=data_cfg.to_registry_params())
        population, frame_behavior = registry.get_population(
            session, spks_type=data_cfg.spks_type
        )
        # ... analysis logic parameterized by analysis_cfg fields ...
        return result
```

---

## Layer 2: DataConfig (wraps RegistryParameters)

No abstract base needed yet — there is currently one data pipeline. Add a
`DataConfigBase` later if you need fundamentally different data loading strategies.

`DataConfig` needs `key()` and `summary()` but not the full `AnalysisConfigBase`
contract. Options: extract a small `KeyableMixin` with just those two methods, or
simply define them directly on `DataConfig`. Either is fine at this scale — do not
over-engineer it.

```python
# data_config.py
from dataclasses import dataclass, asdict
import hashlib, json


@dataclass(frozen=True)
class DataConfig:
    """Describes how to build a Population for a session.

    Wraps RegistryParameters so that changing data-pipeline parameters
    (speed threshold, split strategy, etc.) automatically invalidates
    dependent analysis results via key change — no manual cache clearing needed.
    """
    speed_threshold: float = 1.0
    time_split_groups: int = 4
    time_split_relative_size: tuple[int, ...] = (4, 4, 1, 1)
    time_split_chunks_per_group: int = 10
    time_split_num_buffer: int = 3
    cell_split_force_even: bool = False
    spks_type: str = "oasis"

    def key(self) -> str:
        canon = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(canon.encode()).hexdigest()[:16]

    def summary(self) -> str:
        return f"spd{self.speed_threshold}_{self.spks_type}"

    def to_registry_params(self) -> RegistryParameters:
        return RegistryParameters(
            speed_threshold=self.speed_threshold,
            time_split_groups=self.time_split_groups,
            time_split_relative_size=self.time_split_relative_size,
            time_split_chunks_per_group=self.time_split_chunks_per_group,
            time_split_num_buffer=self.time_split_num_buffer,
            cell_split_force_even=self.cell_split_force_even,
        )
```

---

## Layer 3: AnalysisPlan (generic orchestrator)

`AnalysisPlan` is now fully generic — it dispatches to `acfg.process()` and
requires no `process_fn` argument. Adding a new analysis means writing a new config
class; the plan is never modified.

Multiple analysis types can be mixed in a single plan by concatenating their
`generate_variations()` outputs. Each config type dispatches to its own `process()`.

```python
# plan.py
from dataclasses import dataclass
import joblib
from tqdm import tqdm
from .base import AnalysisConfigBase
from .data_config import DataConfig
from .store import ResultsStore


@dataclass
class AnalysisPlan:
    """Generic orchestrator for (session x data_config x analysis_config) sweeps.

    Conceptually similar to Snakemake: determines what jobs need to be done,
    skips completed ones, dispatches the rest. Job graph is a flat product
    (no DAG) — sufficient for all current analyses.

    Parameters
    ----------
    analysis_configs : list[AnalysisConfigBase]
        All analysis configs to sweep over. Can mix config types —
        e.g. CVPCAConfig.generate_variations() + OtherConfig.generate_variations().
        Each config carries its own process() method, so dispatch is automatic.
    data_configs : list[DataConfig]
        Data configs to sweep over. Usually just [DataConfig()] unless
        explicitly sweeping data pipeline parameters.

    Usage
    -----
        # Single analysis
        plan = AnalysisPlan(
            analysis_configs=CVPCAConfig.generate_variations(),
            data_configs=[DataConfig()],
        )
        plan.analyze(sessions, store, n_jobs=4)

        # Multiple analyses in one plan, sharing the store
        plan = AnalysisPlan(
            analysis_configs=(
                CVPCAConfig.generate_variations() +
                OtherAnalysisConfig.generate_variations()
            ),
            data_configs=[DataConfig()],
        )
        plan.analyze(sessions, store, n_jobs=4)
    """
    analysis_configs: list[AnalysisConfigBase]
    data_configs:     list[DataConfig]

    def analyze(
        self,
        sessions: list,
        store: ResultsStore,
        n_jobs: int = 4,
        force_remake: bool = False,
        snapshot_codebase: bool = True,
    ):
        """Run all missing (session, data_cfg, analysis_cfg) jobs.

        Parameters
        ----------
        sessions : list
            All sessions to process.
        store : ResultsStore
            Results store to check for completed work and write results to.
        n_jobs : int
            Parallel workers passed to joblib. Use 1 for debugging.
        force_remake : bool
            Recompute even if result already exists in store.
        snapshot_codebase : bool
            Save a freezedry snapshot before processing begins. Disable
            during iterative development; enable for production runs.
        """
        if snapshot_codebase:
            store.snapshot_codebase()

        work = self._collect_work(sessions, store, force_remake)
        total = len(sessions) * len(self.data_configs) * len(self.analysis_configs)
        print(f"{len(work)} jobs to run "
              f"({total - len(work)} already complete, {total} total)")

        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(self._run_one)(session, dcfg, acfg, store)
            for session, dcfg, acfg in tqdm(work, desc="Processing")
        )

    def _collect_work(self, sessions, store, force_remake):
        return [
            (session, dcfg, acfg)
            for session in sessions
            for dcfg   in self.data_configs
            for acfg   in self.analysis_configs
            if force_remake or not store.has(session.session_id, dcfg, acfg)
        ]

    def _run_one(self, session, dcfg, acfg, store):
        try:
            # Dispatch via config type — no process_fn needed
            result = acfg.process(session, dcfg)
            store.put(session.session_id, dcfg, acfg, result)
        except Exception as e:
            print(
                f"Error [{session.session_id} | "
                f"{dcfg.summary()} | {acfg.summary()}]: {e}"
            )
        finally:
            session.clear_cache()
```

---

## Layer 4: ResultsStore (SQLite-backed, content-addressed)

Store key is `hash(data_cfg) + hash(analysis_cfg)`. Changing either config produces
a new key — old results untouched. Human-readable metadata stored separately for
`summary_table()`. WAL mode enabled for safe concurrent writers (SGE array jobs).

```python
# store.py
import sqlite3, pickle
from pathlib import Path
from contextlib import contextmanager


class ResultsStore:
    """SQLite-backed result cache keyed by (session_id, data_cfg, analysis_cfg).

    Changing either config automatically produces new keys — no manual cache
    invalidation for parameter changes. Use schema_version + invalidate() for
    implementation-level changes (see Layer 5).
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent writers
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    session_id       TEXT NOT NULL,
                    data_key         TEXT NOT NULL,
                    analysis_key     TEXT NOT NULL,
                    data_summary     TEXT,
                    analysis_summary TEXT,
                    analysis_type    TEXT,            -- display_name, for filtering
                    schema_version   TEXT,
                    result_blob      BLOB NOT NULL,
                    computed_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, data_key, analysis_key)
                )
            """)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _keys(self, data_cfg, analysis_cfg) -> tuple[str, str]:
        return data_cfg.key(), analysis_cfg.key()

    def has(self, session_id: str, data_cfg, analysis_cfg) -> bool:
        dk, ak = self._keys(data_cfg, analysis_cfg)
        with self._conn() as conn:
            return conn.execute(
                "SELECT 1 FROM results "
                "WHERE session_id=? AND data_key=? AND analysis_key=?",
                (session_id, dk, ak),
            ).fetchone() is not None

    def put(self, session_id: str, data_cfg, analysis_cfg, result: dict):
        dk, ak = self._keys(data_cfg, analysis_cfg)
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO results
                    (session_id, data_key, analysis_key,
                     data_summary, analysis_summary, analysis_type,
                     schema_version, result_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, dk, ak,
                data_cfg.summary(),
                analysis_cfg.summary(),
                getattr(analysis_cfg, "display_name", None),
                getattr(analysis_cfg, "schema_version", None),
                pickle.dumps(result),
            ))

    def get(self, session_id: str, data_cfg, analysis_cfg) -> dict | None:
        dk, ak = self._keys(data_cfg, analysis_cfg)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT result_blob FROM results "
                "WHERE session_id=? AND data_key=? AND analysis_key=?",
                (session_id, dk, ak),
            ).fetchone()
        return pickle.loads(row[0]) if row else None

    def invalidate(
        self,
        schema_version: str | None = None,
        data_cfg=None,
        analysis_cfg=None,
        analysis_type: str | None = None,
    ):
        """Delete entries matching any combination of filters.

        At least one filter is required — use invalidate_all() to clear everything.

        Parameters
        ----------
        schema_version : str | None
            Clear all entries with this schema version. Primary mechanism for
            implementation-level invalidation (see Layer 5).
        data_cfg : DataConfig | None
            Clear all entries produced with this specific data config.
        analysis_cfg : AnalysisConfigBase | None
            Clear all entries produced with this specific analysis config.
        analysis_type : str | None
            Clear all entries with this display_name, regardless of version.
            Useful for wiping all results of one analysis type cleanly.

        Examples
        --------
            store.invalidate(schema_version="v1")
            store.invalidate(analysis_type="cvpca", schema_version="v1")
            store.invalidate(data_cfg=old_dcfg)
        """
        clauses, params = [], []
        if schema_version is not None:
            clauses.append("schema_version = ?"); params.append(schema_version)
        if data_cfg is not None:
            clauses.append("data_key = ?");       params.append(data_cfg.key())
        if analysis_cfg is not None:
            clauses.append("analysis_key = ?");   params.append(analysis_cfg.key())
        if analysis_type is not None:
            clauses.append("analysis_type = ?");  params.append(analysis_type)
        if not clauses:
            raise ValueError(
                "invalidate() requires at least one filter. "
                "To clear everything use invalidate_all()."
            )
        with self._conn() as conn:
            n = conn.execute(
                f"DELETE FROM results WHERE {' AND '.join(clauses)}", params
            ).rowcount
        print(f"Invalidated {n} entries.")

    def invalidate_all(self):
        """Nuclear option — clears the entire store."""
        with self._conn() as conn:
            n = conn.execute("DELETE FROM results").rowcount
        print(f"Cleared all {n} entries.")

    def summary_table(self, as_dataframe: bool = False):
        """Human-readable view of what has been computed."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT session_id, analysis_type, data_summary,
                       analysis_summary, schema_version, computed_at
                FROM results ORDER BY computed_at DESC
            """).fetchall()
        records = [
            {"session": r[0], "analysis": r[1], "data": r[2],
             "config": r[3], "version": r[4], "computed_at": r[5]}
            for r in rows
        ]
        if as_dataframe:
            import pandas as pd
            return pd.DataFrame(records)
        return records

    def coverage(self, sessions, data_configs, analysis_configs) -> float:
        """Fraction of the target space that has been computed."""
        total = len(sessions) * len(data_configs) * len(analysis_configs)
        if total == 0:
            return 0.0
        done = sum(
            self.has(s.session_id, dc, ac)
            for s in sessions
            for dc in data_configs
            for ac in analysis_configs
        )
        return done / total
```

---

## Layer 5: Schema Versioning (implementation-level invalidation)

Content-addressed keys handle parameter changes automatically. But sometimes the
*implementation* changes without the *parameters* changing — e.g. switching from
max-normalization to std-normalization without adding a `norm_method` parameter.

`schema_version: str = "v1"` lives on `AnalysisConfigBase` and participates in
`key()`. Bumping it produces new store keys — old results are automatically bypassed
and accumulate as dead weight until explicitly cleaned with `invalidate()`.

**Workflow for a meaningful implementation change:**

```python
# 1. Make your implementation change in CVPCAConfig.process()

# 2. Bump schema_version default in CVPCAConfig:
#    schema_version: str = "v2"

# 3. Re-run — only v2 work is computed, v1 results silently bypassed
plan.analyze(sessions, store, n_jobs=4)

# 4. Clean up v1 when confident v2 is correct:
store.invalidate(schema_version="v1", analysis_type="cvpca")
```

**What NOT to do:** Do not add `norm_method: str = "max"` retroactively to parametrize
an implicit implementation detail. Old entries would be keyed as `norm_method="max"`
regardless of how they were actually computed. Bump `schema_version` instead — be
explicit about the discontinuity.

---

## Layer 6: Codebase Snapshots with freezedry

> **Note**: `freezedry` is a custom package for snapshotting codebase state.
> The integration below assumes an API like `freezedry.snapshot(source_dirs, output_path)`.
> Adjust the call signature to match the actual API when porting.

`schema_version` tells you *that* results were computed with `"v2"`, but not *what
v2 actually was*. A codebase snapshot saved alongside the store gives a precise,
reconstructable record of the code that produced each batch of results.

Add these methods to `ResultsStore`:

```python
def snapshot_codebase(self, source_dirs: list[Path] | None = None):
    """Save a freezedry snapshot of the codebase next to the results DB.

    Snapshots accumulate by timestamp — you get a log of every run, not
    just the latest state. Only call from the submission node, not from
    individual SGE array tasks.

    Parameters
    ----------
    source_dirs : list[Path] | None
        Directories to snapshot. TODO: set default for your repo layout.
    """
    import freezedry
    from datetime import datetime

    snapshot_dir = self.db_path.parent / "codebase_snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = snapshot_dir / f"snapshot_{timestamp}"
    source_dirs = source_dirs or [Path(__file__).parent.parent]  # TODO: adjust
    freezedry.snapshot(source_dirs, out_path)
    print(f"Codebase snapshot saved: {out_path}")

def list_snapshots(self) -> list[Path]:
    """List all codebase snapshots associated with this store."""
    snapshot_dir = self.db_path.parent / "codebase_snapshots"
    return sorted(snapshot_dir.iterdir()) if snapshot_dir.exists() else []
```

Store directory is fully self-contained:

```
measure_cvpca/
    results.db
    codebase_snapshots/
        snapshot_20250309_143022/
        snapshot_20250312_091155/
```

---

## Layer 7: Putting it together

```python
# workflows/measure_cvpca.py  (refactored)
from dimensionality_manuscript.pipeline import (
    CVPCAConfig, DataConfig, AnalysisPlan, ResultsStore,
)
from vrAnalysis.database import get_database

sessiondb = get_database("vrSessions")
store     = ResultsStore(registry_paths.measure_cvpca_path / "results.db")

if __name__ == "__main__":
    sessions = list(sessiondb.iter_sessions(imaging=True))

    # Define the parameter space
    analysis_configs = CVPCAConfig.generate_variations()
    data_configs     = [DataConfig()]  # single default; expand later if needed

    # Build and run
    plan = AnalysisPlan(
        analysis_configs=analysis_configs,
        data_configs=data_configs,
    )
    plan.analyze(sessions, store, n_jobs=4)

    # Inspect
    print(f"Coverage: {store.coverage(sessions, data_configs, analysis_configs):.1%}")
    df = store.summary_table(as_dataframe=True)
    print(df.groupby(["analysis", "version"]).size())
    print(f"Snapshots: {store.list_snapshots()}")
```

To run multiple analyses, mix config types in one plan:

```python
    plan = AnalysisPlan(
        analysis_configs=(
            CVPCAConfig.generate_variations() +
            OtherAnalysisConfig.generate_variations()
        ),
        data_configs=[DataConfig()],
    )
    plan.analyze(sessions, store, n_jobs=4)
```

---

## Migration Notes

- **Existing `.pkl` files**: Write a one-off migration script that loads each `.pkl`,
  reconstructs the `(data_cfg, analysis_cfg)` that would have produced it from the
  filename, and calls `store.put(session_id, data_cfg, analysis_cfg, result)`.
  Otherwise just re-run — the store fills in what is missing.

- **PopulationRegistry**: Its disk cache (`.joblib` per session) remains unchanged.
  `ResultsStore` sits above it, caching analysis outputs. Two caches, two levels of
  the computation graph — do not conflate them.

- **`DataConfig.generate_variations()`**: Start with `[DataConfig()]`. Add
  `generate_variations()` when you actually have a reason to sweep data parameters.

- **Multiple analyses, one store**: All analyses share a single `results.db`. The
  analysis key encodes the config type implicitly via its hash — you do not need
  separate tables per analysis type.

---

## Design Principles

1. **Config keys are stable identities.** Never derive cache validity from filenames
   or field names — only from the hash of the full config that produced the result.

2. **`schema_version` is the escape hatch for implementation changes.** It is manual
   and explicit, because only a human can decide whether a code change is semantically
   meaningful. Bumping it orphans old results until you explicitly `invalidate()`.

3. **`_param_grid()` is the single source of truth for parameter options.** `summary()`
   is for humans. `key()` is for the database. `generate_variations()` is plumbing.

4. **`AnalysisPlan` knows nothing about any specific analysis.** New analyses bring
   their own config class with `process()`. The plan, store, and orchestration are
   never touched.

5. **The store is append-only in normal operation.** Invalidation is a deliberate,
   explicit act — not something that happens automatically during a run.

6. **`process()` is a pure function.** `(session, data_cfg, analysis_cfg) -> dict`.
   No side effects, no persistent state. The plan handles all persistence.

7. **`DataConfig` has no base class yet.** Do not add one speculatively. If you ever
   need to sweep fundamentally different data loading strategies, introduce
   `DataConfigBase` at that point.
