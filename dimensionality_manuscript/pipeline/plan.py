"""AnalysisPlan — orchestrator for running analysis configs across sessions."""

from __future__ import annotations

from dataclasses import dataclass, field

import gc
import traceback as _traceback

import joblib
import torch
from tqdm import tqdm
from vrAnalysis.sessions import B2Session
from ..registry import PopulationRegistry

from .base import AnalysisConfigBase
from .store import ResultsStore, result_uid


@dataclass(frozen=True)
class Job:
    """A single unit of work: one (session, analysis_config) pair.

    Parameters
    ----------
    session : B2Session
        Session to process.
    analysis_config : AnalysisConfigBase
        Analysis configuration (encodes data_config_name).
    """

    session: B2Session
    analysis_config: AnalysisConfigBase

    @property
    def data_config(self):
        """DataConfig derived from analysis_config.data_config_name."""
        return self.analysis_config.data_config

    @property
    def result_uid(self) -> str:
        """Unified hash identifying this job's result in the store."""
        return result_uid(
            self.session.session_uid,
            self.analysis_config.key(),
        )

    def __repr__(self) -> str:
        return (
            f"Job(session={self.session.session_uid!r}, "
            f"data={self.analysis_config.data_config_name!r}, "
            f"analysis={self.analysis_config.summary()!r}, "
            f"uid={self.result_uid!r})"
        )


def _execute_job(
    job: Job,
    registry: PopulationRegistry,
    store: ResultsStore,
    snapshot_path: str | None,
) -> bool:
    """Execute a single job. Designed for use with joblib.

    Parameters
    ----------
    job : Job
        The job to execute.
    registry : PopulationRegistry
        Population registry (shared across jobs with the same data_config_name).
    store : ResultsStore
        Results store (thread-safe via WAL + busy_timeout).
    snapshot_path : str or None
        Codebase snapshot path to record with the result.

    Returns
    -------
    bool
        True if the job succeeded.
    """
    try:
        job.session.params.spks_type = job.data_config.spks_type
        result = job.analysis_config.process(job.session, registry)
        store.put(
            job.session.session_uid,
            job.analysis_config,
            result,
            snapshot_path=snapshot_path,
        )
        store.clear_error(job.session.session_uid, job.analysis_config)
        return True
    except Exception as e:
        msg = str(e)
        trace = _traceback.format_exc()
        print(f"Error {job}: {msg}")
        try:
            store.put_error(job.session.session_uid, job.analysis_config, msg, trace)
        except Exception:
            pass
        return False
    finally:
        job.session.clear_cache()


@dataclass
class AnalysisPlan:
    """Orchestrator that dispatches analysis configs across sessions.

    Parameters
    ----------
    analysis_configs : list of AnalysisConfigBase
        Analysis configurations to run. Each config encodes its own
        data_config_name, so no separate data_configs list is needed.
    """

    analysis_configs: list[AnalysisConfigBase] = field(default_factory=list)

    def print_job_groups(
        self,
        jobs: list[Job],
        *,
        truncated: int = 0,
        label: str = "Dry run",
        show_sessions: bool = False,
    ) -> None:
        """Print grouped config summary with session counts per unique config.

        Parameters
        ----------
        jobs : list of Job
            Jobs to summarize (typically pending or about to run).
        truncated : int
            Number of additional jobs omitted from ``jobs`` (e.g. ``--max-jobs``).
        label : str
            Prefix for the summary line (e.g. ``"Dry run"`` or ``"Pending"``).
        show_sessions : bool
            If True, print the individual session IDs under each config group.
        """
        from dataclasses import asdict
        from collections import defaultdict

        groups: dict[str, list] = defaultdict(list)
        group_cfg: dict[str, AnalysisConfigBase] = {}
        for job in jobs:
            k = job.analysis_config.key()
            groups[k].append(job.session.session_uid)
            group_cfg[k] = job.analysis_config

        n_sessions = len({job.session.session_uid for job in jobs})
        print(
            f"{label}: {len(jobs)} jobs | {len(group_cfg)} unique configs | {n_sessions} sessions"
            + (f" ({truncated} more jobs skipped)" if truncated else "")
            + ":\n"
        )
        for i, (k, cfg) in enumerate(group_cfg.items(), 1):
            raw = asdict(cfg)
            combo = {key: val for key, val in raw.items() if key not in ("schema_version", "data_config_name")}
            combo_str = ", ".join(f"{key}={val!r}" for key, val in combo.items()) if combo else "(no params)"
            print(
                f"  [{i:>3}] {cfg.display_name} {cfg.schema_version}"
                f" | data={cfg.data_config_name}"
                f" | {combo_str}"
                f" | {len(groups[k])} sessions"
            )
            if show_sessions:
                for sid in sorted(groups[k]):
                    print(f"         {sid}")
        if truncated:
            print(f"\n  ... and {truncated} more jobs (use --max-jobs to increase)")

    def analyze(
        self,
        sessions: list[B2Session],
        store: ResultsStore,
        n_jobs: int = 1,
        force_remake: bool = False,
        snapshot_codebase: bool = True,
        dry_run: bool = False,
        max_jobs: int | None = None,
        skip_errors: bool = False,
        show_sessions: bool = False,
    ):
        """Run all analysis/data config combinations across sessions.

        Parameters
        ----------
        sessions : list of B2Session
            Sessions to process.
        store : ResultsStore
            Store for caching results.
        n_jobs : int
            Number of parallel workers. 1 = sequential, >1 = joblib parallel.
        force_remake : bool
            If True, recompute even if results exist.
        snapshot_codebase : bool
            If True, snapshot the codebase before running.
        dry_run : bool
            If True, print the job list without executing anything.
        max_jobs : int or None
            Maximum number of jobs to run. None = no limit.
        skip_errors : bool
            If True, skip (session, config) pairs that already have a recorded error.
        show_sessions : bool
            If True (dry_run only), print session IDs under each config group.
        """
        all_jobs, n_skipped = self._collect_jobs(sessions, store, force_remake, skip_errors=skip_errors)
        if not all_jobs:
            msg = "All results already computed."
            if n_skipped:
                msg += f" ({n_skipped} skipped due to recorded errors.)"
            print(msg)
            return

        jobs = all_jobs[:max_jobs] if max_jobs is not None else all_jobs
        truncated = len(all_jobs) - len(jobs)

        if dry_run:
            self.print_job_groups(jobs, truncated=truncated, show_sessions=show_sessions)
            if n_skipped:
                print(f"\n  ({n_skipped} additional jobs skipped — already have recorded errors)")
            return

        snapshot_path: str | None = None
        if snapshot_codebase:
            snapshot_path = str(store.snapshot_codebase())

        # Build one registry per unique data_config_name
        registries: dict[str, PopulationRegistry] = {}
        for job in jobs:
            dk = job.analysis_config.data_config_name
            if dk not in registries:
                registries[dk] = PopulationRegistry(registry_params=job.data_config.to_registry_params())

        desc = f"Processing ({len(jobs)}/{len(all_jobs)} jobs)"
        success = 0
        errors = 0
        with tqdm(total=len(jobs), desc=desc) as pbar:
            if n_jobs == 1:
                for job in jobs:
                    tqdm.write(f"Starting: {job.session.session_uid} | {job.analysis_config.summary()}")
                    try:
                        ok = _execute_job(job, registries[job.analysis_config.data_config_name], store, snapshot_path)
                    except BaseException as e:
                        tqdm.write(f"FATAL {type(e).__name__} in {job.session.session_uid}: {e}")
                        raise
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    success += ok
                    errors += not ok
                    pbar.update(1)
            else:
                gen = joblib.Parallel(n_jobs=n_jobs, return_as="generator")(
                    joblib.delayed(_execute_job)(job, registries[job.analysis_config.data_config_name], store, snapshot_path) for job in jobs
                )
                for ok in gen:
                    success += ok
                    errors += not ok
                    pbar.update(1)

        print(f"Done: {success}/{len(jobs)} succeeded, {errors} errors.")

    def _collect_jobs(
        self,
        sessions: list[B2Session],
        store: ResultsStore,
        force_remake: bool,
        skip_errors: bool = False,
    ) -> tuple[list[Job], int]:
        """Return (jobs, n_skipped) for (session, analysis_config) pairs not yet in store.

        Parameters
        ----------
        skip_errors : bool
            If True, omit pairs that already have a recorded error.
            ``n_skipped`` counts how many were omitted for this reason.
        """
        jobs = []
        n_skipped = 0
        for session in sessions:
            sid = session.session_uid
            for acfg in self.analysis_configs:
                if force_remake or not store.has(sid, acfg):
                    if skip_errors and store.has_error(sid, acfg):
                        n_skipped += 1
                        continue
                    jobs.append(Job(session=session, analysis_config=acfg))
        return jobs, n_skipped
