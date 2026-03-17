"""AnalysisPlan — orchestrator for running analysis configs across sessions."""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
from tqdm import tqdm
from vrAnalysis.sessions import B2Session
from dimensionality_manuscript.registry import PopulationRegistry

from .base import AnalysisConfigBase
from ..configs.data_config import DataConfig
from .store import ResultsStore, result_uid


@dataclass(frozen=True)
class Job:
    """A single unit of work: one (session, data_config, analysis_config) triple.

    Parameters
    ----------
    session : B2Session
        Session to process.
    data_config : DataConfig
        Data preprocessing configuration.
    analysis_config : AnalysisConfigBase
        Analysis configuration.
    """

    session: B2Session
    data_config: DataConfig
    analysis_config: AnalysisConfigBase

    @property
    def result_uid(self) -> str:
        """Unified hash identifying this job's result in the store."""
        return result_uid(
            self.session.session_uid,
            self.data_config.key(),
            self.analysis_config.key(),
        )

    def __repr__(self) -> str:
        return (
            f"Job(session={self.session.session_uid!r}, "
            f"data={self.data_config.summary()!r}, "
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
        Population registry (shared across jobs with the same DataConfig).
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
            job.data_config,
            job.analysis_config,
            result,
            snapshot_path=snapshot_path,
        )
        return True
    except Exception as e:
        print(f"Error {job}: {e}")
        return False
    finally:
        job.session.clear_cache()


@dataclass
class AnalysisPlan:
    """Orchestrator that dispatches analysis configs across sessions.

    Parameters
    ----------
    analysis_configs : list of AnalysisConfigBase
        Analysis configurations to run.
    data_configs : list of DataConfig
        Data preprocessing configurations to run.
    """

    analysis_configs: list[AnalysisConfigBase] = field(default_factory=list)
    data_configs: list[DataConfig] = field(default_factory=list)

    def analyze(
        self,
        sessions: list[B2Session],
        store: ResultsStore,
        n_jobs: int = 1,
        force_remake: bool = False,
        snapshot_codebase: bool = True,
        dry_run: bool = False,
        max_jobs: int | None = None,
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
        """
        all_jobs = self._collect_jobs(sessions, store, force_remake)
        if not all_jobs:
            print("All results already computed.")
            return

        jobs = all_jobs[:max_jobs] if max_jobs is not None else all_jobs
        truncated = len(all_jobs) - len(jobs)

        if dry_run:
            print(f"Dry run: {len(jobs)} jobs would be executed" + (f" ({truncated} more skipped)" if truncated else "") + ":\n")
            for i, job in enumerate(jobs, 1):
                print(f"  [{i:>4}] {job}")
            if truncated:
                print(f"\n  ... and {truncated} more (use --max-jobs to increase)")
            return

        snapshot_path: str | None = None
        if snapshot_codebase:
            snapshot_path = str(store.snapshot_codebase())

        # Build one registry per unique DataConfig
        registries: dict[str, PopulationRegistry] = {}
        for job in jobs:
            dk = job.data_config.key()
            if dk not in registries:
                registries[dk] = PopulationRegistry(registry_params=job.data_config.to_registry_params())

        desc = f"Processing ({len(jobs)}/{len(all_jobs)} jobs)"
        success = 0
        errors = 0
        with tqdm(total=len(jobs), desc=desc) as pbar:
            if n_jobs == 1:
                for job in jobs:
                    ok = _execute_job(job, registries[job.data_config.key()], store, snapshot_path)
                    success += ok
                    errors += not ok
                    pbar.update(1)
            else:
                gen = joblib.Parallel(n_jobs=n_jobs, return_as="generator")(
                    joblib.delayed(_execute_job)(job, registries[job.data_config.key()], store, snapshot_path) for job in jobs
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
    ) -> list[Job]:
        """Return Jobs for (session, dcfg, acfg) triples not yet in store."""
        jobs = []
        for session in sessions:
            sid = session.session_uid
            for dcfg in self.data_configs:
                for acfg in self.analysis_configs:
                    if force_remake or not store.has(sid, dcfg, acfg):
                        jobs.append(Job(session=session, data_config=dcfg, analysis_config=acfg))
        return jobs
