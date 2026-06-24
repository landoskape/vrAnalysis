"""Populate the SGE job queue and submit worker array jobs to MYRIAD.

Usage
-----
    python -m dimensionality_manuscript.scripts.sge_submit [options]

Run on the MYRIAD login node after transferring data. This script:
1. Collects pending (session, config) pairs not yet in the results store.
2. Creates a new batch and inserts those jobs into the job_queue table.
3. Submits an SGE array job whose workers drain that batch concurrently.

Each call creates an independent batch so re-submitting never disturbs active
workers.  Workers are bound to their batch via DIM_MANUSCRIPT_BATCH_ID.

Use --dry-run to populate the batch without submitting.

Examples
--------
Dry run (inspect what would be submitted before committing):

    python -m dimensionality_manuscript.scripts.sge_submit --dry-run

Submit all analyses (16 workers, default walltime and memory):

    python -m dimensionality_manuscript.scripts.sge_submit --n-workers 16

Submit only specific analysis types with a custom sessions file:

    python -m dimensionality_manuscript.scripts.sge_submit \\
        --analyses cvpca stimspace \\
        --n-workers 8

Retry failed jobs (creates a new batch; already-completed results are skipped):

    python -m dimensionality_manuscript.scripts.sge_submit --n-workers 16

Retry only jobs that don't have a recorded error:

    python -m dimensionality_manuscript.scripts.sge_submit --skip-errors --n-workers 16

Filter to a specific model and spike type:

    python -m dimensionality_manuscript.scripts.sge_submit \\
        --param-filters model_name=rrr spks_type=oasis \\
        --n-workers 8

Submit on A100 (40G) GPU nodes, 1 GPU per worker:

    python -m dimensionality_manuscript.scripts.sge_submit \\
        --gpu-type a100-40 --n-workers 4

Interactive sessions (qrsh) — for manual testing outside the queue, e.g.
trying ``TilburyFitConfig(...).process(..., method="descent", device="cuda")``
by hand before submitting a batch:

    # CPU, 1 core
    qrsh -l h_rt=1:00:00 -l mem=8G

    # CPU, multiple cores
    qrsh -l h_rt=1:00:00 -l mem=8G -pe smp 4

    # GPU (1x A100 40G), 1 core
    qrsh -l h_rt=1:00:00 -l mem=16G -l gpu=1 -ac allow=L

    # GPU (1x A100 40G), multiple cores
    qrsh -l h_rt=1:00:00 -l mem=16G -l gpu=1 -ac allow=L -pe smp 4

``-ac allow=`` letters follow ``_GPU_NODE_LETTERS`` above (EF=V100, L=A100 40G,
UV=A100 80G). Once on the node, the usual venv activation applies (see
MYRIAD_SETUP.md / the ``vrAnalysis``/``vrAnalysisGPU`` shell functions).
"""

import argparse
import subprocess
from pathlib import Path

from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript.pipeline import AnalysisPlan, JobQueue, ResultsStore
from dimensionality_manuscript.scripts.run import (
    build_analysis_configs,
    collect_sessions,
    collect_sessions_from_file,
    _parse_param_filters,
)

REGISTRY_PATHS = RegistryPaths()

_WORKER_SCRIPT = Path(__file__).parent / "myriad_worker.sh"
_REPO_ROOT = Path(__file__).resolve().parents[2]

# MYRIAD GPU node letters by architecture (see RC docs: -ac allow=<letters>).
_GPU_NODE_LETTERS = {"v100": "EF", "a100-40": "L", "a100-80": "UV"}


def submit(
    analyses: list[str] | None = None,
    sessions_file: Path | None = None,
    n_workers: int = 16,
    walltime: str = "8:00:00",
    mem: str = "16G",
    cores_per_worker: int = 1,
    gpu_type: str = "cpu",
    n_gpus: int = 1,
    gpu_nodes: str | None = None,
    dry_run: bool = False,
    skip_errors: bool = False,
    param_filters: dict | None = None,
) -> str | None:
    """Create a batch, populate the queue, and optionally submit an SGE array job.

    Each call produces an independent batch.  Re-submitting while workers are
    active creates a new batch and leaves the active one untouched.

    Parameters
    ----------
    analyses : list[str] or None
        Subset of analysis types to queue. None = all.
    sessions_file : Path or None
        JSON file from export_sessions.py. If None, falls back to the live
        vrSessions database (local use only; not available on MYRIAD).
    n_workers : int
        Number of parallel SGE worker slots (``-t 1-N``).
    walltime : str
        SGE wall-clock time limit (``h_rt``), e.g. ``"8:00:00"``.
    mem : str
        SGE memory per core (``mem``), e.g. ``"16G"``. On MYRIAD this is
        per-slot, so total memory per worker is ``cores_per_worker * mem``.
    cores_per_worker : int
        Cores per array task (``-pe smp``). Each worker exposes this as
        ``NSLOTS``, which configs read to cap intra-job joblib parallelism (e.g.
        the per-neuron Tilbury fits). ``1`` keeps the previous serial behaviour.
    gpu_type : {"cpu", "v100", "a100-40", "a100-80"}
        GPU architecture to request. ``"cpu"`` (default) submits a plain CPU
        job — no ``-l gpu`` / ``-ac allow`` flags are added. Any other value
        adds both, mapped via ``_GPU_NODE_LETTERS``.
    n_gpus : int
        Number of GPUs per worker (``-l gpu=N``). Ignored when ``gpu_type="cpu"``.
    gpu_nodes : str or None
        Explicit ``-ac allow=`` letters, overriding the ``gpu_type`` mapping
        (e.g. if MYRIAD's node-letter scheme changes). Ignored when
        ``gpu_type="cpu"``.
    dry_run : bool
        If True, populate the batch but do not submit via qsub.
        Useful for inspecting what would run before committing.
    skip_errors : bool
        If True, omit (session, config) pairs that already have a recorded error.

    Returns
    -------
    str or None
        The batch_id created, or None if there was nothing to do.
    """
    db_path = REGISTRY_PATHS.pipeline_v2_db_path
    store = ResultsStore(db_path)
    queue = JobQueue(db_path)

    if sessions_file is not None:
        sessions = collect_sessions_from_file(sessions_file)
    else:
        sessions = collect_sessions()

    analysis_configs = build_analysis_configs(include=analyses, param_filters=param_filters)
    plan = AnalysisPlan(analysis_configs=analysis_configs)
    pending_jobs, n_skipped = plan._collect_jobs(sessions, store, force_remake=False, skip_errors=skip_errors)

    skip_suffix = f" ({n_skipped} skipped — recorded errors)" if n_skipped else ""
    print(f"Sessions:         {len(sessions)}")
    print(f"Analysis configs: {len(analysis_configs)}")
    print(f"Pending jobs:     {len(pending_jobs)}{skip_suffix}")
    print(f"Database:         {db_path}")
    print()

    if pending_jobs:
        plan.print_job_groups(pending_jobs, label="Pending")
        print()

    if not pending_jobs:
        print("Nothing to do — all results already computed.")
        _print_batch_summary(queue)
        return None

    batch_id = queue.create_batch(analyses)
    n_added = queue.populate(pending_jobs, batch_id)
    summary = queue.status_summary(batch_id)
    print(f"\nBatch:        {batch_id}")
    print(f"Jobs added:   {n_added} | {summary}")

    if dry_run:
        print("\n[dry-run] Batch populated. Run smoke_test to validate, then re-run without --dry-run to submit.")
        _print_queue_preview(n_workers, walltime, mem, cores_per_worker, gpu_type, n_gpus, gpu_nodes, db_path, sessions_file, batch_id)
        return batch_id

    qsub_cmd = _build_qsub_command(n_workers, walltime, mem, cores_per_worker, gpu_type, n_gpus, gpu_nodes, db_path, sessions_file, batch_id)
    print(f"\nSubmitting: {' '.join(qsub_cmd)}")
    result = subprocess.run(qsub_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"qsub failed:\n{result.stderr}")
        raise SystemExit(1)
    print(result.stdout.strip())
    return batch_id


def _build_qsub_command(
    n_workers: int,
    walltime: str,
    mem: str,
    cores_per_worker: int,
    gpu_type: str,
    n_gpus: int,
    gpu_nodes: str | None,
    db_path: Path,
    sessions_file: Path | None,
    batch_id: str,
) -> list[str]:
    env_vars = f"DIM_MANUSCRIPT_DB_PATH={db_path}," f"DIM_MANUSCRIPT_BATCH_ID={batch_id}," f"DIM_MANUSCRIPT_REPO={_REPO_ROOT}"
    if sessions_file is not None:
        env_vars += f",DIM_MANUSCRIPT_SESSIONS_FILE={Path(sessions_file).resolve()}"
    cmd = [
        "qsub",
        "-t",
        f"1-{n_workers}",
        "-l",
        f"h_rt={walltime}",
        "-l",
        f"mem={mem}",
        "-pe",
        "smp",
        str(cores_per_worker),
    ]
    if gpu_type != "cpu":
        letters = gpu_nodes or _GPU_NODE_LETTERS[gpu_type]
        cmd += ["-l", f"gpu={n_gpus}", "-ac", f"allow={letters}"]
    cmd += ["-v", env_vars, str(_WORKER_SCRIPT)]
    return cmd


def _print_queue_preview(n_workers, walltime, mem, cores_per_worker, gpu_type, n_gpus, gpu_nodes, db_path, sessions_file, batch_id):
    qsub_cmd = _build_qsub_command(n_workers, walltime, mem, cores_per_worker, gpu_type, n_gpus, gpu_nodes, db_path, sessions_file, batch_id)
    print(f"qsub command: {' '.join(qsub_cmd)}")


def _print_batch_summary(queue: JobQueue) -> None:
    batches = queue.list_batches()
    if batches:
        print("\nExisting batches (newest first):")
        for b in batches[:5]:
            print(f"  {b['batch_id']}  pending={b['pending']} running={b['running']} done={b['done']} failed={b['failed']}")


def main():
    parser = argparse.ArgumentParser(description="Queue SGE jobs and submit to MYRIAD")
    parser.add_argument("--analyses", nargs="+", help="Analysis types to queue (default: all)")
    parser.add_argument(
        "--sessions-file",
        type=Path,
        default=_REPO_ROOT / "sessions.json",
        help="JSON session list from export_sessions.py (default: <repo>/sessions.json)",
    )
    parser.add_argument("--n-workers", type=int, default=16, help="Number of parallel SGE worker slots (default: 16)")
    parser.add_argument("--walltime", default="8:00:00", help="SGE wall-clock limit, e.g. 8:00:00 (default: 8:00:00)")
    parser.add_argument("--mem", default="16G", help="SGE memory per core, e.g. 16G (default: 16G)")
    parser.add_argument(
        "--cores-per-worker",
        type=int,
        default=1,
        help="Cores per array task (-pe smp); exposed to configs as NSLOTS for intra-job parallelism (default: 1)",
    )
    parser.add_argument(
        "--gpu-type",
        choices=["cpu", "v100", "a100-40", "a100-80"],
        default="cpu",
        help="GPU architecture to request (default: cpu — no GPU resource flags added)",
    )
    parser.add_argument("--n-gpus", type=int, default=1, help="GPUs per worker (-l gpu=N). Ignored if --gpu-type=cpu (default: 1)")
    parser.add_argument(
        "--gpu-nodes",
        default=None,
        help="Override -ac allow=<letters> node selection (default: mapped from --gpu-type)",
    )
    parser.add_argument("--dry-run", "-n", action="store_true", help="Populate batch but do not submit via qsub")
    parser.add_argument("--skip-errors", action="store_true", help="Omit (session, config) pairs that already have a recorded error")
    parser.add_argument(
        "--param-filters",
        nargs="+",
        metavar="KEY=VALUE",
        help="Filter config grid by fixed param values, e.g. --param-filters model_name=rrr spks_type=oasis",
    )
    args = parser.parse_args()

    submit(
        analyses=args.analyses,
        sessions_file=args.sessions_file,
        n_workers=args.n_workers,
        walltime=args.walltime,
        mem=args.mem,
        cores_per_worker=args.cores_per_worker,
        gpu_type=args.gpu_type,
        n_gpus=args.n_gpus,
        gpu_nodes=args.gpu_nodes,
        dry_run=args.dry_run,
        skip_errors=args.skip_errors,
        param_filters=_parse_param_filters(args.param_filters),
    )


if __name__ == "__main__":
    main()
