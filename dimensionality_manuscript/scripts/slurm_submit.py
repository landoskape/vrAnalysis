"""Submit CPU worker arrays to Myriad-ng (Slurm).

Example (run on a myriad-ng login node with the new environment activated)::

    python -m dimensionality_manuscript.scripts.slurm_submit --dry-run
    python -m dimensionality_manuscript.scripts.slurm_submit --analyses cvpca --n-workers 4

Set VRANALYSIS_PYTHON_MODULE to a Python module from ucl-stack/2026-07 and
VRANALYSIS_VENV to the corresponding venv before submitting. See MYRIAD_SLURM.md.
Like sge_submit, --dry-run creates a batch but does not submit workers.
Memory is per CPU; each array element runs ONE worker with cores_per_worker CPUs.
"""

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from dimensionality_manuscript.registry import RegistryPaths
from dimensionality_manuscript.scripts.run import _parse_param_filters
from dimensionality_manuscript.scripts.submit_common import prepare_batch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_SCRIPT = Path(__file__).with_name("myriad_slurm_worker.sh")
_LOG_DIR = Path(__file__).parent / "logs"


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _build_sbatch_command(n_workers, walltime, mem, cores_per_worker, tmpfs, log_dir) -> list[str]:
    """Keep resource settings in one place; all apply to each array element."""
    return [
        "sbatch", "--parsable",
        f"--array=1-{n_workers}",
        "--nodes=1", "--ntasks=1",
        f"--cpus-per-task={cores_per_worker}",
        f"--time={walltime}", f"--mem-per-cpu={mem}",
        f"--gres=tmpfs:{tmpfs}",
        "--job-name=dim_manuscript",
        f"--chdir={_REPO_ROOT}",
        f"--output={Path(log_dir) / 'dim_manuscript.%A.%a.log'}",
        "--export=ALL",
        str(_WORKER_SCRIPT),
    ]


def submit(
    analyses: list[str] | None = None,
    sessions_file: Path = _REPO_ROOT / "sessions.json",
    n_workers: int = 16,
    walltime: str = "8:00:00",
    mem: str = "16G",
    cores_per_worker: int = 1,
    tmpfs: str = "10G",
    dry_run: bool = False,
    skip_errors: bool = False,
    param_filters: dict | None = None,
    db_path: Path | None = None,
    max_jobs_per_worker: int | None = None,
    python_module: str | None = None,
    venv: Path | None = None,
    log_dir: Path = _LOG_DIR,
) -> str | None:
    """Create a batch and submit CPU workers; return its ID, or None if empty.

    --db-path isolates queue/results and their sibling blobs/ directory for
    migration tests. Data and population caches still follow paths.toml.
    --max-jobs-per-worker bounds the number of analyses each worker attempts;
    remaining jobs stay pending. Dry runs populate the queue, as in SGE.
    """
    for name, value in (("n_workers", n_workers), ("cores_per_worker", cores_per_worker),
                        ("max_jobs_per_worker", max_jobs_per_worker)):
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    sessions_file = Path(sessions_file).expanduser().resolve()
    if not sessions_file.is_file():
        raise FileNotFoundError(f"Export and transfer the session manifest first: {sessions_file}")
    python_module = python_module or os.environ.get("VRANALYSIS_PYTHON_MODULE")
    if not python_module:
        raise ValueError("Set VRANALYSIS_PYTHON_MODULE or --python-module to a Python module from the new stack")
    venv = Path(venv or os.environ.get("VRANALYSIS_VENV") or "~/Scratch/envs/vrAnalysis-ng").expanduser().resolve()
    if not (venv / "bin" / "activate").is_file():
        raise FileNotFoundError(f"Create the myriad-ng venv first: {venv}")
    db_path = Path(db_path or RegistryPaths.pipeline_v2_db_path).expanduser().resolve()
    log_dir = Path(log_dir).expanduser().resolve()

    batch_id = prepare_batch(db_path, analyses, sessions_file, skip_errors, param_filters)
    if batch_id is None:
        return None
    cmd = _build_sbatch_command(n_workers, walltime, mem, cores_per_worker, tmpfs, log_dir)
    # Put values in the child environment rather than comma-delimited --export
    # assignments: paths may contain commas/spaces, and stale shell batch IDs
    # must never override this submission's freshly created batch.
    env = os.environ.copy()
    env.update(
        DIM_MANUSCRIPT_DB_PATH=str(db_path),
        DIM_MANUSCRIPT_BATCH_ID=batch_id,
        DIM_MANUSCRIPT_REPO=str(_REPO_ROOT),
        DIM_MANUSCRIPT_SESSIONS_FILE=str(sessions_file),
        VRANALYSIS_PYTHON_MODULE=python_module,
        VRANALYSIS_VENV=str(venv),
    )
    env.pop("DIM_MANUSCRIPT_MAX_JOBS", None)
    if max_jobs_per_worker is not None:
        env["DIM_MANUSCRIPT_MAX_JOBS"] = str(max_jobs_per_worker)

    for key in ("DIM_MANUSCRIPT_DB_PATH", "DIM_MANUSCRIPT_BATCH_ID", "DIM_MANUSCRIPT_SESSIONS_FILE",
                "VRANALYSIS_PYTHON_MODULE", "VRANALYSIS_VENV", "DIM_MANUSCRIPT_MAX_JOBS"):
        if key in env:
            print(f"  {key}={env[key]}")
    print(f"\n{'[dry-run] Would submit' if dry_run else 'Submitting'}: {shlex.join(cmd)}")
    if dry_run:
        print("Batch populated; no sbatch call. Re-running creates a NEW batch.")
        return batch_id

    # Slurm opens the output file BEFORE executing the shell launcher.
    log_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"sbatch failed:\n{result.stderr}")
        print(f"Batch {batch_id} remains pending; no workers were submitted by this command.")
        raise SystemExit(1)
    print(f"Slurm job: {result.stdout.strip()}")
    return batch_id


def main():
    parser = argparse.ArgumentParser(description="Queue CPU analyses and submit a Myriad-ng Slurm array")
    parser.add_argument("--analyses", nargs="+", help="Analysis types (default: same as sge_submit)")
    parser.add_argument("--sessions-file", type=Path, default=_REPO_ROOT / "sessions.json")
    parser.add_argument("--n-workers", type=_positive_int, default=16)
    parser.add_argument("--walltime", default="8:00:00")
    parser.add_argument("--mem", default="16G", help="Memory PER CPU (default: 16G)")
    parser.add_argument("--cores-per-worker", type=_positive_int, default=1)
    parser.add_argument("--tmpfs", default="10G", help="Local temporary storage per worker (default: 10G)")
    parser.add_argument("--db-path", type=Path, help="Override results/queue DB; use a separate directory for tests")
    parser.add_argument("--log-dir", type=Path, default=_LOG_DIR)
    parser.add_argument("--max-jobs-per-worker", type=_positive_int, help="Stop each worker after N attempts (for testing)")
    parser.add_argument("--python-module", help="New-stack Python module (default: VRANALYSIS_PYTHON_MODULE)")
    parser.add_argument("--venv", type=Path, help="Venv (default: VRANALYSIS_VENV or ~/Scratch/envs/vrAnalysis-ng)")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Populate a batch without calling sbatch")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--param-filters", nargs="+", metavar="KEY=VALUE")
    args = parser.parse_args()
    options = vars(args)
    options["param_filters"] = _parse_param_filters(args.param_filters)
    submit(**options)


if __name__ == "__main__":
    main()
