# CPU analyses on Myriad-ng (Slurm)

`slurm_submit.py` prepares the same batches as `sge_submit.py` and submits an
array of CPU workers. Each array element runs the existing Python queue worker.
GPU requests are not implemented in this entry point.

UCL references: [migration](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/),
[Slurm usage and Myriad resources](https://www.rc.ucl.ac.uk/docs/Supplementary/Slurm/),
[example scripts](https://www.rc.ucl.ac.uk/docs/Supplementary/Slurm_Example_Jobscripts/).
Software module names must be checked on the cluster.

## 1. New environment on the myriad-ng login node

Connect using your UCL username:

```bash
ssh YOUR_UCL_USERNAME@myriad-ng.rc.ucl.ac.uk
cd ~/vrAnalysis
module purge
module load ucl-stack/2026-07
module load default-modules/2026-07
module avail 2>&1 | grep -i python
```

Choose an available Python module compatible with the project (prefer Python
3.11 if available). Substitute its **actual module name** below. The old
`python/3.11.4` name is not assumed to exist in this stack.

```bash
export VRANALYSIS_PYTHON_MODULE='REPLACE_WITH_AVAILABLE_PYTHON_MODULE'
module load "$VRANALYSIS_PYTHON_MODULE"
export VRANALYSIS_VENV="$HOME/Scratch/envs/vrAnalysis-ng"
uv venv "$VRANALYSIS_VENV" --python "$(command -v python)"
source "$VRANALYSIS_VENV/bin/activate"

uv pip install scikit-image matplotlib syd joblib tqdm numpyencoder 'speedystats>=0.1.1' numba optuna opencv-python umap-learn --only-binary :all:
uv pip install scipy pandas scikit-learn 'numpy<2' --only-binary :all:
uv pip install freezedry
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ~/vrAnalysis --no-deps
python -c "import numpy, scipy, torch, speedystats; from dimensionality_manuscript import CVPCAConfig, ResultsStore; print('Imports OK; CUDA available:', torch.cuda.is_available())"
```

This installs CPU PyTorch; CUDA being unavailable is expected. No GPU test is
needed for this workflow. Reuse your existing `paths.toml`, transferred data,
session manifest, and population caches; see [MYRIAD_SETUP.md](MYRIAD_SETUP.md).
Do not overwrite the old SGE venv. On later logins, reload the stack and Python
module, re-export both `VRANALYSIS_*` variables, and activate the new venv.
The launcher rebuilds module state before activating that same venv.

## 2. One compute-node environment test

From the configured login shell:

```bash
srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem-per-cpu=8G --gres=tmpfs:10G --time=0:15:00 --pty bash -l
cd ~/vrAnalysis
module purge
module load ucl-stack/2026-07
module load default-modules/2026-07
module load "$VRANALYSIS_PYTHON_MODULE"
source "$VRANALYSIS_VENV/bin/activate"
hostname
python -c "import numpy, scipy, torch, speedystats; from dimensionality_manuscript import CVPCAConfig, ResultsStore; print('Compute-node imports OK')"
exit
```

Expect a compute-node hostname and successful imports. Install missing packages
on the login node, then repeat; compute nodes need no downloads.

## 3. Prepare a small, isolated test

On the login node, select one session whose data is already transferred. This
example uses the first record; choose another record if needed.

```bash
cd ~/vrAnalysis
TEST_ROOT="$HOME/Scratch/slurm-validation-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TEST_ROOT"
python -c "import json; from pathlib import Path; records=json.loads(Path('sessions.json').read_text()); assert records, 'Empty manifest'; Path('$TEST_ROOT/sessions.json').write_text(json.dumps(records[:1], indent=2))"

python -m dimensionality_manuscript.scripts.slurm_submit \
    --analyses cvpca --sessions-file "$TEST_ROOT/sessions.json" \
    --db-path "$TEST_ROOT/single/results.db" \
    --n-workers 1 --cores-per-worker 1 --mem 16G \
    --walltime 0:30:00 --max-jobs-per-worker 1 --dry-run

python -m dimensionality_manuscript.scripts.smoke_test \
    --n-jobs 2 --db-path "$TEST_ROOT/single/results.db" \
    --sessions-file "$TEST_ROOT/sessions.json"

python -m dimensionality_manuscript.scripts.concurrency_test \
    --n-jobs 3 --n-workers 3 --db-path "$TEST_ROOT/single/results.db" \
    --sessions-file "$TEST_ROOT/sessions.json"
```

Expect a printed `sbatch` command, resolution checks with no failures, and a
concurrency-test PASS. If fewer than three jobs are pending, lower `--n-jobs`.
The concurrency test runs local subprocesses against a temporary DB; it does
not exercise Slurm or concurrent access from different compute nodes.

`--dry-run` **writes a batch**, but never calls `sbatch`. A subsequent submit
creates a NEW batch, so the unused preview batch stays pending. It does not
run automatically. Always inspect the batch ID from the actual submission.

The test DB and sibling `blobs/` are separate from production. Analyses still
read data and may populate normal reusable caches under `paths.toml` storage.

## 4. Submit one real analysis

```bash
python -m dimensionality_manuscript.scripts.slurm_submit \
    --analyses cvpca --sessions-file "$TEST_ROOT/sessions.json" \
    --db-path "$TEST_ROOT/single/results.db" \
    --n-workers 1 --cores-per-worker 1 --mem 16G \
    --walltime 0:30:00 --max-jobs-per-worker 1

squeue --me
# Replace JOB_ID below with the printed Slurm job number.
sacct -j JOB_ID --format=JobID,State,ExitCode,AllocCPUS,NodeList
tail -f dimensionality_manuscript/scripts/logs/dim_manuscript.JOB_ID.1.log
```

Expect one successful analysis and a `.pkl` under `$TEST_ROOT/single/blobs/`.
The worker stops after one attempt, leaving the rest pending intentionally.
If 30 minutes is insufficient, inspect the log and retry with more walltime.
`scancel JOB_ID` cancels that submission if needed.

Check application results after the job ends (paste its printed batch ID):

```bash
python - "$TEST_ROOT/single/results.db" BATCH_ID <<'PY'
import sqlite3
import sys
with sqlite3.connect(sys.argv[1]) as db:
    print('Queue:', db.execute('SELECT status, count(*) FROM job_queue WHERE batch_id=? GROUP BY status', (sys.argv[2],)).fetchall())
    print('Results:', db.execute('SELECT count(*) FROM results').fetchone()[0])
    print('Errors:', db.execute('SELECT session_id, error_message FROM errors').fetchall())
PY
```

Require `done=1`, one stored result, and no errors for this fresh test directory.
Check these even if Slurm reports COMPLETED: the existing Python worker records
analysis failures in the queue and can still exit with code zero.

## 5. Small real array

Use a fresh DB directory so the single-worker result does not skip the test:

```bash
python -m dimensionality_manuscript.scripts.slurm_submit \
    --analyses cvpca --sessions-file "$TEST_ROOT/sessions.json" \
    --db-path "$TEST_ROOT/array/results.db" \
    --n-workers 3 --cores-per-worker 2 --mem 8G \
    --walltime 0:30:00 --max-jobs-per-worker 1
```

This requests three workers with two CPUs and 16G RAM **each**, each attempting
one analysis. Repeat the accounting/log/DB checks from step 4 with the new job
and batch IDs and `$TEST_ROOT/array/results.db`. If at least three jobs were
queued, expect three done, three results, no errors, and separate worker IDs
`slurm.JOB_ID.1`, `.2`, `.3`. Check `AllocCPUS=2` and the logged thread count.

**Existing storage limitation:** both the queue and results store use SQLite
WAL mode. [SQLite requires WAL clients to be on the same host and does not
support WAL over network filesystems](https://sqlite.org/wal.html). Slurm arrays
can place workers on different hosts. A passing local concurrency test, or
even an array that happens to pass, does not establish that this arrangement
is supported. Use disposable DBs for migration tests; resolve the database
deployment/journaling approach with RC support before scaling shared-DB arrays
across nodes. This limitation also applies to the existing SGE design.

## 6. Normal CPU submission after validation

When the environment and shared-store deployment are validated:

```bash
python -m dimensionality_manuscript.scripts.slurm_submit \
    --sessions-file ~/vrAnalysis/sessions.json --analyses cvpca \
    --n-workers 16 --cores-per-worker 1 --mem 16G --walltime 8:00:00
```

Omit `--db-path` to use production results, and omit `--max-jobs-per-worker`
to drain each batch normally. Omit `--analyses` to use the same default analysis
set as SGE. Existing `--skip-errors` and `--param-filters KEY=VALUE ...` work.
Batch planning is shared between the two entry points; the SGE GPU options
remain available only through `sge_submit`.

`--mem` always means memory per CPU: `--cores-per-worker 4 --mem 16G` requests
64G per worker. `--tmpfs` defaults to 10G per worker. The launcher sets
`NSLOTS` for compatibility and caps OpenMP/MKL/OpenBLAS/NumExpr thread counts
to `SLURM_CPUS_PER_TASK`; it does not automatically parallelize serial analyses.
Log directories are created before submission, because Slurm opens logs before
the launcher runs.

Avoid overlapping SGE/Slurm production submissions for the same pending
analyses: batches are independent, so active work in another batch is not
deduplicated. Existing data-transfer and result-sync commands are unchanged.

## Local regression tests

From PowerShell in the repository:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& 'C:\Users\Andrew\AppData\Local\miniforge3\envs\vrAnalysis\python.exe' -m pytest tests/test_hpc_submit.py -q
```

These test real temporary batch queues, mocked scheduler submission, SGE
compatibility, preflight failures, and the shell launcher with fake
modules/Python. They do not submit cluster jobs or verify UCL module availability.
