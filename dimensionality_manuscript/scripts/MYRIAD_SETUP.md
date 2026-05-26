# Running the dimensionality manuscript pipeline on MYRIAD (UCL)

MYRIAD is a Sun Grid Engine (SGE) cluster. This guide covers environment setup,
data transfer, job submission, and syncing results back to your local machine.

---

## Overview

```
Local machine                          MYRIAD login node
─────────────────────────────────────  ──────────────────────────────────────────
1. export_sessions.py  ──sessions.json──▶  sge_submit.py --sessions-file
2. rsync raw data      ──────────────────▶  ~/Scratch/data/
                                           sge_submit.py populates job_queue
                                           qsub -t 1-N → worker array
                                           workers drain queue → results.db + blobs/
3. sync_from_myriad.py ◀─────────────────  results.db + blobs/
   local results.db merged transparently
```

---

## 1. SSH configuration

Add to `~/.ssh/config` so you can use `myriad` as a shorthand everywhere:

```
Host myriad
    HostName login12.myriad.ucl.ac.uk
    User YOUR_UCL_USERNAME
    ForwardAgent yes
```

Test: `ssh myriad hostname`

---

## 2. Environment setup on MYRIAD (one-time)

Do this on a **login node** (which has internet). Compute nodes do not.

### 2a. Install uv

```bash
ssh myriad
curl -LsSf https://astral.sh/uv/install.sh | sh
# uv is now at ~/.local/bin/uv — already on PATH after shell restart
exec bash
```

### 2b. Find a Python 3.11 module

```bash
module avail 2>&1 | grep -i python
# Look for python/3.11.x — exact name varies
module load python/3.11.4  # adjust version
python --version            # confirm
```

### 2c. Create virtual environment on Scratch

Use `~/Scratch/` (Lustre, large quota) not `~` (home, tiny quota).
`$(which python)` may not work in all shells — pass the version string directly:

```bash
uv venv ~/Scratch/envs/vrAnalysis --python=3.11.4
# module load must precede activation — the venv links against libpython from the module
module load python/3.11.4   # must match version used when creating the venv
source ~/Scratch/envs/vrAnalysis/bin/activate
```

### 2d. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/vrAnalysis ~/vrAnalysis
cd ~/vrAnalysis
```

### 2e. Install dependencies

Do NOT use `uv pip install -e .` — `setup.py` pulls in jupyterlab, pyodbc,
and other packages the worker never needs. Install only what the worker uses.

MYRIAD defaults to the Intel ICC compiler, which fails to compile C extensions
for numpy, scipy, etc. Use `--only-binary :all:` to force pre-built wheels and
skip ICC entirely. One dependency (`freezedry`) has a pure-Python transitive dep
with no wheel, so install it separately without the flag:

```bash
# Step 1: all scientific deps — force pre-built wheels, no ICC compilation
# numpy<2 must be last — opencv-python has a loose numpy constraint and can bump to numpy 2.x
uv pip install scikit-image matplotlib syd joblib tqdm numpyencoder speedystats numba optuna opencv-python umap-learn --only-binary :all:
uv pip install scipy pandas scikit-learn "numpy<2" --only-binary :all:

# Step 2: freezedry separately (its dep gitignore-parser is pure Python — no compilation)
uv pip install freezedry

# Step 3: PyTorch CPU-only
# uv's download can stall on the login node (170 MB wheel). wget is much faster.
# The filename MUST include the version — uv rejects a bare "torch.whl".
wget -c "https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl" \
    -O "/tmp/torch-2.6.0+cpu-cp311-cp311-linux_x86_64.whl"
uv pip install "/tmp/torch-2.6.0+cpu-cp311-cp311-linux_x86_64.whl"

# Step 4: install the package itself without pulling in setup.py deps
uv pip install -e ~/vrAnalysis --no-deps

# Verify
python -c "from dimensionality_manuscript import CVPCAConfig, ResultsStore; print('OK')"
```

### 2f. Configure paths.toml

`paths.toml` is gitignored (each machine has its own). Create it on MYRIAD:

```bash
cp ~/vrAnalysis/paths.toml.example ~/vrAnalysis/paths.toml
```

Edit `~/vrAnalysis/paths.toml`:

```toml
[paths]
repo        = "/home/YOUR_UCL_USERNAME/vrAnalysis"
storage     = "/home/YOUR_UCL_USERNAME/Scratch/data"
literature  = "/home/YOUR_UCL_USERNAME/Scratch/literature"
odbc_driver = ""   # not used on Linux — vrSessions loaded from JSON instead
```

The `storage` path is where `local_data_path()` points — session data files
must live under this root (mirroring your local layout).

### 2g. Create log directory

```bash
mkdir -p ~/vrAnalysis/dimensionality_manuscript/scripts/logs
```

---

## 3. Data transfer (local → MYRIAD)

Run these **locally**. Large transfers: run inside `tmux` on MYRIAD or use
`nohup` locally.

### 3a. Export the session list

The vrSessions database is a Windows-only Microsoft Access file. Export a
portable JSON instead:

```bash
# On local machine:
cd /path/to/vrAnalysis
python -m dimensionality_manuscript.scripts.export_sessions --output sessions.json
```

### 3b. Install rsync in Git Bash (Windows — one-time)

Windows does not ship rsync. Git Bash does not include it. Install the MSYS2
binaries into `~/bin` (no admin required):

```bash
# In Git Bash:
curl -L "https://repo.msys2.org/msys/x86_64/rsync-3.3.0-1-x86_64.pkg.tar.zst" -o /tmp/rsync.pkg.tar.zst
curl -L "https://repo.msys2.org/msys/x86_64/libxxhash-0.8.3-1-x86_64.pkg.tar.zst" -o /tmp/xxhash.pkg.tar.zst

mkdir -p ~/bin
cd /tmp && bsdtar -xf rsync.pkg.tar.zst && bsdtar -xf xxhash.pkg.tar.zst
cp /tmp/usr/bin/rsync.exe ~/bin/
cp /tmp/usr/bin/msys-*.dll ~/bin/ 2>/dev/null || true
cp /tmp/usr/bin/msys-xxhash-0.dll ~/bin/

export PATH="$HOME/bin:$PATH"
rsync --version  # confirm MSYS2 rsync, not choco
```

Add `export PATH="$HOME/bin:$PATH"` to `~/.bashrc` so it persists.

> **Note:** The Chocolatey rsync (`choco install rsync`) does NOT work —
> it misinterprets Windows drive letters (`D:`) as remote hostnames.
> Use the MSYS2 binary above instead.

### 3c. Transfer sessions.json and session data

Three things are needed per session — `oneData/` (neural + behavioural
arrays), `roicat/` (per-session classifier results), and `vrExperiment*.json`
(session config). Plus one global file — `analysis/roicat_classification/train_classifier.joblib`
— which all workers need to construct sessions. Everything else (`suite2p/`,
`spkmaps/`, raw timeline `.npy`, `.mat` files) is either already processed
into `oneData/` or recomputed on MYRIAD.

Use `transfer_to_myriad.py` to rsync exactly the sessions in your manifest and
nothing more:

```powershell
# Session manifest (small)
scp sessions.json myriad:~/vrAnalysis/sessions.json

# Run from PowerShell — NOT Git Bash (Git Bash expands ~ before Python sees it)
# If conda activate doesn't work in PowerShell, prefix with: conda run -n vrAnalysis

# Dry run first — confirm what will be sent
conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.transfer_to_myriad --sessions-file sessions.json --local-data D:/localData --host myriad --remote-data ~/Scratch/data --dry-run

# Real transfer (session data only)
conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.transfer_to_myriad --sessions-file sessions.json --local-data D:/localData --host myriad --remote-data ~/Scratch/data

# Transfer session data AND seed MYRIAD with local results.db so already-computed
# jobs are skipped (no blobs needed — the db is enough for workers to skip them)
conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.transfer_to_myriad --sessions-file sessions.json --local-data D:/localData --host myriad --remote-data ~/Scratch/data --include-results
```

The MYRIAD results DB is created automatically on first run if it does not exist.

---

## 4. Submitting jobs

Run on the **MYRIAD login node**:

```bash
ssh myriad
cd ~/vrAnalysis
module load python/3.11.4   # must match version used when creating the venv
source ~/Scratch/envs/vrAnalysis/bin/activate

# Dry run first — populates the queue but does NOT submit.
# Run smoke tests (§5) after this to validate before committing.
python -m dimensionality_manuscript.scripts.sge_submit --sessions-file ~/vrAnalysis/sessions.json --dry-run

# Submit for real (16 workers, 8h walltime, 16G RAM each)
python -m dimensionality_manuscript.scripts.sge_submit \
    --sessions-file ~/vrAnalysis/sessions.json \
    --n-workers 16 \
    --walltime 8:00:00 \
    --mem 16G

# Run only specific analysis types
python -m dimensionality_manuscript.scripts.sge_submit \
    --sessions-file ~/vrAnalysis/sessions.json \
    --analyses cvpca stimspace \
    --n-workers 8
```

### Monitoring

Each `sge_submit` call creates a named batch (``YYYYMMDD_HHMMSS_<8hex>``).
Workers are bound to their batch — re-submitting never disturbs active workers.

```bash
qstat -u $USER              # SGE job status
qstat -j JOB_ID             # detailed status for one job
tail -f logs/dim_manuscript.JOB_ID.1.log   # live worker log

# List all batches with per-status counts:
python -c "
from dimensionality_manuscript.pipeline import JobQueue
from dimensionality_manuscript.registry import RegistryPaths
for b in JobQueue(RegistryPaths.pipeline_v2_db_path).list_batches():
    print(b['batch_id'], 'pending=%d running=%d done=%d failed=%d' % (b['pending'], b['running'], b['done'], b['failed']))
"

# Status of a specific batch:
python -c "
from dimensionality_manuscript.pipeline import JobQueue
from dimensionality_manuscript.registry import RegistryPaths
print(JobQueue(RegistryPaths.pipeline_v2_db_path).status_summary('BATCH_ID_HERE'))
"
```

### Re-submitting

Failed jobs in a batch stay as-is.  To retry them, just re-run `sge_submit` —
it creates a fresh batch containing only jobs not yet in the results store
(which includes any that failed, since they produced no result).

```bash
# Re-submit remaining/failed work (creates a new batch):
python -m dimensionality_manuscript.scripts.sge_submit \
    --sessions-file ~/vrAnalysis/sessions.json \
    --n-workers 16
```

---

## 5. Smoke tests (run on MYRIAD login node before submitting)

Two scripts validate the pipeline before committing to a full job array run.
Both are safe to run while the real queue is live. Run them on the login node
where `results.db` and `sessions.json` live.

```bash
ssh myriad
cd ~/vrAnalysis
module load python/3.11.4   # must match version used when creating the venv
source ~/Scratch/envs/vrAnalysis/bin/activate
```

### 5a. Resolution check

Verifies that N queued jobs resolve to known sessions and analysis configs:

```bash
python -m dimensionality_manuscript.scripts.smoke_test --n-jobs 2 --sessions-file ~/vrAnalysis/sessions.json
```

Reports `[OK]` or `[FAIL]` per job, then restores all jobs to `pending`.
A `FAIL` means `session_id` or `analysis_key` is unrecognised — usually a
stale queue (re-populate with `sge_submit.py`) or a missing entry in `sessions.json`.

### 5b. Concurrency check

Proves that W workers draining the same queue don't double-claim jobs.
Copies N pending jobs to a temp DB, spawns W local worker processes in
dry-run mode, then asserts every job ends up `done` exactly once:

```bash
python -m dimensionality_manuscript.scripts.concurrency_test --n-jobs 6 --n-workers 3 --sessions-file ~/vrAnalysis/sessions.json
```

The real DB is never touched — workers run against the temp copy.
`PASS` confirms SQLite `BEGIN IMMEDIATE` atomicity holds under concurrent access,
which is the same guarantee the SGE array relies on.

---

## 6. Syncing results back to local

Run **locally** after MYRIAD jobs finish. Requires the MSYS2 rsync installed
in §3b — the script finds it automatically via `~/bin/rsync.exe`.


```powershell
# Run from PowerShell (not Git Bash — ~ expansion issue)

# Dry run — shows rsync output without transferring
conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.sync_from_myriad --host myriad --dry-run

# Real sync
conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.sync_from_myriad --host myriad
```

This:
1. rsyncs `.pkl` blob files (skips blobs already present locally)
2. Downloads the MYRIAD `results.db` to a temp file
3. Merges new rows into your local `results.db` via `INSERT OR IGNORE`

After syncing, `ResultsStore` and `ResultsAggregator` are completely unaware
the results came from a server — they see a fully-populated local store.

---

## 7. Scratch disk layout on MYRIAD

```
~/Scratch/
├── data/                          ← raw session data + pipeline outputs (storage root)
│   ├── MOUSE1/
│   │   └── 2024-01-15/
│   │       └── 001/
│   ├── ...
│   └── dimensionality-manuscript/
│       └── cache/
│           └── pipeline_v2/
│               ├── results.db     ← SQLite results + job_queue tables
│               └── blobs/
│                   └── <result_uid>.pkl
└── envs/
    └── vrAnalysis/                ← uv virtual environment
```

The `paths.toml` `storage` key must point to `~/Scratch/data` so that
`local_data_path()` resolves session file paths correctly on MYRIAD.
`RegistryPaths.pipeline_v2_db_path` then resolves to
`~/Scratch/data/dimensionality-manuscript/cache/pipeline_v2/results.db`.

---

## 8. Common problems

| Problem | Fix |
|---------|-----|
| `libpython3.11.so.1.0: cannot open shared object file` | `myriad_worker.sh` loads `python/3.11.4` before the venv; if your module name differs, set `VRANALYSIS_PYTHON_MODULE` in `qsub -v` or edit the worker default |
| `Repo: /var/opt/sge` in worker log | Old worker used `$0` to find the repo; SGE stages scripts under `/var/opt/sge`. Update `myriad_worker.sh` and re-submit (uses `DIM_MANUSCRIPT_REPO` from `sge_submit.py`) |
| `mkdir: ... /var/opt/sge/.../logs: Permission denied` | Same as above — harmless warning on old scripts; logs still go to `#$ -o` under `~/vrAnalysis/.../logs/` |
| `uv: command not found` on compute node | Check `~/.local/bin` is on PATH in `~/.bashrc`; or use full path `~/.local/bin/uv` |
| `venv not found at ...` | Wrong `VRANALYSIS_VENV`; check path or unset env var to use default |
| `paths.toml not found` | Create it from `paths.toml.example` (see §2f) |
| Jobs claim 0 pending | Queue not populated — run `sge_submit.py` first |
| `Unknown analysis_key` in worker log | Config schema changed since queue was populated — re-populate with `sge_submit.py` |
| SQLite `database is locked` | Increase `busy_timeout` in `job_queue.py`; or check for a hung process holding a write lock |
| Blobs on Lustre slow | Normal — Lustre has high latency for small files. Consider `--tmpfs` in SGE script for local scratch during a job, then copy back |
| `torch` import slow on first run | Expected — torch is large. Subsequent imports use the disk cache |

---

## 9. Quick reference

**Local** (conda env):
```powershell
# Export sessions
conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.export_sessions --output sessions.json

# Transfer data (+ optionally seed results.db)
conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.transfer_to_myriad --sessions-file sessions.json --local-data D:/localData --host myriad --remote-data ~/Scratch/data
conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.transfer_to_myriad --sessions-file sessions.json --local-data D:/localData --host myriad --remote-data ~/Scratch/data --include-results

# Skip rsync, upload results.db only
conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.transfer_to_myriad --sessions-file sessions.json --host myriad --skip-transfer --include-results

# Sync results back
conda run -n vrAnalysis python -m dimensionality_manuscript.scripts.sync_from_myriad --host myriad
```

**MYRIAD** (`module load python/3.11.4 && source ~/Scratch/envs/vrAnalysis/bin/activate` first):
```bash
# Smoke test: resolution check
python -m dimensionality_manuscript.scripts.smoke_test --n-jobs 2 --sessions-file ~/vrAnalysis/sessions.json

# Smoke test: concurrency check
python -m dimensionality_manuscript.scripts.concurrency_test --n-jobs 6 --n-workers 3 --sessions-file ~/vrAnalysis/sessions.json

# Dry-run submit (creates batch, no qsub — use for smoke test)
python -m dimensionality_manuscript.scripts.sge_submit --sessions-file ~/vrAnalysis/sessions.json --dry-run

# Real submit (creates new batch, submits workers)
python -m dimensionality_manuscript.scripts.sge_submit --sessions-file ~/vrAnalysis/sessions.json --n-workers 16

# Submit subset of analyses
python -m dimensionality_manuscript.scripts.sge_submit --sessions-file ~/vrAnalysis/sessions.json --analyses cvpca stimspace --n-workers 8

# List batches and status
python -c "from dimensionality_manuscript.pipeline import JobQueue; from dimensionality_manuscript.registry import RegistryPaths; [print(b['batch_id'], b['pending'], b['running'], b['done'], b['failed']) for b in JobQueue(RegistryPaths.pipeline_v2_db_path).list_batches()]"

# Check queue
python -c "from dimensionality_manuscript.pipeline import JobQueue; from dimensionality_manuscript.registry import RegistryPaths; print(JobQueue(RegistryPaths.pipeline_v2_db_path).status_summary())"
```
