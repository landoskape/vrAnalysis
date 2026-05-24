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
module load python/3.11.6  # adjust version
python --version            # confirm
```

### 2c. Create virtual environment on Scratch

Use `~/Scratch/` (Lustre, large quota) not `~` (home, tiny quota).
`$(which python)` may not work in all shells — pass the version string directly:

```bash
uv venv ~/Scratch/envs/vrAnalysis --python=3.11.4
source ~/Scratch/envs/vrAnalysis/bin/activate
```

### 2d. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/vrAnalysis ~/vrAnalysis
cd ~/vrAnalysis
```

### 2e. Install dependencies

Do NOT use `uv pip install -e .` — `setup.py` pulls in matplotlib, jupyterlab,
pyodbc, syd, and other packages the worker never needs. Install only what the
worker actually uses.

MYRIAD defaults to the Intel ICC compiler, which fails to compile C extensions
for numpy, scipy, etc. Use `--only-binary :all:` to force pre-built wheels and
skip ICC entirely. One dependency (`freezedry`) has a pure-Python transitive dep
with no wheel, so install it separately without the flag:

```bash
# Step 1: all scientific deps — force pre-built wheels, no ICC compilation
uv pip install numpy"<2" scipy pandas scikit-learn joblib tqdm numpyencoder speedystats numba --only-binary :all:

# Step 2: freezedry separately (its dep gitignore-parser is pure Python — no compilation)
uv pip install freezedry

# Step 3: PyTorch CPU-only
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

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

### 3b. Transfer sessions.json and raw data

```bash
# Session manifest (small)
scp sessions.json myriad:~/vrAnalysis/sessions.json

# Raw session data (large — mirror your local layout under Scratch/data/)
rsync -avP --progress \
    /path/to/local/data/ \
    myriad:~/Scratch/data/

# If you already have partial results locally and want to seed MYRIAD's DB:
rsync -avP \
    /path/to/local/pipeline_v2/results.db \
    myriad:~/Scratch/dim_manuscript/pipeline_v2/results.db
rsync -avP \
    /path/to/local/pipeline_v2/blobs/ \
    myriad:~/Scratch/dim_manuscript/pipeline_v2/blobs/
```

The MYRIAD results DB is created automatically on first run if it does not exist.

---

## 4. Submitting jobs

Run on the **MYRIAD login node**:

```bash
ssh myriad
cd ~/vrAnalysis
source ~/Scratch/envs/vrAnalysis/bin/activate

# Dry run first — see pending job count and qsub command
python -m dimensionality_manuscript.scripts.sge_submit \
    --sessions-file ~/vrAnalysis/sessions.json \
    --dry-run

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

```bash
qstat -u $USER              # SGE job status
qstat -j JOB_ID             # detailed status for one job
tail -f logs/dim_manuscript.JOB_ID.1.log   # live worker log

# Queue summary (from Python):
python -c "
from dimensionality_manuscript.pipeline import JobQueue
from dimensionality_manuscript.registry import RegistryPaths
q = JobQueue(RegistryPaths.pipeline_v2_db_path)
print(q.status_summary())
"
```

### Re-queuing failures

```bash
python -m dimensionality_manuscript.scripts.sge_submit \
    --sessions-file ~/vrAnalysis/sessions.json \
    --force-repopulate \
    --dry-run   # check count first

# Then resubmit without --dry-run
```

---

## 5. Syncing results back to local

Run **locally** after MYRIAD jobs finish:

```bash
# Dry run — shows rsync output without transferring
python -m dimensionality_manuscript.scripts.sync_from_myriad \
    --host myriad \
    --dry-run

# Real sync
python -m dimensionality_manuscript.scripts.sync_from_myriad \
    --host myriad \
    --remote-db "~/Scratch/dim_manuscript/pipeline_v2/results.db" \
    --remote-blobs "~/Scratch/dim_manuscript/pipeline_v2/blobs/"
```

This:
1. rsyncs `.pkl` blob files (skips blobs already present locally)
2. Downloads the MYRIAD `results.db` to a temp file
3. Merges new rows into your local `results.db` via `INSERT OR IGNORE`

After syncing, `ResultsStore` and `ResultsAggregator` are completely unaware
the results came from a server — they see a fully-populated local store.

---

## 6. Scratch disk layout on MYRIAD

```
~/Scratch/
├── data/                          ← raw session data (mirrors local storage/)
│   ├── MOUSE1/
│   │   └── 2024-01-15/
│   │       └── 001/
│   └── ...
├── dim_manuscript/
│   └── pipeline_v2/
│       ├── results.db             ← SQLite results + job_queue tables
│       └── blobs/
│           └── <result_uid>.pkl
└── envs/
    └── vrAnalysis/                ← uv virtual environment
```

The `paths.toml` `storage` key must point to `~/Scratch/data` so that
`local_data_path()` resolves session file paths correctly on MYRIAD.

---

## 7. Common problems

| Problem | Fix |
|---------|-----|
| `uv: command not found` on compute node | Check `~/.local/bin` is on PATH in `~/.bashrc`; or use full path `~/.local/bin/uv` |
| `venv not found at ...` | Wrong `VRANALYSIS_VENV`; check path or unset env var to use default |
| `paths.toml not found` | Create it from `paths.toml.example` (see §2f) |
| Jobs claim 0 pending | Queue not populated — run `sge_submit.py` first |
| `Unknown analysis_key` in worker log | Config schema changed since queue was populated — re-populate with `sge_submit.py` |
| SQLite `database is locked` | Increase `busy_timeout` in `job_queue.py`; or check for a hung process holding a write lock |
| Blobs on Lustre slow | Normal — Lustre has high latency for small files. Consider `--tmpfs` in SGE script for local scratch during a job, then copy back |
| `torch` import slow on first run | Expected — torch is large. Subsequent imports use the disk cache |

---

## 8. Quick reference

```bash
# Local: export sessions
python -m dimensionality_manuscript.scripts.export_sessions --output sessions.json

# MYRIAD: dry-run submit
python -m dimensionality_manuscript.scripts.sge_submit --sessions-file sessions.json --dry-run

# MYRIAD: real submit
python -m dimensionality_manuscript.scripts.sge_submit --sessions-file sessions.json --n-workers 16

# MYRIAD: check queue
python -c "from dimensionality_manuscript.pipeline import JobQueue; from dimensionality_manuscript.registry import RegistryPaths; print(JobQueue(RegistryPaths.pipeline_v2_db_path).status_summary())"

# Local: sync back
python -m dimensionality_manuscript.scripts.sync_from_myriad --host myriad
```
