#!/bin/bash
# SGE worker script for MYRIAD (UCL).
# Submit via sge_submit.py, which injects DIM_MANUSCRIPT_DB_PATH and
# DIM_MANUSCRIPT_SESSIONS_FILE via qsub -v.
#
# Each array task loops claiming jobs from the shared SQLite queue until
# it is empty, then exits cleanly.
#
# Tune -l h_rt and -l mem per job profile.
# Tune -t 1-N via --n-workers in sge_submit.py.

#$ -N dim_manuscript
#$ -l h_rt=8:00:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -pe smp 1
#$ -cwd
#$ -j y
#$ -o /home/$USER/vrAnalysis/dimensionality_manuscript/scripts/logs/dim_manuscript.$JOB_ID.$TASK_ID.log

# ── Validate injected env vars ────────────────────────────────────────────────
if [ -z "$DIM_MANUSCRIPT_DB_PATH" ]; then
    echo "ERROR: DIM_MANUSCRIPT_DB_PATH not set. Use sge_submit.py to submit." >&2
    exit 1
fi
if [ -z "$DIM_MANUSCRIPT_BATCH_ID" ]; then
    echo "ERROR: DIM_MANUSCRIPT_BATCH_ID not set. Use sge_submit.py to submit." >&2
    exit 1
fi
if [ -z "$DIM_MANUSCRIPT_SESSIONS_FILE" ]; then
    echo "ERROR: DIM_MANUSCRIPT_SESSIONS_FILE not set. Run export_sessions.py locally first." >&2
    exit 1
fi

# ── Repo (do not use $0 — SGE stages this script under /var/opt/sge/...) ─────
REPO_DIR="${DIM_MANUSCRIPT_REPO:-$HOME/vrAnalysis}"
cd "$REPO_DIR" || { echo "ERROR: could not cd to $REPO_DIR" >&2; exit 1; }

# ── Python module + venv ──────────────────────────────────────────────────────
# The venv links against libpython from the module; compute nodes do not load it
# automatically (unlike an interactive login shell).
if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck source=/dev/null
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
    # shellcheck source=/dev/null
    source /usr/share/Modules/init/bash
fi
PYTHON_MODULE="${VRANALYSIS_PYTHON_MODULE:-python/3.11.4}"
if ! module load "$PYTHON_MODULE"; then
    echo "ERROR: module load $PYTHON_MODULE failed. Set VRANALYSIS_PYTHON_MODULE or see MYRIAD_SETUP.md." >&2
    exit 1
fi

VENV="${VRANALYSIS_VENV:-$HOME/Scratch/envs/vrAnalysis}"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: venv not found at $VENV. See MYRIAD_SETUP.md." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

# ── Worker ───────────────────────────────────────────────────────────────────
WORKER_ID="${JOB_ID}.${SGE_TASK_ID}"
LOG_DIR="$REPO_DIR/dimensionality_manuscript/scripts/logs"
mkdir -p "$LOG_DIR"
echo "[$WORKER_ID] Starting on $(hostname) at $(date)"
echo "[$WORKER_ID] DB:       $DIM_MANUSCRIPT_DB_PATH"
echo "[$WORKER_ID] Batch:    $DIM_MANUSCRIPT_BATCH_ID"
echo "[$WORKER_ID] Sessions: $DIM_MANUSCRIPT_SESSIONS_FILE"
echo "[$WORKER_ID] Repo:     $REPO_DIR"

python -m dimensionality_manuscript.scripts.sge_worker \
    --worker-id "$WORKER_ID" \
    --db-path "$DIM_MANUSCRIPT_DB_PATH" \
    --batch-id "$DIM_MANUSCRIPT_BATCH_ID" \
    --sessions-file "$DIM_MANUSCRIPT_SESSIONS_FILE"

EXIT_CODE=$?
echo "[$WORKER_ID] Finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
