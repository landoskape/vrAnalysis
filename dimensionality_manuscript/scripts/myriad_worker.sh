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

# ── Environment ───────────────────────────────────────────────────────────────
# Activate the uv-managed virtual environment (created once on login node).
# Default: ~/Scratch/envs/vrAnalysis — override by setting VRANALYSIS_VENV.
VENV="${VRANALYSIS_VENV:-$HOME/Scratch/envs/vrAnalysis}"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: venv not found at $VENV. See MYRIAD_SETUP.md." >&2
    exit 1
fi
source "$VENV/bin/activate"

# ── Repo ─────────────────────────────────────────────────────────────────────
REPO_DIR="$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"
cd "$REPO_DIR" || { echo "ERROR: could not cd to $REPO_DIR" >&2; exit 1; }

# ── Worker ───────────────────────────────────────────────────────────────────
WORKER_ID="${JOB_ID}.${SGE_TASK_ID}"
mkdir -p "$(dirname "$0")/logs"
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
