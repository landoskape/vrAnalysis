#!/bin/bash
# CPU launcher for Myriad-ng; resources and log path come from slurm_submit.py.
set -eo pipefail

: "${DIM_MANUSCRIPT_DB_PATH:?Use slurm_submit.py to submit}"
: "${DIM_MANUSCRIPT_BATCH_ID:?Use slurm_submit.py to submit}"
: "${DIM_MANUSCRIPT_SESSIONS_FILE:?Export sessions.json first}"
: "${DIM_MANUSCRIPT_REPO:?Use slurm_submit.py to submit}"
: "${VRANALYSIS_PYTHON_MODULE:?Set a Python module from ucl-stack/2026-07}"
: "${VRANALYSIS_VENV:?Set the matching myriad-ng venv}"
: "${SLURM_ARRAY_JOB_ID:?Expected a Slurm array}"
: "${SLURM_ARRAY_TASK_ID:?Expected a Slurm array}"

cd "$DIM_MANUSCRIPT_REPO"
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
fi
# Rebuild module state: Slurm inherits the submitting shell's environment.
module purge
module load ucl-stack/2026-07
module load default-modules/2026-07
module load "$VRANALYSIS_PYTHON_MODULE"
source "$VRANALYSIS_VENV/bin/activate"
# Module initialization/activation scripts may reference unset shell variables.
set -u

# One Python worker per array element; constrain native library threading.
export NSLOTS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$NSLOTS"
export MKL_NUM_THREADS="$NSLOTS"
export OPENBLAS_NUM_THREADS="$NSLOTS"
export NUMEXPR_NUM_THREADS="$NSLOTS"
WORKER_ID="slurm.${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID}"
echo "[$WORKER_ID] Starting on $(hostname) at $(date)"
echo "[$WORKER_ID] Python: $(command -v python)"
echo "[$WORKER_ID] CPUs: $NSLOTS; TMPDIR: ${TMPDIR:-unset}"
echo "[$WORKER_ID] DB: $DIM_MANUSCRIPT_DB_PATH; Batch: $DIM_MANUSCRIPT_BATCH_ID"

args=(--worker-id "$WORKER_ID" --db-path "$DIM_MANUSCRIPT_DB_PATH"
      --batch-id "$DIM_MANUSCRIPT_BATCH_ID" --sessions-file "$DIM_MANUSCRIPT_SESSIONS_FILE")
if [ -n "${DIM_MANUSCRIPT_MAX_JOBS:-}" ]; then
    args+=(--max-jobs "$DIM_MANUSCRIPT_MAX_JOBS")
fi
# The Python queue worker is scheduler-independent despite its historical name.
exec python -u -m dimensionality_manuscript.scripts.sge_worker "${args[@]}"
