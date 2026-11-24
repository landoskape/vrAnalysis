#!/bin/bash
#$ -N myriad_sge_smoke
#$ -l h_rt=00:02:00
#$ -l mem=1G
#$ -cwd
#$ -j y

# From a dedicated directory in ~/Scratch on legacy Myriad:
# qsub ~/vrAnalysis/dimensionality_manuscript/scripts/myriad_sge_smoke.sh
set -eu

: "${JOB_ID:?Submit this script with qsub on legacy Myriad}"
host=$(hostname)
printf 'SGE_SMOKE_OK job=%s host=%s\n' "$JOB_ID" "$host"
