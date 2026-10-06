#!/usr/bin/env bash
#SBATCH --job-name=ct-repl-015
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0
#SBATCH --array=1-20%1
set -euo pipefail
umask 077
: "${CT_UPDATE_CODE_ROOT:?}"
: "${CT_UPDATE_REMOTE_ROOT:?}"
: "${CT_UPDATE_REMOTE_DATA_ROOT:?}"
: "${CT_UPDATE_PANTRACK_REMOTE_ROOT:?}"
: "${CT_UPDATE_PYTHON:?}"
: "${CT_UPDATE_PACKAGE_ROOT:?}"
: "${CT_UPDATE_MODEL_FREEZE_SHA256:?}"
: "${CT_REPLICATION_POLICY_SHA256:?}"
: "${CT_REPLICATION_SOURCE_SHA256:?}"
: "${SLURM_ARRAY_TASK_ID:?}"
cd "$CT_UPDATE_CODE_ROOT"
sha256sum --check --status EXECUTION_SHA256SUMS
test "$(sha256sum EXECUTION_SHA256SUMS | cut -d' ' -f1)" = "$CT_REPLICATION_SOURCE_SHA256"
export PYTHONPATH="$CT_UPDATE_CODE_ROOT/src:$CT_UPDATE_PACKAGE_ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MONAI_DATA_DIRECTORY="$CT_UPDATE_REMOTE_ROOT/runtime/monai"
"$CT_UPDATE_PYTHON" -m unittest discover -s tests -p 'test_ct_*.py'
policy=config/preanalysis/ct_replication_015_2026-09-25.json
index="$SLURM_ARRAY_TASK_ID"
seed="$("$CT_UPDATE_PYTHON" -m careport.ct_replication_run seed --policy "$policy" --index "$index")"
execution="$("$CT_UPDATE_PYTHON" -m careport.ct_replication_run execution-policy --policy "$policy" --index "$index")"
training="$CT_UPDATE_REMOTE_ROOT/runs/replication_015/seed_${seed}/training"
echo "{\"stage\": \"replication_run\", \"index\": $index, \"seed\": $seed}"
# The unchanged trainer creates its output exclusively; a failed attempt must be
# renamed and preserved before one permitted technical resubmission.
if [[ ! -f "$training/training_complete.json" ]]; then
  "$CT_UPDATE_PYTHON" -m careport.ct_update_train \
    --config config/preanalysis/ct_update_001_2026-09-15.json \
    --cache "$CT_UPDATE_REMOTE_ROOT/derived/patches_001" \
    --output "$training" --execution-policy "$execution"
fi
"$CT_UPDATE_PYTHON" -m careport.ct_replication_run evaluate --policy "$policy" --index "$index"
