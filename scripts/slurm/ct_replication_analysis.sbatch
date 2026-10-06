#!/usr/bin/env bash
#SBATCH --job-name=ct-repl-015-analysis
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0
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
: "${CT_REPLICATION_STAGE:?}"
[[ "$CT_REPLICATION_STAGE" == analyze || "$CT_REPLICATION_STAGE" == verify ]]
cd "$CT_UPDATE_CODE_ROOT"
sha256sum --check --status EXECUTION_SHA256SUMS
test "$(sha256sum EXECUTION_SHA256SUMS | cut -d' ' -f1)" = "$CT_REPLICATION_SOURCE_SHA256"
export PYTHONPATH="$CT_UPDATE_CODE_ROOT/src:$CT_UPDATE_PACKAGE_ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
"$CT_UPDATE_PYTHON" -m unittest discover -s tests -p 'test_ct_*.py'
"$CT_UPDATE_PYTHON" -m careport.ct_replication_run "$CT_REPLICATION_STAGE" \
  --policy config/preanalysis/ct_replication_015_2026-09-25.json
