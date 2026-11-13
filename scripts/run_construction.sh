#!/usr/bin/env bash
# Run the four MTGS reconstruction stages in order, each in its conda env.
#
# Usage (from anywhere):
#   bash scripts/run_construction.sh
#
# Overrides:
#   CONDA_ENV        env for stages 1, 2, 3 and 4 (default: trisafe)
#   MTGS_METHOD      stage 4 method: mtgs_st (default) or mtgs_deformable
#   CONFIG_DIR, LIST_FILE, NUM_WORKERS, NUM_GPUS, CUDA_VISIBLE_DEVICES
#                    see scripts/construction_common.sh
#
# The stages can also be run one at a time:
#   bash scripts/construction_batch_1.sh   # ... then _2, _3, _4

set -euo pipefail

# Make conda activate available in non-interactive shells.
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not available. Please initialize conda first." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-trisafe}"
DEPTH_CONDA_ENV="${DEPTH_CONDA_ENV:-${CONDA_ENV}}"

run_step() {
    local step_name=$1
    local env_name=$2
    local script_path=$3
    local setup_cmds=${4:-}

    echo "============================================"
    echo "[${step_name}] Activating conda env: ${env_name}"
    echo "============================================"
    conda activate "${env_name}"

    if [[ -n "${setup_cmds}" ]]; then
        echo "[${step_name}] Running environment setup: ${setup_cmds}"
        eval "${setup_cmds}"
    fi

    echo "[${step_name}] Running: ${script_path}"
    bash "${script_path}"

    echo "[${step_name}] Finished successfully."
    echo ""
}

# Step 1: register the MTGS method and nuPlan dataparser with nerfstudio.
run_step "Step 1" "${CONDA_ENV}"       "scripts/construction_batch_1.sh" \
    'export PYTHONPATH="${PYTHONPATH:-}"; source dev.sh && mtgs_setup mtgs/config/MTGS.py'
run_step "Step 2" "${CONDA_ENV}" "scripts/construction_batch_2.sh"
run_step "Step 3" "${CONDA_ENV}"       "scripts/construction_batch_3.sh"
run_step "Step 4" "${CONDA_ENV}"       "scripts/construction_batch_4.sh"

echo "All four construction stages completed."
