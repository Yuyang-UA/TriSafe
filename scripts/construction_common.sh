#!/usr/bin/env bash
# Shared driver for scripts/construction_batch_{1,2,3,4}.sh.
#
# Runs nuplan_scripts/auto_reconstruction/recon_single_config_<stage>.sh once per
# reconstruction config. Each scene is written to <data_root>/<config name>, where
# data_root is read from the config itself. Later stages and the NAVSIM renderer
# look up the reconstruction in the same place.
#
# Environment overrides:
#   CONFIG_DIR            Directory of reconstruction configs
#                         (default: data/auto_reconstruction/config/navtest_available_merged_pointcloud_good)
#   LIST_FILE             Optional file with one config filename per line ('#' comments allowed).
#                         Default: every *.yaml in CONFIG_DIR.
#   EXPORT_DIR            Export directory passed to the stage scripts
#                         (default: data/auto_reconstruction/export/<basename of CONFIG_DIR>)
#   NUM_WORKERS           CPU workers per scene (default: 4)
#   NUM_GPUS              GPUs per scene for the accelerate stages (default: 1)
#   CUDA_VISIBLE_DEVICES  GPU(s) to use (default: 0)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_DIR="${CONFIG_DIR:-data/auto_reconstruction/config/navtest_available_merged_pointcloud_good}"
LIST_FILE="${LIST_FILE:-}"
EXPORT_DIR="${EXPORT_DIR:-data/auto_reconstruction/export/$(basename "${CONFIG_DIR}")}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_GPUS="${NUM_GPUS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

list_configs() {
    if [[ -n "${LIST_FILE}" ]]; then
        grep -vE '^[[:space:]]*(#|$)' "${LIST_FILE}"
    else
        find "${CONFIG_DIR}" -maxdepth 1 -name '*.yaml' -printf '%f\n' | sort
    fi
}

config_data_root() {
    # Top-level "data_root: <path>" entry of a reconstruction config.
    sed -n 's/^data_root:[[:space:]]*//p' "$1" | head -n 1 | tr -d "'\""
}

# run_stage <stage> [extra args for recon_single_config_<stage>.sh ...]
run_stage() {
    local stage=$1
    shift
    local stage_script="nuplan_scripts/auto_reconstruction/recon_single_config_${stage}.sh"

    if [[ ! -d "${CONFIG_DIR}" ]]; then
        echo "Error: CONFIG_DIR does not exist: ${CONFIG_DIR}" >&2
        exit 1
    fi
    if [[ -n "${LIST_FILE}" && ! -f "${LIST_FILE}" ]]; then
        echo "Error: LIST_FILE does not exist: ${LIST_FILE}" >&2
        exit 1
    fi

    local config_name config config_id data_root
    while IFS= read -r config_name; do
        config="${CONFIG_DIR}/${config_name}"
        config_id="$(basename "${config}" .yaml)"
        data_root="$(config_data_root "${config}")"
        if [[ -z "${data_root}" ]]; then
            echo "Error: no data_root in ${config}" >&2
            exit 1
        fi

        echo "========================================"
        echo "Stage ${stage}: ${config}"
        echo "Output:  ${data_root}/${config_id}"
        echo "========================================"

        bash "${stage_script}" \
            --config "${config}" \
            --output-dir "${data_root}/${config_id}" \
            --export-dir "${EXPORT_DIR}" \
            --workers "${NUM_WORKERS}" \
            --gpus "${NUM_GPUS}" \
            "$@"
    done < <(list_configs)
}
