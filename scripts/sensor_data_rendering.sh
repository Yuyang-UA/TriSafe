#!/usr/bin/env bash
# Stage 2: render multi-view camera images (MTGS) for the SOLVABLE risky cases only.
# Inputs are the stage-1 package from scripts/risky_case_construction.sh and the
# solvable token list written by nuplan_scripts/run_solvability_check.py.
# Each reconstruction config's MTGS model is loaded once.
#
# Output (SENSOR_DATA_ROOT, a NAVSIM package with solvable scenes only):
#   sensor_blobs/<log>/<camera>/<frame token>.jpg
#   synthetic_scene_pickles/<scene token>.pkl   (same tokens as stage 1; checked)
#   selected_candidates_manifest.json / _token_mapping.json / _render_summary.json
#
# Example:
#   PYTHON_BIN=/path/to/envs/mtgs/bin/python bash scripts/sensor_data_rendering.sh
#
# Useful overrides:
#   RISKY_CASE_ROOT=path              Stage-1 package (must match risky_case_construction.sh).
#   SOLVABLE_TOKENS=path              Token list (default: $RISKY_CASE_ROOT/solvability_check/synthetic_tokens_nc_1_dac_1.txt).
#   SENSOR_DATA_ROOT=path             Output package location.
#   DRY_RUN=1                         Print the planned renderer commands only.
#   OVERWRITE=1                       Replace scenes/images from an earlier run.
#   CONTINUE_ON_ERROR=1               Finish the batch and report all failures.
#   CHECKPOINT_EVERY=50               Persist progress after this many configs.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${OPENSCENE_DATA_ROOT:-$PROJECT_ROOT/../dataset}"
RISKY_CASE_ROOT="${RISKY_CASE_ROOT:-$DATA_ROOT/trisafe_risky_cases}"
SOLVABLE_TOKENS="${SOLVABLE_TOKENS:-$RISKY_CASE_ROOT/solvability_check/synthetic_tokens_nc_1_dac_1.txt}"
SENSOR_DATA_ROOT="${SENSOR_DATA_ROOT:-$DATA_ROOT/trisafe_solvable_sensor_data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    echo "Set PYTHON_BIN to the Python interpreter for the MTGS/NAVSIM environment." >&2
    exit 1
fi

for required_path in "$RISKY_CASE_ROOT/selected_candidates_manifest.json" "$SOLVABLE_TOKENS"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Error: required path does not exist: $required_path" >&2
        echo "Run scripts/risky_case_construction.sh and nuplan_scripts/run_solvability_check.py first." >&2
        exit 1
    fi
done

extra_args=()
[[ -n "${NAVSIM_LOG_PATH:-}" ]] && extra_args+=(--navsim-log-path "$NAVSIM_LOG_PATH")
[[ "${DRY_RUN:-0}" == "1" ]] && extra_args+=(--dry-run)
[[ "${OVERWRITE:-0}" == "1" ]] && extra_args+=(--overwrite)
[[ "${CONTINUE_ON_ERROR:-0}" == "1" ]] && extra_args+=(--continue-on-error)

"$PYTHON_BIN" nuplan_scripts/auto_reconstruction/selected_candidates_pipeline.py sensors \
    --risky-case-root "$RISKY_CASE_ROOT" \
    --solvable-tokens "$SOLVABLE_TOKENS" \
    --output-root "$SENSOR_DATA_ROOT" \
    --checkpoint-every "${CHECKPOINT_EVERY:-25}" \
    "${extra_args[@]}"
