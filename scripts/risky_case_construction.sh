#!/usr/bin/env bash
# Stage 1: build NAVSIM synthetic scene pickles for every HGS-selected candidate,
# WITHOUT sensor data. The MTGS model is not loaded and no images are rendered
# (renderer --dry-run), so this needs no GPU. The pickles are the input of
# nuplan_scripts/run_solvability_check.py; solvable scenes are then rendered by
# scripts/sensor_data_rendering.sh.
#
# Output (RISKY_CASE_ROOT):
#   synthetic_scene_pickles/<scene token>.pkl
#   selected_candidates_manifest.json        stage-2 input: scene token <-> config/token/candidate
#   selected_candidates_token_mapping.json   new_token -> original_token, candidate index
#   selected_candidates_render_summary.json  settings, counts, skipped windows, failures
#
# Example:
#   PYTHON_BIN=/path/to/envs/mtgs/bin/python bash scripts/risky_case_construction.sh
#   python nuplan_scripts/run_solvability_check.py \
#       --synthetic-scenes-path "$RISKY_CASE_ROOT/synthetic_scene_pickles" \
#       --output-dir "$RISKY_CASE_ROOT/solvability_check" --workers 8
#
# Useful overrides:
#   SELECTED_CANDIDATES_JSON=path     HGS output (default: greedy_search.sh output).
#   RISKY_CASE_ROOT=path              Output package location.
#   DRY_RUN=1                         Print the planned renderer commands only.
#   OVERWRITE=1                       Replace scene pickles from an earlier run.
#   CONTINUE_ON_ERROR=1               Finish the batch and report all failures.
#   CHECKPOINT_EVERY=50               Persist progress after this many configs.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${OPENSCENE_DATA_ROOT:-$PROJECT_ROOT/../dataset}"
SELECTED_CANDIDATES_JSON="${SELECTED_CANDIDATES_JSON:-Hierarchical Greedy Search/K_9/summary_all/selected_candidates_by_token.json}"
CONFIG_DIR="${CONFIG_DIR:-data/auto_reconstruction/config/navtest_available_merged_pointcloud_good}"
MODEL_TOKEN_SUMMARY="${MODEL_TOKEN_SUMMARY:-data/auto_reconstruction/summary.json}"
CANDIDATE_TRAJECTORY_PATH="${CANDIDATE_TRAJECTORY_PATH:-trajectory_candidate/human_trajectory_candidate.npy}"
RISKY_CASE_ROOT="${RISKY_CASE_ROOT:-$DATA_ROOT/trisafe_risky_cases}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-$DATA_ROOT/navsim_logs/test}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    echo "Set PYTHON_BIN to the Python interpreter for the MTGS/NAVSIM environment." >&2
    exit 1
fi

for required_path in "$SELECTED_CANDIDATES_JSON" "$CONFIG_DIR" "$MODEL_TOKEN_SUMMARY" "$CANDIDATE_TRAJECTORY_PATH" "$NAVSIM_LOG_PATH"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Error: required path does not exist: $required_path" >&2
        exit 1
    fi
done

extra_args=()
[[ "${DRY_RUN:-0}" == "1" ]] && extra_args+=(--dry-run)
[[ "${OVERWRITE:-0}" == "1" ]] && extra_args+=(--overwrite)
[[ "${CONTINUE_ON_ERROR:-0}" == "1" ]] && extra_args+=(--continue-on-error)

"$PYTHON_BIN" nuplan_scripts/auto_reconstruction/selected_candidates_pipeline.py construct \
    --selected-candidates-json "$SELECTED_CANDIDATES_JSON" \
    --config-dir "$CONFIG_DIR" \
    --model-token-summary "$MODEL_TOKEN_SUMMARY" \
    --candidate-trajectory-path "$CANDIDATE_TRAJECTORY_PATH" \
    --navsim-log-path "$NAVSIM_LOG_PATH" \
    --output-root "$RISKY_CASE_ROOT" \
    --checkpoint-every "${CHECKPOINT_EVERY:-25}" \
    "${extra_args[@]}"
