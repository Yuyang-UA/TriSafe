#!/usr/bin/env bash
# Solvability check of the risky cases built by scripts/risky_case_construction.sh.
# Reads RISKY_CASE_ROOT/synthetic_scene_pickles (no sensor data needed) and writes
# RISKY_CASE_ROOT/solvability_check/, whose synthetic_tokens_nc_1_dac_1.txt is the
# default input of scripts/sensor_data_rendering.sh.
#
# Output (SOLVABILITY_OUTPUT_DIR):
#   synthetic_tokens_nc_1_dac_1.txt               solvable scenes (NC = 1 and DAC = 1)
#   synthetic_tokens_{nc_0_or_0_5_dac_1,nc_1_dac_0,nc_0_or_0_5_dac_0,unscored_errors}.txt
#   raw_reactive_idm_solvability.csv              per-scene metrics and selected candidate
#   solvable_trajectories/<token>.csv             witness trajectory of each solvable scene
#   raw_reactive_idm_solvability_summary.json     settings and counts
#
# Example:
#   PYTHON_BIN=/path/to/envs/mtgs/bin/python bash scripts/solvability_verification.sh
#
# Useful overrides:
#   RISKY_CASE_ROOT=path              Stage-1 package (must match risky_case_construction.sh).
#   SOLVABILITY_OUTPUT_DIR=path       Output directory (default: $RISKY_CASE_ROOT/solvability_check).
#   WORKERS=8                         Parallel scene workers.
#   ITERATIONS=3                      Ego/traffic best-response rounds.
#   TOKENS_FILE=path                  Only check the scene tokens listed in this file.
#   MAX_SCENES=20                     Only check the first N scenes (quick test).
#   NUPLAN_MAPS_ROOT=path             nuPlan maps (default: $OPENSCENE_DATA_ROOT/maps).
#   VERBOSE=1                         Log each scene's category and selected candidate.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${OPENSCENE_DATA_ROOT:-$PROJECT_ROOT/../dataset}"
RISKY_CASE_ROOT="${RISKY_CASE_ROOT:-$DATA_ROOT/trisafe_risky_cases}"
SYNTHETIC_SCENES_PATH="$RISKY_CASE_ROOT/synthetic_scene_pickles"
SOLVABILITY_OUTPUT_DIR="${SOLVABILITY_OUTPUT_DIR:-$RISKY_CASE_ROOT/solvability_check}"
WORKERS="${WORKERS:-8}"
ITERATIONS="${ITERATIONS:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    echo "Set PYTHON_BIN to the Python interpreter for the MTGS/NAVSIM environment." >&2
    exit 1
fi

if [[ ! -d "$SYNTHETIC_SCENES_PATH" ]]; then
    echo "Error: scene pickles not found: $SYNTHETIC_SCENES_PATH" >&2
    echo "Run scripts/risky_case_construction.sh first." >&2
    exit 1
fi

extra_args=()
[[ -n "${NUPLAN_MAPS_ROOT:-}" ]] && extra_args+=(--map-root "$NUPLAN_MAPS_ROOT")
[[ -n "${TOKENS_FILE:-}" ]] && extra_args+=(--tokens-file "$TOKENS_FILE")
[[ -n "${MAX_SCENES:-}" ]] && extra_args+=(--max-scenes "$MAX_SCENES")
[[ "${VERBOSE:-0}" == "1" ]] && extra_args+=(--verbose)

"$PYTHON_BIN" nuplan_scripts/run_solvability_check.py \
    --synthetic-scenes-path "$SYNTHETIC_SCENES_PATH" \
    --output-dir "$SOLVABILITY_OUTPUT_DIR" \
    --workers "$WORKERS" \
    --iterations "$ITERATIONS" \
    "${extra_args[@]}"
