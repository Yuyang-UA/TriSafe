#!/usr/bin/env bash
# Regenerate the route driving command of every rendered risky case from its
# solvable PDM witness (nuplan_scripts/regenerate_driving_commands_from_solvability.py).
# Run after scripts/sensor_data_rendering.sh and before model testing.
# The scene pickles in RENDERED_DATA_ROOT are updated in place.
#
# Output (RENDERED_DATA_ROOT):
#   synthetic_token_driving_commands.json              token -> one-hot [left, straight, right, unknown],
#                                                      read by scripts/test_ltf.sh / test_mlp.sh
#   driving_command_regeneration_from_solvability.json per-scene manifest (old/new command, geometry)
#   synthetic_scene_pickles/<token>.pkl                driving_command replaced in every history frame
#
# Example:
#   RENDERED_DATA_ROOT=path/to/rendered_data PYTHON_BIN=/path/to/envs/mtgs/bin/python \
#       bash scripts/generate_driving_command.sh
#
# Useful overrides:
#   RENDERED_DATA_ROOT=path   Rendered package (output of scripts/sensor_data_rendering.sh).
#   RISKY_CASE_ROOT=path      Stage-1 package; its solvability_check/ is the witness source.
#   SOLVABILITY_DIR=path      Solvability output (default: $RISKY_CASE_ROOT/solvability_check).
#   NUPLAN_MAPS_ROOT=path     nuPlan maps (default: $OPENSCENE_DATA_ROOT/maps).
#   DRY_RUN=1                 Write the manifest only; pickles and command JSON are untouched.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${OPENSCENE_DATA_ROOT:-$PROJECT_ROOT/../dataset}"
RENDERED_DATA_ROOT="${RENDERED_DATA_ROOT:-path/to/rendered_data}"
RISKY_CASE_ROOT="${RISKY_CASE_ROOT:-$DATA_ROOT/trisafe_risky_cases}"
SOLVABILITY_DIR="${SOLVABILITY_DIR:-$RISKY_CASE_ROOT/solvability_check}"
MAP_ROOT="${NUPLAN_MAPS_ROOT:-$DATA_ROOT/maps}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    echo "Set PYTHON_BIN to the Python interpreter for the MTGS/NAVSIM environment." >&2
    exit 1
fi

for required_path in "$RENDERED_DATA_ROOT/synthetic_scene_pickles" \
    "$SOLVABILITY_DIR/raw_reactive_idm_solvability.csv" \
    "$SOLVABILITY_DIR/solvable_trajectory_index.csv"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Error: required path does not exist: $required_path" >&2
        echo "Set RENDERED_DATA_ROOT / RISKY_CASE_ROOT, and run scripts/solvability_verification.sh" \
            "and scripts/sensor_data_rendering.sh first." >&2
        exit 1
    fi
done

extra_args=()
[[ "${DRY_RUN:-0}" == "1" ]] && extra_args+=(--dry-run)

"$PYTHON_BIN" nuplan_scripts/regenerate_driving_commands_from_solvability.py \
    --synthetic-scenes-path "$RENDERED_DATA_ROOT/synthetic_scene_pickles" \
    --solvability-dir "$SOLVABILITY_DIR" \
    --map-root "$MAP_ROOT" \
    --manifest-path "$RENDERED_DATA_ROOT/driving_command_regeneration_from_solvability.json" \
    --commands-path "$RENDERED_DATA_ROOT/synthetic_token_driving_commands.json" \
    "${extra_args[@]}"
