#!/usr/bin/env bash
# Shared launcher for scripts/test_{cv,ltf,mlp}.sh: standalone synthetic evaluation
# (navsim/planning/script/run_pdm_score_synthetic.py) with NAVSIM-aligned
# five-metric scoring (NC, DAC, EP, TTC, comfort) and reactive IDM traffic.
# Comfort checks the simulated proposal; progress is gated by NC and DAC.
# Source this file; do not run it directly.
#
# Useful overrides:
#   PACKAGE_ROOT=path            Rendered package (default: $OPENSCENE_DATA_ROOT/navhazard).
#   TRAIN_TEST_SPLIT=name        Split built by build_navsim_package_split.py (default: navhazard).
#   METRIC_CACHE_PATH=path       Metric cache from scripts/metric_cache.sh (default: $PACKAGE_ROOT/metric_cache).
#   DRIVING_COMMANDS_PATH=path   Per-token one-hot driving commands (LTF/MLP only).
#   WORKER=ray_distributed       Hydra worker config.
#   EXPERIMENT_NAME=name         Output under $NAVSIM_EXP_ROOT/<name>/.
#   PYTHON_BIN=python            Interpreter of the NAVSIM environment.
# Extra Hydra overrides can be appended to the command line.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$PROJECT_ROOT/../dataset}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$PROJECT_ROOT/exp}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PACKAGE_ROOT="${PACKAGE_ROOT:-$OPENSCENE_DATA_ROOT/navhazard}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navhazard}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$PACKAGE_ROOT/metric_cache}"
WORKER="${WORKER:-ray_distributed}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    exit 1
fi

for required_path in "$PACKAGE_ROOT/navsim_config" "$PACKAGE_ROOT/synthetic_scene_pickles" \
    "$PACKAGE_ROOT/sensor_blobs" "$METRIC_CACHE_PATH"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Error: required path does not exist: $required_path" >&2
        echo "See docs/model_testing.md (split generation and metric caching)." >&2
        exit 1
    fi
done

# run_model_test <experiment name> <agent hydra overrides...> -- <user overrides...>
run_model_test() {
    local experiment_name="$1"
    shift
    local agent_args=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        agent_args+=("$1")
        shift
    done
    [[ $# -gt 0 ]] && shift

    exec "$PYTHON_BIN" navsim/planning/script/run_pdm_score_synthetic.py \
        "hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$PACKAGE_ROOT/navsim_config]" \
        "train_test_split=$TRAIN_TEST_SPLIT" \
        "worker=$WORKER" \
        "${agent_args[@]}" \
        "experiment_name=${EXPERIMENT_NAME:-$experiment_name}" \
        "metric_cache_path=$METRIC_CACHE_PATH" \
        "synthetic_sensor_path=$PACKAGE_ROOT/sensor_blobs" \
        "synthetic_scenes_path=$PACKAGE_ROOT/synthetic_scene_pickles" \
        scorer.config.five_metric_only=True \
        scorer._target_=navsim.planning.simulation.planner.pdm_planner.scoring.navtest_aligned_five_metric_scorer.NavtestAlignedFiveMetricScorer \
        "$@"
}

# Driving commands for LTF/MLP: replace the current ego-status command by the
# package's per-token four-value one-hot mapping when it exists.
driving_command_args() {
    local path="${DRIVING_COMMANDS_PATH:-$PACKAGE_ROOT/synthetic_token_driving_commands.json}"
    if [[ -f "$path" ]]; then
        echo "synthetic_driving_commands_path=$path"
    elif [[ -n "${DRIVING_COMMANDS_PATH:-}" ]]; then
        echo "Error: DRIVING_COMMANDS_PATH does not exist: $path" >&2
        return 1
    else
        echo "Warning: $path not found; using the driving commands stored in the scene pickles." >&2
    fi
}

require_checkpoint() {
    if [[ -z "${CHECKPOINT_PATH:-}" || ! -f "$CHECKPOINT_PATH" ]]; then
        echo "Error: set CHECKPOINT_PATH to an existing checkpoint (got: '${CHECKPOINT_PATH:-}')." >&2
        exit 1
    fi
}
