#!/usr/bin/env bash
# Build the metric cache of a rendered TriSafe package for model testing.
# The output ($PACKAGE_ROOT/metric_cache) is the default METRIC_CACHE_PATH of
# scripts/test_{cv,ltf,mlp}.sh.
#
# Example:
#   PACKAGE_ROOT=path/to/rendered_data bash scripts/metric_cache.sh
#
# Useful overrides:
#   PACKAGE_ROOT=path        Rendered package (default: $OPENSCENE_DATA_ROOT/navhazard).
#   TRAIN_TEST_SPLIT=name    Split built by build_navsim_package_split.py (default: navhazard).
#   METRIC_CACHE_PATH=path   Output directory (default: $PACKAGE_ROOT/metric_cache).
#   WORKER=sequential        Hydra worker config.
#   PYTHON_BIN=python        Interpreter of the NAVSIM environment.

set -euo pipefail

# Repository root (the TriSafe directory).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$PROJECT_ROOT/../dataset}"
PACKAGE_ROOT="${PACKAGE_ROOT:-$OPENSCENE_DATA_ROOT/navhazard}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navhazard}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$PACKAGE_ROOT/metric_cache}"
WORKER="${WORKER:-sequential}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" navsim/planning/script/run_metric_caching.py \
    "hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$PACKAGE_ROOT/navsim_config]" \
    "train_test_split=$TRAIN_TEST_SPLIT" \
    "synthetic_scenes_path=$PACKAGE_ROOT/synthetic_scene_pickles" \
    "metric_cache_path=$METRIC_CACHE_PATH" \
    "worker=$WORKER" \
    "$@"
