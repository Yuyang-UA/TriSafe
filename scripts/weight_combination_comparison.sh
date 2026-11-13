#!/bin/bash

# Batch runner for "Hierarchical Greedy Search"/hgs_summary_exploration_direct_random.py
# Reads omega_b/omega_s/omega_h combinations from parameter_new.txt and executes
# the script for each line. All combinations share one sampling seed, so every
# combination is evaluated on exactly the same config/token pairs.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
PARAM_FILE="${PARAM_FILE:-$PROJECT_ROOT/Hierarchical Greedy Search/parameter_new.txt}"
PYTHON_SCRIPT="Hierarchical Greedy Search/hgs_summary_exploration_direct_random.py"
# One sampling seed for every parameter combination, so all combinations are
# evaluated on exactly the same config/token pairs. Override it per batch:
# HGS_SEED=12345 bash run_hgs_random_new.sh
SEED="${HGS_SEED:-70}"
WORKERS="${HGS_WORKERS:-10}"

if [[ ! -f "$PARAM_FILE" ]]; then
    echo "Error: parameter file not found: $PARAM_FILE"
    exit 1
fi

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines
    [[ -z "${line// }" ]] && continue

    # Read the three parameters from the line
    read -r omega_b omega_s omega_h <<< "$line"

    summary_root=""

    echo "Running with omega_b=$omega_b omega_s=$omega_s omega_h=$omega_h seed=$SEED workers=$WORKERS"
    echo "Summary root: $summary_root"

    python "$PYTHON_SCRIPT" \
        --workers "$WORKERS" \
        --total-iterations 1000 \
        --omega-b "$omega_b" \
        --omega-s "$omega_s" \
        --omega-h "$omega_h" \
        --seed "$SEED" \
        --summary-root "$summary_root" \
        --write-run-details

    echo "Finished run for $summary_root"
    echo "----------------------------------------"
done < "$PARAM_FILE"
