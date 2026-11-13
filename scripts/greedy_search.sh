#!/bin/bash

# Run hierarchical greedy search over every config/token pair in
# data/auto_reconstruction/summary.json with one omega_b/omega_s/omega_h setting.
# Default data paths are relative to the TriSafe root, so run from there.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python "Hierarchical Greedy Search/hgs_summary_exploration_direct_new.py" \
    --workers 6 \
    --total-iterations 1000 \
    --omega-b 0.3 \
    --omega-s 3.0 \
    --omega-h 3.0 \
    --summary-root "Hierarchical Greedy Search/K_9/summary_all" \
    --write-run-details
