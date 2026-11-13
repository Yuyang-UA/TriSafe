#!/usr/bin/env bash
# MTGS reconstruction, stage 3: RGB point-cloud stacking.
# Runs for every config in CONFIG_DIR (see scripts/construction_common.sh for overrides).

source "$(dirname "${BASH_SOURCE[0]}")/construction_common.sh"
run_stage 3
