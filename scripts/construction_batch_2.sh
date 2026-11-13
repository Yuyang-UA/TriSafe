#!/usr/bin/env bash
# MTGS reconstruction, stage 2: dense depth with UniDepth.
# Runs for every config in CONFIG_DIR (see scripts/construction_common.sh for overrides).

source "$(dirname "${BASH_SOURCE[0]}")/construction_common.sh"
run_stage 2
