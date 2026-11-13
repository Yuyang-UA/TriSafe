#!/usr/bin/env bash
# MTGS reconstruction, stage 4: MTGS training and rendering of the reconstruction.
# Runs for every config in CONFIG_DIR (see scripts/construction_common.sh for overrides).
# MTGS_METHOD: mtgs_st (default) or mtgs_deformable (better for pedestrians).
MTGS_METHOD="${MTGS_METHOD:-mtgs_st}"
source "$(dirname "${BASH_SOURCE[0]}")/construction_common.sh"
run_stage 4 --method "${MTGS_METHOD}"
