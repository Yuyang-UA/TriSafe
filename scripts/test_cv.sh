#!/usr/bin/env bash
# Evaluate the constant-velocity agent on a rendered TriSafe package with
# NAVSIM-aligned five-metric scoring. See scripts/model_testing_common.sh for overrides.
#
# Example:
#   PYTHON_BIN=/path/to/envs/navsim/bin/python bash scripts/test_cv.sh

source "$(dirname "${BASH_SOURCE[0]}")/model_testing_common.sh"

run_model_test constant_velocity_trisafe \
    agent=constant_velocity_agent \
    -- "$@"
