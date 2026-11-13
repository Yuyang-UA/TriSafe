#!/usr/bin/env bash
# Evaluate Latent TransFuser on a rendered TriSafe package with NAVSIM-aligned
# five-metric scoring. Driving commands come from the package's per-token JSON
# mapping when present. See scripts/model_testing_common.sh for overrides.
#
# Example:
#   CHECKPOINT_PATH=/path/to/ltf_seed_0.ckpt PYTHON_BIN=/path/to/envs/navsim/bin/python \
#       bash scripts/test_ltf.sh

source "$(dirname "${BASH_SOURCE[0]}")/model_testing_common.sh"
require_checkpoint
commands_arg="$(driving_command_args)"

run_model_test latent_transfuser_trisafe \
    agent=transfuser_agent \
    agent.config.latent=True \
    "agent.checkpoint_path=$CHECKPOINT_PATH" \
    ${commands_arg:+"$commands_arg"} \
    -- "$@"
