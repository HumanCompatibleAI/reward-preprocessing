#!/bin/bash

# quit on errors, forbid undefined variables, print commands
set -euxo pipefail

env_name="$1"
AGENT="results/expert_demos/$env_name/policies/final/model.zip"
# ROLLOUTS="[(0, \"$AGENT\", \"expert\")]"
ROLLOUTS="[(1, None, \"random\")]"

exec xvfb-run poetry run python src/reward_preprocessing/create_rollouts.py with \
    "rollouts=$ROLLOUTS" \
    "env.$env_name" \
    "out_dir=results/rollouts/$env_name" \
    "${@:2}"