#!/bin/bash

# quit on errors, forbid undefined variables, print commands
set -euxo pipefail

env_name="$1"
ROLLOUT_PATH="results/rollouts/$env_name"
ROLLOUTS="[(0, None, \"expert\", \"$ROLLOUT_PATH/expert.pkl\"), (0, None, \"random\", \"$ROLLOUT_PATH/random.pkl\"), (0, None, \"mixed\", \"$ROLLOUT_PATH/mixed.pkl\")]"

exec xvfb-run poetry run python src/reward_preprocessing/create_densities.py with \
    "rollouts=$ROLLOUTS" \
    "env.$env_name" \
    "out_dir=results/rollouts/$env_name" \
    "${@:2}"