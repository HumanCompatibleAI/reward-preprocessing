#!/bin/bash

# quit on errors, forbid undefined variables, print commands
set -euxo pipefail

env_name="$1"
AGENT="results/agents/$env_name/model.zip"
ROLLOUTS="[(0, \"$AGENT\", \"expert\"), (1, None, \"random\"), (0.5, \"$AGENT\", \"mixed\")]"

exec xvfb-run poetry run python src/reward_preprocessing/create_rollouts.py with \
    "rollouts=$ROLLOUTS" \
    "env.$env_name" \
    "out_dir=results/rollouts/$env_name" \
    "${@:2}"