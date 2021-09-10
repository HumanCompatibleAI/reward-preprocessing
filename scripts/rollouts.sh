#!/bin/bash

# quit on errors, forbid undefined variables, print commands
set -euxo pipefail

env_name="$1"

# rollout config: use expert, 50-50 split of expert/random actions,
# and purely random actions, in an even mix
exec poetry run python src/reward_preprocessing/create_rollouts.py with \
    "rollouts=[(0, \"results/agents/$env_name\", \"expert\"), (0.5, \"results/agents/$env_name\", \"mixed\"), (1, \"\", \"random\")]" \
    save_path="results/data/$env_name" \
    "env.$env_name" \
    "${@:2}"