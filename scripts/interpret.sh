#!/bin/bash

# quit on errors, forbid undefined variables, print commands
set -euxo pipefail

env_name="$1"

exec pipenv run python src/reward_preprocessing/interpret.py with \
    model_path="results/models/$env_name.pt" \
    agent_path="results/agents/$env_name" \
    sparsify.data_path="results/data/$env_name" \
    "env.$env_name" \
    "${@:2}"