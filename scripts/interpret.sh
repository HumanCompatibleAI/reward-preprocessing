#!/bin/bash

# quit on errors, forbid undefined variables, print commands
set -euxo pipefail

env_name="$1"

if [[ "$env_name" == half_cheetah ]]; then
    MODEL_TYPE=sas
else
    MODEL_TYPE=ss
fi

exec pipenv run python src/reward_preprocessing/interpret.py with \
    model_path="results/models/$env_name.pt" \
    agent_path="results/agents/$env_name" \
    sparsify.data_path="results/data/$env_name" \
    model_type="$MODEL_TYPE" \
    "env.$env_name" \
    "${@:2}"