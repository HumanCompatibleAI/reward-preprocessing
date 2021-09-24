#!/bin/bash

# quit on errors, forbid undefined variables, print commands
set -euxo pipefail

env_name="$1"
AGENT="results/agents/$env_name/model.zip"
VALUE_NET="results/value_nets/$env_name/model.zip"
# MODEL="results/models/$env_name.pt"
ROLLOUT_PATH="results/rollouts/$env_name"
# ROLLOUTS="[(0, None, \"expert\", \"$ROLLOUT_PATH/expert.pkl\"), (0, None, \"random\", \"$ROLLOUT_PATH/random.pkl\"), (0, None, \"mixed\", \"$ROLLOUT_PATH/mixed.pkl\")]"
ROLLOUTS="[(0, None, \"expert\", \"$ROLLOUT_PATH/expert.pkl\"), (0, None, \"random\", \"$ROLLOUT_PATH/random.pkl\")]"
# ROLLOUTS="[(0, None, \"random\", \"$ROLLOUT_PATH/random.pkl\")]"
# ROLLOUTS="[(0, None, \"semiexpert\", \"$ROLLOUT_PATH/semiexpert.pkl\")]"
# ROLLOUTS="[(0, \"$AGENT\", \"expert\"), (1, None, \"random\"), (0.5, \"$AGENT\", \"mixed\")]"

exec xvfb-run poetry run python src/reward_preprocessing/interpret.py with \
    "value_net.path=$VALUE_NET" \
    "rewards.rollouts=$ROLLOUTS" \
    "optimize.rollouts=$ROLLOUTS" \
    "env.$env_name" \
    "${@:2}"