#!/bin/bash

# quit on errors, forbid undefined variables
set -euo pipefail

DRLHP=0
AIRL=0
EXPERT=0

CONFIGS=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --airl)
      AIRL=1
      shift
      ;;
    --expert)
      EXPERT=1
      shift
      ;;
    --drlhp)
      DRLHP=1
      shift
      ;;
    *)    # unknown option
      CONFIGS+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${CONFIGS[@]}" # restore positional parameters

# separate with spaces for passing to Sacred
config_args=$(printf " %s" "${CONFIGS[@]}")
config_args=${config_args:1}

# and separate with underscores for filenames
config_path=$(printf "_%s" "${CONFIGS[@]}")
config_path=${config_path:1}

if [[ $DRLHP == 1 ]]; then
    poetry run python -m imitation.scripts.train_preference_comparisons with \
        $config_args \
        "log_dir=results/preference_comparisons/$config_path"
fi

if [[ $EXPERT == 1 ]]; then
    poetry run python -m imitation.scripts.expert_demos with \
        $config_args \
        "log_dir=results/expert_demos/$config_path"
fi

if [[ $AIRL == 1 ]]; then
    poetry run python -m imitation.scripts.train_adversarial with \
        airl \
        $config_args \
        "rollout_path=results/expert_demos/$config_path/rollouts/final.pkl" \
        "log_dir=results/adversarial/$config_path"
fi