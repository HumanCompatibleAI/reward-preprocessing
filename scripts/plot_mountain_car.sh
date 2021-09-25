#!/bin/bash

# quit on errors, forbid undefined variables
set -euo pipefail

DRLHP=0
AIRL=0
TRUTH=0

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --airl)
      AIRL=1
      shift
      ;;
    --ground-truth)
      TRUTH=1
      shift
      ;;
    --drlhp)
      DRLHP=1
      shift
      ;;
    --all)
      DRLHP=1
      AIRL=1
      TRUTH=1
      shift
      ;;
  esac
done

mkdir -p fig/mountain_car

if [[ $DRLHP == 1 ]]; then
  echo ""
fi

if [[ $AIRL == 1 ]]; then
  echo ""
fi

if [[ $TRUTH == 1 ]]; then
  MODEL_PATHS=$(find processed/ground_truth -type f -path "*/mountain_car_*.pt" -printf "%P\n" | sed 's/\.[a-z0-9]*\.pt$//' | sort | uniq)
  path_list=$(echo $MODEL_PATHS | sed 's/\(\S*\)/"\1",/g')
  path_list="[$path_list]"
  echo $path_list
  poetry run python -m reward_preprocessing.plot_reward_curves with \
    env.mountain_car \
    base_path=processed/ground_truth \
    "model_base_paths=$path_list" \
    'objectives=["unmodified"]' \
    save_path=fig/mountain_car/ground_truth.pdf \
    'rollout_cfg=(1, )'
fi