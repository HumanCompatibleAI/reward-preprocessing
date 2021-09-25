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

ROLLOUT_PATH="results/expert_demos/seals_mountain_car/rollouts/final.pkl"
ROLLOUT="(0, None, \"expert\", \"$ROLLOUT_PATH\")"

mkdir -p fig/mountain_car

if [[ $DRLHP == 1 ]]; then
  MODEL_PATHS=$(find processed/preference_comparisons -type f -path "*/shaped_mountain_car*_medium*.pt" -printf "%P\n" | sed 's/\.[a-z0-9_]*\.pt$//' | sort | uniq)
  path_list=$(echo $MODEL_PATHS | sed 's/\(\S*\)/"\1",/g')
  path_list="[$path_list]"
  echo $path_list
  for mode in l1 log; do
    poetry run python -m reward_preprocessing.plot_reward_curves with \
      env.mountain_car \
      base_path=processed/preference_comparisons \
      "model_base_paths=$path_list" \
      save_path=fig/mountain_car/preference_comparisons_medium_$mode.pdf \
      "rollout_cfg=$ROLLOUT" \
      $mode
  done
fi

if [[ $AIRL == 1 ]]; then
  MODEL_PATHS=$(find processed/adversarial -type f -path "*/shaped_mountain_car*.pt" -printf "%P\n" | sed 's/\.[a-z0-9_]*\.pt$//' | sort | uniq)
  path_list=$(echo $MODEL_PATHS | sed 's/\(\S*\)/"\1",/g')
  path_list="[$path_list]"
  echo $path_list
  poetry run python -m reward_preprocessing.plot_reward_curves with \
    env.mountain_car \
    base_path=processed/adversarial \
    "model_base_paths=$path_list" \
    save_path=fig/mountain_car/adversarial.pdf \
    "rollout_cfg=$ROLLOUT"
fi

if [[ $TRUTH == 1 ]]; then
  MODEL_PATHS=$(find processed/ground_truth -type f -path "*/shaped_mountain_car*.pt" -printf "%P\n" | sed 's/\.[a-z0-9_]*\.pt$//' | sort | uniq)
  path_list=$(echo $MODEL_PATHS | sed 's/\(\S*\)/"\1",/g')
  path_list="[$path_list]"
  echo $path_list
  for mode in l1 log; do
    poetry run python -m reward_preprocessing.plot_reward_curves with \
      env.mountain_car \
      base_path=processed/ground_truth \
      "model_base_paths=$path_list" \
      save_path=fig/mountain_car/ground_truth_$mode.pdf \
      "rollout_cfg=$ROLLOUT" \
      $mode
  done
fi