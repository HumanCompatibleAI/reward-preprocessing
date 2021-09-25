#!/bin/bash

# quit on errors, forbid undefined variables
set -euo pipefail

DRLHP=0
AIRL=0
TRUTH=0
FAST=""
JOBS=1

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
    --fast)
      FAST="fast"
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
    -j|--jobs)
      JOBS=$2
      shift
      shift
      ;;
  esac
done

cmds=$(mktemp)
trap "rm -f $cmds" 0 2 3 15

ROLLOUT_PATH="results/expert_demos/seals_mountain_car/rollouts/final.pkl"
ROLLOUTS="[(0, None, \"expert\", \"$ROLLOUT_PATH\")]"

if [[ $DRLHP == 1 ]]; then
  mkdir -p processed/preference_comparisons
  MODEL_PATHS=$(find results/preference_comparisons -type f -path "*/shaped_mountain_car*/final_reward_net.pt" -printf "%P\n" | sed 's/\/final_reward_net\.pt$//')
  for path in $MODEL_PATHS; do
    if [[ path == *random ]]; then
      potential=mlp
    else
      potential=linear
    fi
    echo "Adding results/preference_comparisons/$path"
    echo "poetry run python -m reward_preprocessing.optimize_continuous with env.shaped_mountain_car $FAST model_path=results/preference_comparisons/$path/final_reward_net.pt $potential save_path=processed/preference_comparisons/$path 'rollouts=$ROLLOUTS'" >> $cmds
  done
fi

if [[ $AIRL == 1 ]]; then
  mkdir -p processed/adversarial
  MODEL_PATHS=$(find results/adversarial -type f -path "*/shaped_mountain_car*/checkpoints/final/reward_test.pt" -printf "%P\n" | sed 's/\/checkpoints\/final\/reward_test\.pt$//')
  for path in $MODEL_PATHS; do
    if [[ path == *random ]]; then
      potential=mlp
    else
      potential=linear
    fi
    echo "Adding results/adversarial/$path"
    echo "poetry run python -m reward_preprocessing.optimize_continuous with env.shaped_mountain_car $FAST model_path=results/adversarial/$path/checkpoints/final/reward_test.pt $potential save_path=processed/adversarial/$path 'rollouts=$ROLLOUTS'" >> $cmds
  done
fi

if [[ $TRUTH == 1 ]]; then
  mkdir -p processed/ground_truth
  MODEL_PATHS=$(find results/ground_truth_models -type f -path "*/shaped_mountain_car*.pt" -printf "%P\n" | sed 's/\.pt$//')
  for path in $MODEL_PATHS; do
    if [[ path == *random ]]; then
      potential=mlp
    else
      potential=linear
    fi
    echo "Adding results/ground_truth_models/$path"
    echo "poetry run python -m reward_preprocessing.optimize_continuous with env.shaped_mountain_car $FAST model_path=results/ground_truth_models/$path.pt $potential save_path=processed/ground_truth/$path 'rollouts=$ROLLOUTS'" >> $cmds
  done
fi

echo Running commands:
cat ${cmds}

parallel -j $JOBS --progress < ${cmds}

rm ${cmds}