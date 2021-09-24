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

mkdir -p fig

if [[ $DRLHP == 1 ]]; then
  for size in 4 10; do
    MODEL_PATHS=$(find processed/preference_comparisons -type f -path "*/empty_maze_$size*.pt" -printf "%P\n" | sed 's/\.[a-z0-9]*\.pt$//' | sort | uniq)
    path_list=$(echo $MODEL_PATHS | sed 's/\(\S*\)/"\1",/g')
    path_list="[$path_list]"
    echo $path_list
    poetry run python -m reward_preprocessing.plot_heatmaps with \
      env.empty_maze_$size \
      base_path=processed/preference_comparisons \
      "model_base_paths=$path_list" \
      save_path=fig/preference_comparisons_$size.pdf
  done
fi

if [[ $AIRL == 1 ]]; then
  for size in 4 10; do
    MODEL_PATHS=$(find processed/adversarial -type f -path "*/empty_maze_$size*.pt" -printf "%P\n" | sed 's/\.[a-z0-9]*\.pt$//' | sort | uniq)
    path_list=$(echo $MODEL_PATHS | sed 's/\(\S*\)/"\1",/g')
    path_list="[$path_list]"
    echo $path_list
    poetry run python -m reward_preprocessing.plot_heatmaps with \
      env.empty_maze_$size \
      base_path=processed/adversarial \
      "model_base_paths=$path_list" \
      save_path=fig/adversarial_$size.pdf
  done
fi

if [[ $TRUTH == 1 ]]; then
  for size in 4 10; do
    MODEL_PATHS=$(find processed/ground_truth -type f -path "*/empty_maze_$size*.pt" -printf "%P\n" | sed 's/\.[a-z0-9]*\.pt$//' | sort | uniq)
    path_list=$(echo $MODEL_PATHS | sed 's/\(\S*\)/"\1",/g')
    path_list="[$path_list]"
    echo $path_list
    poetry run python -m reward_preprocessing.plot_heatmaps with \
      env.empty_maze_$size \
      base_path=processed/ground_truth \
      "model_base_paths=$path_list" \
      save_path=fig/ground_truth_$size.pdf
  done
fi