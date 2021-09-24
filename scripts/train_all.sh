#!/bin/bash

# quit on errors, forbid undefined variables
set -euo pipefail

FAST=0
JOBS=1

CONFIGS=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --fast)
      FAST=1
      shift
      ;;
    -j|--jobs)
      JOBS=$2
      shift
      shift
      ;;
  esac
done

expert_cmds=$(mktemp)
airl_cmds=$(mktemp)
drlhp_cmds=$(mktemp)
trap "rm -f $expert_cmds" 0 2 3 15
trap "rm -f $airl_cmds" 0 2 3 15
trap "rm -f $drlhp_cmds" 0 2 3 15

echo Commands that will be run:

for size in 4 10; do
    for reward in goal path; do
        if [[ $FAST == 1 ]]; then
            echo "scripts/train.sh --expert empty_maze_$size $reward fast" | tee -a ${expert_cmds}
        else
            echo "scripts/train.sh --expert empty_maze_$size $reward" | tee -a ${expert_cmds}
        fi

        for shaping in unshaped dense antidense random; do
            if [[ $FAST == 1 ]]; then
                # the optimal policy is the same for each shaping
                ln -sf "./empty_maze_${size}_${reward}_fast" "results/expert_demos/empty_maze_${size}_${reward}_${shaping}_fast"
                echo "scripts/train.sh --airl empty_maze_$size $reward $shaping fast" | tee -a ${airl_cmds}
                echo "scripts/train.sh --drlhp empty_maze_$size $reward $shaping fast" | tee -a ${drlhp_cmds}
            else
                ln -sf "./empty_maze_${size}_${reward}" "results/expert_demos/empty_maze_${size}_${reward}_${shaping}"
                echo "scripts/train.sh --airl empty_maze_$size $reward $shaping" | tee -a ${airl_cmds}
                echo "scripts/train.sh --drlhp empty_maze_$size $reward $shaping" | tee -a ${drlhp_cmds}
            fi
        done
    done
done

echo Training DRLHP and experts
cat  ${expert_cmds} ${drlhp_cmds} | parallel -j $JOBS --progress

echo Training AIRL
# we need the expert results first
parallel -j $JOBS --progress < ${airl_cmds}

rm ${expert_cmds}
rm ${airl_cmds}
rm ${drlhp_cmds}