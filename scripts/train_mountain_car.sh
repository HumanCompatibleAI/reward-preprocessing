#!/bin/bash

# quit on errors, forbid undefined variables
set -euo pipefail

FAST=0
JOBS=1
DRLHP=0
AIRL=0
EXPERT=0

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
    --all)
      DRLHP=1
      EXPERT=1
      AIRL=1
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

if [[ $FAST == 1 ]]; then
    if [[ $EXPERT == 1 ]]; then
        echo "scripts/train.sh --expert seals_mountain_car fast" | tee -a ${expert_cmds}
    fi
    if [[ $AIRL == 1 ]]; then
        echo "scripts/train.sh --airl seals_mountain_car fast" | tee -a ${airl_cmds}
    fi
else
    if [[ $EXPERT == 1 ]]; then
        echo "scripts/train.sh --expert seals_mountain_car" | tee -a ${expert_cmds}
    fi
    if [[ $AIRL == 1 ]]; then
        echo "scripts/train.sh --airl seals_mountain_car" | tee -a ${airl_cmds}
    fi
fi

if [[ $DRLHP == 1 ]]; then
    for shaping in unshaped dense antidense random; do
        if [[ $FAST == 1 ]]; then
            echo "scripts/train.sh --drlhp shaped_mountain_car $shaping fast" | tee -a ${drlhp_cmds}
        else
            echo "scripts/train.sh --drlhp shaped_mountain_car $shaping" | tee -a ${drlhp_cmds}
        fi
    done
fi

echo Training expert
parallel -j $JOBS --progress < ${expert_cmds}

echo Training DRLHP and AIRL
# we need the expert results first
cat  ${drlhp_cmds} ${airl_cmds} | parallel -j $JOBS --progress

rm ${expert_cmds}
rm ${airl_cmds}
rm ${drlhp_cmds}
