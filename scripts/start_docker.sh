#!/bin/bash

# Use this script to get interactive access to a
# Docker container. $REWARD_PREPROCESSING_DIR must
# be the path to the repository, it will be mounted
# into the Docker container.
# You may pass additional arguments to this script
# which will be passed as options to docker run
# (e.g. --cpus=2 --gpus 'device=2').

# quit on errors, forbid undefined variables
set -euo pipefail

docker run \
    --rm -it \
    --env MUJOCO_KEY_URL="$MUJOCO_KEY_URL" \
    --mount type=bind,src="$REWARD_PREPROCESSING_DIR",target=/reward_preprocessing \
    "$@" \
    ejenner/reward_preprocessing:dependencies \
    ci/docker_setup.sh
