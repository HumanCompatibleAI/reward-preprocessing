#!/bin/bash

# Use this script to get interactive access to a
# Docker container. $REWARD_PREPROCESSING_DIR must
# be the path to the repository, it will be mounted
# into the Docker container.

# quit on errors, forbid undefined variables
set -euo pipefail

sudo docker run \
    --rm -it \
    --env MUJOCO_KEY_URL="$MUJOCO_KEY_URL" \
    --mount type=bind,src="$REWARD_PREPROCESSING_DIR",target=/reward_preprocessing \
    ejenner/reward_preprocessing:dependencies \
    ci/docker_setup.sh