#!/bin/bash

# Use this script to get interactive access to a
# Docker container. $REWARD_PREPROCESSING_DIR must
# be the path to the repository, it will be mounted
# into the Docker container.
# You may pass additional arguments to this script
# which will be passed as options to docker run
# (e.g. --cpus=2 --gpus 'device=2').

# quit on errors, forbid undefined variables
set -xeuo pipefail

# we pass on the .netrc contents into the container,
# so that we're already logged in to weights & biases
docker run \
    --rm -it \
    --env MUJOCO_KEY_URL="$MUJOCO_KEY_URL" \
    --env NETRC_CONTENTS="$(cat $HOME/.netrc)" \
    --mount type=bind,src="$REWARD_PREPROCESSING_DIR",target=/reward_preprocessing \
    --mount type=bind,src="$IMITATION_DIR",target=/imitation \
    "$@" \
    ejenner/reward_preprocessing:dependencies \
    ci/docker_setup.sh
