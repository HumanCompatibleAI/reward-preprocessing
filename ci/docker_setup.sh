#!/bin/bash

# This command should be run when the "dependencies" tag
# of the Docker image is used to finish the setup.

# quit on errors, forbid undefined variables
set -euo pipefail

cd /reward_preprocessing

poetry run pip install -e /imitation
poetry run pip install -e .

# Download the Mujoco key
curl -o /root/.mujoco/mjkey.txt "$MUJOCO_KEY_URL"
# put the weights & biases credentials into .netrc
echo "$NETRC_CONTENTS" > /root/.netrc
# afterwards, give the user an interactive shell
# the --init-file hack immediately activates the virtual environment
exec /bin/bash --init-file <(echo "poetry shell")