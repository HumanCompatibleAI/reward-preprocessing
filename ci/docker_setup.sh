#!/bin/bash

# This command should be run when the "dependencies" tag
# of the Docker image is used to finish the setup.

# quit on errors, forbid undefined variables
set -euo pipefail

cd /reward_preprocessing

# Build a wheel then install to avoid copying whole directory (pip issue #2195)
# Note that all dependencies were already installed in the previous stage.
# The purpose of this is only to make the local code available as a package for
# easier import.
pipenv run python setup.py sdist bdist_wheel
pipenv run pip install dist/reward_preprocessing-*.whl

# Download the Mujoco key
curl -o /root/.mujoco/mjkey.txt "$MUJOCO_KEY_URL"
# put the weights & biases credentials into .netrc
echo "$NETRC_CONTENTS" > /root/.netrc
# afterwards, give the user an interactive shell
# the --init-file hack immediately activates the virtual environment
exec /bin/bash --init-file <(echo "pipenv shell")