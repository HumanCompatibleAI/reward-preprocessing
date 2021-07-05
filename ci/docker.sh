#!/bin/bash

set -e
set -x

sudo docker build --target dependencies -t ejenner/reward_preprocessing:dependencies .
sudo docker build -t ejenner/reward_preprocessing .
# Just a simple check for whether the built Docker container works at all.
# If it doesn't, we stop before pushing to DockerHub.
sudo docker run --rm ejenner/reward_preprocessing pipenv run pytest tests/test_envs.py
sudo docker push ejenner/reward_preprocessing
sudo docker push ejenner/reward_preprocessing:dependencies