#!/usr/bin/env bash

# If you change these, also change .circle/config.yml.
SRC_FILES=(src/ tests/ setup.py)

set -x  # echo commands
set -e  # quit immediately on error

pipenv run flake8 ${SRC_FILES[@]}
pipenv run black --check ${SRC_FILES[@]}
pipenv run pytype ${SRC_FILES[@]}