# base stage contains just dependencies.
FROM python:3.9.6-slim as dependencies
ARG DEBIAN_FRONTEND=noninteractive

RUN pip install pipenv && pip cache purge

WORKDIR /reward_preprocessing
# Copy only necessary dependencies to build virtual environment.
# This minimizes how often this layer needs to be rebuilt.
COPY ./setup.py ./setup.py
COPY ./setup.cfg ./setup.cfg
COPY ./pyproject.toml ./pyproject.toml
COPY ./src/reward_preprocessing/__init__.py ./src/reward_preprocessing/__init__.py
COPY ./Pipfile.lock ./Pipfile.lock
COPY ./Pipfile ./Pipfile
# clear the pipenv cache to keep the image a bit smaller
RUN pipenv sync --dev && pipenv --clear

# full stage contains everything.
# Can be used for deployment and local testing.
FROM dependencies as full

# Delay copying (and installing) the code until the very end
COPY . /reward_preprocessing
# Build a wheel then install to avoid copying whole directory (pip issue #2195)
RUN pipenv run python setup.py sdist bdist_wheel
RUN pipenv run pip install dist/reward_preprocessing-*.whl

# Default entrypoints
CMD ["pipenv", "run", "pytest"]
