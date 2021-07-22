# base stage contains just dependencies.
FROM python:3.9.6-slim as dependencies
ARG DEBIAN_FRONTEND=noninteractive

# without tini, xvfb-run hangs
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

RUN apt-get update && apt-get install -y --no-install-recommends \
    # ffmpeg is needed to capture videos
    ffmpeg \
    # xvfb is needed to run the unit tests on a headless server
    # (e.g. for CI)
    xauth \
    xvfb \
    # MountainCar and other gym environments use GLX for rendering,
    # which needs this library to work
    libglu1-mesa \
    # git is needed by Sacred
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pipenv && pip cache purge

WORKDIR /reward_preprocessing
# Copy only necessary dependencies to build virtual environment.
# This minimizes how often this layer needs to be rebuilt.
COPY ./Pipfile.lock ./Pipfile.lock
# clear the pipenv cache to keep the image a bit smaller
RUN pipenv sync --dev && pipenv --clear
# clear the directory again (this is necessary so that CircleCI can checkout
# into the directory)
RUN rm Pipfile.lock Pipfile

# full stage contains everything.
# Can be used for deployment and local testing.
FROM dependencies as full

# Delay copying (and installing) the code until the very end
COPY . /reward_preprocessing
# Build a wheel then install to avoid copying whole directory (pip issue #2195)
# Note that all dependencies were already installed in the previous stage.
# The purpose of this is only to make the local code available as a package for
# easier import.
RUN pipenv run python setup.py sdist bdist_wheel
RUN pipenv run pip install dist/reward_preprocessing-*.whl

# Default entrypoints
ENTRYPOINT ["/tini", "--", "xvfb-run"]
CMD ["pipenv", "run", "pytest"]
