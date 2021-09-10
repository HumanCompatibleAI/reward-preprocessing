# base stage contains just dependencies.
FROM python:3.9.6-slim as dependencies
ARG DEBIAN_FRONTEND=noninteractive

# without tini, xvfb-run hangs
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--", "xvfb-run"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    # needed for the Mujoco installation commands below
    curl \
    unzip \
    # needed for mujuco_py installation
    gcc \
    libc6-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
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

# patchelf is needed for mujoco_py
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

RUN mkdir -p /root/.mujoco \
    && curl -o mujoco.zip https://www.roboti.us/download/mujoco200_linux.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm mujoco.zip \
    && touch /root/.mujoco/mjkey.txt
# the Mujoco key file needs to exist for building mujoco_py

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
ENV PATH="/root/.local/bin:$PATH"


WORKDIR /reward_preprocessing
# Copy only necessary dependencies to build virtual environment.
# This minimizes how often this layer needs to be rebuilt.
COPY poetry.lock pyproject.toml ./
# Clear the cache immediately. Annoyingly, there doesn't seem to be a good
# option for confirming, so we just pipe yes into it
# (see https://github.com/python-poetry/poetry/issues/521)
RUN poetry install --no-interaction --no-ansi && yes | poetry cache clear . --all
# HACK: make sure we have all the dependencies of imitation but then uninstall
# imitation itself. We want to use the git version of that and that changes too
# frequently to include in this stage
RUN poetry run pip install imitation && poetry run pip uninstall --yes imitation && poetry run pip cache purge

# clear the directory again (this is necessary so that CircleCI can checkout
# into the directory)
RUN rm poetry.lock pyproject.toml

# full stage contains everything.
# Can be used for deployment and local testing.
FROM dependencies as full

# Delay copying (and installing) the code until the very end
COPY . /reward_preprocessing
RUN poetry run pip install git+https://github.com/HumanCompatibleAI/imitation
# Build a wheel then install to avoid copying whole directory (pip issue #2195)
# Note that all dependencies were already installed in the previous stage.
# The purpose of this is only to make the local code available as a package for
# easier import.
RUN poetry run python setup.py sdist bdist_wheel
RUN poetry run pip install dist/reward_preprocessing-*.whl

CMD ["poetry", "run", "pytest"]
