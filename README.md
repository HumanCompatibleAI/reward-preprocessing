[![CircleCI](https://circleci.com/gh/HumanCompatibleAI/reward-preprocessing/tree/main.svg?style=svg&circle-token=5689f087396d3f526afd49f3af9d4b098560f79c)](https://circleci.com/gh/HumanCompatibleAI/reward-preprocessing/tree/main)
# Reward preprocessing
## Installation
First clone this repository.
We use [`pipenv`](https://pipenv.pypa.io/en/latest/) to manage dependencies.
If you haven't installed it yet, you can use for example
```
pip install --user pipenv
```
Then you can reproduce the exact environment we use with
```
ci/setup.sh
```
(run this command inside the cloned repo). This will automatically create a new
virtual environment.
Use `pipenv shell` to start a shell inside this virtual environment.

## Running experiments
You can train an agent using
```
pipenv run python src/reward_preprocessing/train_agent.py
```
Arguments are passed using Sacred, e.g.
```
pipenv run python src/reward_preprocessing/train_agent.py with steps=10000
```
to train for 10000 steps.

The trained agent model and a video recording of the agent are stored via Sacred.

## Tests
Run
```
pipenv run pytest
```
to run all available tests (or just run `pytest` if the environment is active).

## Code style and linting
`ci/code_checks.sh` contains formatting, linting and type checks.
We recommend using it as a pre-commit hook:
```
ln -s ../../ci/code_checks.sh .git/hooks/pre-commit
```

To automatically change the formatting (rather than just checking it), run
```
ci/format_code.sh
```
Alternatively, you may want to configure your editor to run `black` and `isort` when saving a file.