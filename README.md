# Reward preprocessing
## Installation
First clone this repository.
To work with this project, we recommend using [`pipenv`](https://pipenv.pypa.io/en/latest/):
```
pip install --user pipenv
```
Then you can reproduce the exact environment we use with
```
pipenv sync --dev
```
(run this command inside the cloned repo; it will automatically create a new
virtual environment). Remove the `--dev` flag if you don't need the development
dependencies (formatter, testing, ...).

To effectively activate the `pipen` environment, run `pipenv shell`.
Run `exit` to deactivate it.

Alternatively, you can also use `pip`. Install this project and all its
dependencies in editable mode using
```
pip install -e .
```
This will not install the development dependencies. You might also get somewhat
different package versions than those in the lockfile.

## Tests
Assuming you use `pipenv`, run
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
Unfortunately, this currently only works when using `pipenv`
(as a workaround to [this VS Code issue](https://github.com/microsoft/vscode-python/issues/10165)).

To automatically change the formatting (rather than just checking it), run
```
ci/format_code.sh
```
Alternatively, you may want to configure your editor to run `black` and `isort` when saving a file.