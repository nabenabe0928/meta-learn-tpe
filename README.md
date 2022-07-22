# A repository template

[![Build Status](https://github.com/nabenabe0928/repo-template/workflows/Functionality%20test/badge.svg?branch=main)](https://github.com/nabenabe0928/repo-template)
[![codecov](https://codecov.io/gh/nabenabe0928/repo-template/branch/main/graph/badge.svg?token=FQWPWEJSWE)](https://codecov.io/gh/nabenabe0928/repo-template)

Before copying the repository, please make sure to change the following parts:
1. The name of the `repo_name` directory
2. `include` in `.coveragerc`
3. The URLs to `Build Status` and `codecov` (we need to copy from the `codecov` website) in `README.md`
4. Setting up the `codecov` of the repository
5. The token of `codecov.yml`
6. `Copyright` in `LICENSE`
7. `name`, `author`, `author email`, and `url` in `setup.py`
8. The targets of `.pre-commit-config.yaml` (Lines 8, 14)
9. `--cov=<target>` in Line 46 of `python-app.yml` (if there are multiple targets, use `--cov=<target 1> --cov=<target 2> ...`)
10. `target` in `check_github_actions_locally.sh`

## Local check

In order to check if the codebase passes Github actions, run the following:

```shell
$ pip install black pytest unittest flake8 pre-commit pytest-cov
$ ./check_github_actions_locally.sh
```
