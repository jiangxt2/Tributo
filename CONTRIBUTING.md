# Contributing to Tributo

Thank you for contributing! Tributo is a telecom-native ML framework built on
Ray for PU Learning, behavioral sequence pre-training, and billion-scale
user look-alike.

## Getting Started

- **Python**: >=3.12,<3.14 (see `pyproject.toml`)
- **Package manager**: uv (see `pyproject.toml`)

```bash
git clone https://github.com/jiangxt2/tributo.git
cd tributo
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Development Workflow

1. Fork the repository and create a feature branch from `master`.
2. Make your changes, including tests for new functionality.
3. Run lint: `ruff check . && ruff format --check .`
4. Run unit tests: `uv run pytest tests/ -m "not integration and not slow"`
5. Commit with a clear message and `Signed-off-by` line.
6. Open a pull request against `master`.

## Pull Request Guidelines

- Keep PRs focused — one issue per PR.
- All new features must include tests.
- Public API additions require `@PublicAPI(stability=...)`.
- Follow the PR template (`.github/PULL_REQUEST_TEMPLATE.md`).
- All checks (lint, tests) must pass before merge.

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
Run `ruff check .` and `ruff format .` before pushing.

- Line length: 88
- Docstrings: Google-style
- Type annotations required on all public functions

## Reporting Bugs

Use the Bug Report template (`.github/ISSUE_TEMPLATE/bug_report.yml`).
Include: Tributo version, Python version, Ray version, and steps to reproduce.

## Feature Requests

Use the Feature Request template
(`.github/ISSUE_TEMPLATE/feature_request.yml`).

## DCO

All commits must be signed off: `Signed-off-by: Your Name <email@example.com>`.
We follow the [Developer Certificate of Origin](https://developercertificate.org/).

**Recommended**: Install the auto-signoff hook to append `Signed-off-by` automatically:

```bash
cp scripts/prepare-commit-msg .git/hooks/prepare-commit-msg
```
