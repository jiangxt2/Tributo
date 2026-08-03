# Contributing to Tributo

Thank you for contributing! Tributo is a telecom-native ML framework built on
Ray for PU Learning, behavioral sequence pre-training, and billion-scale
user look-alike.

## Getting Started

- **Python**: 3.10+
- **Package manager**: uv (see `pyproject.toml`)

```bash
git clone https://github.com/jiangxt2/tributo.git
cd tributo
uv sync --extra dev --locked
```

The first `uv sync` may need access to the configured package index. Once the
environment is provisioned, repository checks use only the locked project
environment and do not install tools implicitly.

## Development Workflow

1. Fork the repository and create a feature branch from `master`.
2. Make your changes, including tests for new functionality.
3. Run the repository precheck: `uv run --locked --no-sync python scripts/pr-precheck.py --skip-tests`
4. Run unit tests: `uv run --locked --no-sync pytest tests/ -m "not integration and not slow and not minio_compat and not ray_runtime_env"`
5. Run the MinIO compatibility gate when Docker is available:
   `uv run --locked --no-sync pytest tests/integration/test_minio_compat.py -m minio_compat`
6. Run the Ray runtime-environment gate:
   `RAY_ENABLE_UV_RUN_RUNTIME_ENV=1 uv run --locked --no-sync pytest tests/integration/test_ray_runtime_env.py -m ray_runtime_env`
7. Commit with a clear message and `Signed-off-by` line.
8. Open a pull request against `master`.

## Pull Request Guidelines

- Keep PRs focused — one issue per PR.
- All new features must include tests.
- Public API additions require `@PublicAPI(stability=...)`.
- Follow the PR template (`.github/PULL_REQUEST_TEMPLATE.md`).
- All checks (lint, tests) must pass before merge.

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run
`uv run --locked --no-sync python scripts/pr-precheck.py --skip-tests` before
pushing. The precheck is repository-owned so local and CI checks use the same
implementation and locked dependencies.

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
