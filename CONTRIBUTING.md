# Contributing to Tributo

Thank you for contributing! Tributo is a Ray-native machine learning SDK.
Core owns execution and model delivery; algorithm implementations are
maintained as independently installable packages.

## Getting started

- **Python**: 3.12 or 3.13 (`>=3.12,<3.14`)
- **Package manager**: uv (see `pyproject.toml`)

```bash
git clone https://github.com/jiangxt2/Tributo.git
cd Tributo
uv sync --locked --extra dev
uv run --locked --no-sync pre-commit install --hook-type pre-commit
```

The first `uv sync` may need access to the configured package index. Once the
environment is provisioned, repository checks use only the locked project
environment and do not install tools implicitly.

This prepares Core development dependencies. It does not install an official
algorithm Wheel. Use the [algorithm installation guide](docs/algorithms/getting-started.md)
when your change requires one. Follow the [installation guide](docs/getting-started/installation.md)
to select runtime extras, and retain all required extras when syncing an
environment.

## Development workflow

1. Fork the repository and create a feature branch from `master`.
2. Make your changes, including tests for new functionality.
3. Audit the test inventory and inspect the plan for your changed paths.
4. Run the selected bounded suites and any applicable external validation.
5. Run the repository precheck before review or push.
6. Commit with a clear message and `Signed-off-by` line.
7. Open a pull request against `master`.

Use `ci/test-suites.json` and `scripts/ci_test_plan.py` as the test-policy
source of truth. For example, inspect an installation-guide change:

```bash
python3 scripts/ci_test_plan.py audit
python3 scripts/ci_test_plan.py plan --event pull_request --mode pr \
  --changed-path docs/getting-started/installation.md
```

Pass each changed path separately. An unmatched path deliberately selects the
full fast matrix and reports external and quarantined suites for review.
The report is not authorization to run those suites.

Run a CI-authorized suite through the controlled runner:

```bash
python3 scripts/ci_test_plan.py run --suite policy --prepare
```

Replace `policy` with a selected `ci_fast` or applicable `ci_scheduled`
suite. The runner prepares its declared extras and enforces its budget.
Some suites also consume artifacts from their workflow's preceding steps.
For `documentation-api`, prepare the documentation dependencies and
real-import Sphinx site before running the suite. Follow the
[documentation guide](docs/developer/documentation.md) and the corresponding
[CI steps](.github/workflows/pr-test-suite.yml); `--prepare` does not build
that HTML artifact.
Do not use it for `manual_external` or `quarantine` suites. Docker,
databases, Ray Jobs, and multi-worker validation use their owned external
entry points after the environment, execution scope, logs, and cleanup are
agreed. A semantic pytest marker alone does not select or authorize a Gate.
See the [test execution policy](docs/developer/testing.md).

The full repository precheck is:

```bash
uv run --locked --no-sync python scripts/pr-precheck.py
```

It uses the existing checks, including the CI Python matrix and applicable
documentation gates. Its default is offline: prepare the declared
interpreters and locked dependencies first. Use `--allow-network` only when
you intend to permit its dependency resolution and environment preparation.
`--skip-tests` skips only the changed-test layer and is not the full precheck.

## Pull request guidelines

- Keep PRs focused — one issue per PR.
- All new features must include tests.
- Public API additions require `@PublicAPI(stability=...)`.
- Follow the PR template (`.github/PULL_REQUEST_TEMPLATE.md`).
- All checks (lint, tests) must pass before merge.

## Design proposals

Use the [`design-docs/`](design-docs/) process before implementation when a
change affects a public API or persisted contract, crosses component ownership
boundaries, introduces distributed failure or security semantics, adds an
execution engine or extension point, or requires compatibility and migration
decisions.

Start with a feature issue that establishes the problem and use cases. Changes
to the product boundary use a `[SCOPE]` issue as defined by
[`docs/architecture/product-scope.md`](docs/architecture/product-scope.md).
Then open a draft pull request containing only a proposal copied from
[`design-docs/template.md`](design-docs/template.md) and any supporting images.
Use line comments to review design details and keep unresolved decisions in the
proposal's open-questions section.

A maintainer must explicitly accept the design before its pull request merges.
Implement the accepted design in a separate pull request and link both records.
Acceptance approves the direction; it does not establish that a capability is
implemented or supported. Update architecture, API, support, and user
documentation when the behavior and its required evidence are delivered.

A design proposal is normally unnecessary for a contract-preserving bug fix,
local refactor, test addition, or documentation correction. See
[`design-docs/README.md`](design-docs/README.md) for the complete lifecycle,
status rules, review criteria, and document responsibilities.

## Code style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run
`uv run --locked --no-sync python scripts/pr-precheck.py` before
pushing. The precheck is repository-owned so local and CI checks use the same
implementation and locked dependencies.

- Line length: 88
- Docstrings: Google-style
- Type annotations required on all public functions

## Reporting bugs

Use the Bug Report template (`.github/ISSUE_TEMPLATE/bug_report.yml`).
Include: Tributo version, Python version, Ray version, and steps to reproduce.

## Feature requests

Use the Feature Request template
(`.github/ISSUE_TEMPLATE/feature_request.yml`).

## DCO

All commits must be signed off: `Signed-off-by: Your Name <email@example.com>`.
We follow the [Developer Certificate of Origin](https://developercertificate.org/).
