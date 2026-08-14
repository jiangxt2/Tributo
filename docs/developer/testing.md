# Test execution policy

Tributo separates fast repository checks from environment-owned integration
validation. The policy follows Ray's practical split between small pull-request
tests, budgeted scheduled shards, and explicitly owned external suites. A test
is never promoted into GitHub Actions merely because it is named an
integration test; its dependencies, runtime, evidence, and cleanup contract
must be bounded first.

The authoritative inventory is `ci/test-suites.json`.
`scripts/ci_test_plan.py` validates that inventory, selects affected suites,
and is the only entry point used by the PR and nightly test matrices.

## Execution tiers

| Tier | Purpose | GitHub Actions behavior |
| --- | --- | --- |
| `ci_fast` | Unit tests and small self-owned contracts | May run on PR, push, and merge queue events within a 12-minute suite budget |
| `ci_scheduled` | Broader optional-dependency regression shards | May run only from the scheduled workflow; each shard has a 10-minute execution budget |
| `manual_external` | Docker, Ray Jobs, multi-worker, database, model-download, or other environment-owned validation | Reported as required external evidence; never executed by a GitHub Actions event |
| `quarantine` | Tests without a reliable lifecycle, deterministic fixture, or accountable owner | Reported when affected and excluded from all automated execution |

The scheduled shard budgets sum to 30 minutes. Workflow timeouts are larger
only to allow checkout and locked dependency preparation; the suite runner
enforces the narrower test-command budget itself.

## Selection and fallback

Changed paths are evaluated against the manifest's ordered path rules. The
first matching rule supplies the affected domains, and suites select those
domains or declare exact trigger paths. This makes precedence easy to review and
avoids multiple workflow-local path maps.

Selection fails safe:

- A merge-queue event selects every `ci_fast` suite.
- An unknown path or unresolved Git diff selects every `ci_fast` suite and
  reports every external and quarantined suite for review.
- A documentation-only change runs the policy and documentation suites without
  installing or running the unit matrix.
- A source or test-policy change selects the relevant bounded suites and emits
  an external-validation impact report in the job summary.

The impact report is a validation-ledger requirement, not authorization to run
its commands. External suites still require an approved environment and the
project's long-running-test discipline.

## Marker policy

The manifest assigns collection-time execution markers through
`tests/conftest.py`:

- `ci_safe` identifies a bounded integration contract that may run in CI.
- `manual_it` identifies an external suite that must not run in CI.
- `quarantine` identifies a test excluded from automation; quarantined tests
  also receive `manual_it`.

The repository default deselects `manual_it` and `quarantine`. The manifest
audit and controlled runner prevent every automated suite from selecting
either marker, including the policy suite, which uses an exact audited target
rather than a marker expression. The PR unit suite additionally excludes
`ci_safe` and `s3_contract`. In PR CI, bounded local integration contracts and
S3 cases run once in their dedicated Python 3.12 shards instead of being
repeated by both unit Python versions. Nightly scheduled suites may
intentionally exercise the same cases again as part of broader regression
coverage. The manifest runner rejects a request to execute either non-CI tier
before starting a subprocess.

Some modules contain both ordinary unit cases and marker-selected bounded
contracts. Such a module keeps its default owner, while the manifest must name
exactly one controlled CI suite whose argument list selects the governed
marker. This shared-selector rule currently applies to `s3_contract`; it
prevents duplicate execution without forcing unrelated unit cases into the S3
shard.

Existing semantic markers such as `integration`, `distributed`,
`ray_runtime_env`, `minio_compat`, and `tributo_walking_skeleton` remain useful
for describing test behavior. They do not independently authorize CI
execution.

## Required evidence

Every suite declaration records:

- one accountable owner and one execution tier;
- a structured argument list rather than a shell command string;
- test ownership and impact paths;
- required infrastructure and locked dependency extras;
- a hard execution budget;
- a log or result-evidence contract;
- the reason for its current tier.

Automated suites must use the exact controlled `python -m pytest` entry point.
The audit and the runner both reject arbitrary programs and shell entry points
before dependency preparation or test subprocess creation.

All `scripts/run_*_it.sh` entry points must have exactly one
`manual_external` owner. Integration-sensitive test modules must have an
explicit owner; ordinary unit modules use the one declared default owner. A
governed shared marker must instead have exactly one controlled selector in
the manifest.

Required CI-safe suites cannot silently pass by skipping. Suites marked
`forbid_skips` emit XML test evidence, and the runner fails if required tests are
missing or skipped. Infrastructure absence is therefore either a suite failure
or a reason to classify the suite as external, never green evidence.

## Local checks

Audit the inventory, budgets, markers, path rules, and workflows without
installing project dependencies:

```bash
python3 scripts/ci_test_plan.py audit
```

Inspect the PR plan for explicit changed paths:

```bash
python3 scripts/ci_test_plan.py plan \
  --event pull_request \
  --mode pr \
  --changed-path src/tributo/data/reader.py
```

Run a CI-authorized suite through its locked environment and hard budget:

```bash
python3 scripts/ci_test_plan.py run --suite policy --prepare
```

Run the repository PR check before review:

```bash
uv run --locked --no-sync python scripts/pr-precheck.py
```

Do not call `run` for an external suite. Use the lifecycle-owned command shown
in the impact report only after recording the suite in the long-running test
ledger and obtaining any required Docker or infrastructure authorization.

## Adding or moving a test

When a new test needs only the default unit environment, no explicit manifest
entry is required. Add an explicit suite owner when the module name or path is
integration-sensitive, when it declares a governed marker, or when it has a
distinct environment or evidence contract.

Before assigning `ci_fast` or `ci_scheduled`, verify all of the following:

- the test owns every service it starts and uses no shared fixed port;
- dependencies are locked and require no external credentials or model
  download;
- the suite is deterministic, bounded, and produces traceable failure
  evidence;
- cleanup is exact-project scoped;
- the declared budget is representative and enforced by the runner;
- the workflow contains no direct test-path or external-runner bypass.

If any condition is not met, use `manual_external` with a lifecycle-owned
runner, or `quarantine` with a concrete reliability rationale. Update the
manifest first; the policy tests will reject orphaned modules, duplicate
owners, IT runners without owners, forbidden CI infrastructure, and workflow drift.
