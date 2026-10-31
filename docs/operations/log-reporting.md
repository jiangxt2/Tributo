# Lifecycle log reporting

This feature is Alpha. Reporting is explicitly opt-in and disabled by default. It adds an independent
bounded branch to selected Python logging namespaces across an execution
domain. It does not tail files or forward worker console output a second time.
Ray native/C++ logs, stdout, prints, and records disabled at their source are
outside this Python logging branch. Namespace selection can include `ray`,
`ray_doris`, `ray_clickhouse`, and other installed Python libraries.

## Prepare dependencies and the destination

Use this source checkout's locked `log-reporting` extra. The implementation
pins OTel SDK and HTTP exporter 1.45.1; it owns a separate provider and does not
replace an application's global provider. A compatible OTLP HTTP/protobuf logs
receiver and a pre-created Doris log table are required.

The owner uses separate credential references for Stream Load and SQL readback.
Create a logs table with the schema required by the pinned
[Doris exporter](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/v0.163.0/exporter/dorisexporter).
Set `create_schema=false`; no task performs implicit DDL. Grant the readback
user SELECT only on that table, and the writer the required Stream Load
privileges. Inject credentials through the environment keys referenced below.
Do not put passwords or resolved headers in ordinary JSON configuration.

In managed mode, provision a platform-matching executable before running the
task and obtain its SHA256 from a trusted distribution manifest. Startup checks
its checksum, version, executable status and required component inventory.
Tasks never download, compile or install a Collector. The baseline is Contrib
0.163.2. A smaller distribution can be provisioned using official OCB:

```bash
python -m tools.build_reporting_collector \
  --output /absolute/new-build-directory --goos linux --goarch arm64
```

The checked-in `tools/reporting-collector.json` pins OTLP, Doris, memory limiter,
health check and file/env config providers. The builder and runtime are
separate tools. Retain generated `go.mod`, `go.sum`, binary and `manifest.json`
with the deployment artifact. The builder requires Go and network access;
the task does not. The trimmed build retains the upstream Doris exporter's
Alpha maturity and best-effort delivery semantics.

## Explicitly enable one execution domain

Write an owner configuration such as `/absolute/reporting.json`:

```json
{
  "mode": "managed",
  "sender": {
    "endpoint": "http://127.0.0.1:4318/v1/logs",
    "namespaces": ["tributo", "ray", "ray_doris", "ray_clickhouse"]
  },
  "collector": {
    "binary": "/absolute/tributo-otelcol",
    "sha256": "REPLACE_WITH_VERIFIED_SHA256",
    "otlp_bind": "127.0.0.1:4318",
    "health_bind": "127.0.0.1:13133"
  },
  "read_profile": {
    "host": "doris-fe.internal", "database": "telemetry", "table": "otel_logs",
    "username_ref": "env://REPORTING_READ_USER",
    "password_ref": "env://REPORTING_READ_PASSWORD"
  },
  "write_profile": {
    "host": "doris-fe.internal", "database": "telemetry", "table": "otel_logs",
    "http_endpoint": "http://doris-fe.internal:8030",
    "username_ref": "env://REPORTING_WRITE_USER",
    "password_ref": "env://REPORTING_WRITE_PASSWORD"
  }
}
```

Replace the checksum placeholder and supply all credential references through
your trusted process/deployment environment. Then explicitly enable reporting:

```bash
export TRIBUTO_LOG_REPORTING_ENABLED=1
export TRIBUTO_LOG_REPORTING_CONFIG=/absolute/reporting.json
python - <<'PY'
from tributo.observability import reporting_session
with reporting_session() as reporting:
    print(reporting.snapshot())
PY
```

The independent switch accepts only `0` or `1`. The file must not contain an
`enabled` key. When disabled, reporting ignores that file and OTEL conflicts
and starts no SDK/export thread, Collector or network check. A provided SDK
configuration instead sets `ReportingConfig(enabled=True, ...)` explicitly.

SDK callers can hold an explicit domain around eager and lazy work:

```python
import ray
from tributo.observability import reporting_session

with reporting_session() as reporting:
    # For an existing caller-owned runtime, configure this when first starting it.
    ray.init(address="local", runtime_env=reporting.runtime_env())
    try:
        # Execute the complete workload, including lazy materialization here.
        pass
    finally:
        ray.shutdown()
```

Framework CLI dispatch, local entrypoint execution, owned training/batch
entrypoints, algorithm execution and registry operations borrow the active
domain. Nested compatible sessions share a sender and gateway. Incompatible
concurrent configurations fail explicitly. An explicitly disabled nested
configuration also conflicts with an active domain. It cannot turn off its
enclosing owner's handler. Existing unrelated Ray workers are not retrofitted.
Use `reporting.runtime_env(existing_options)` at initial Ray
setup; it copies options and composes trusted existing setup hooks.

Core task and reducer submissions also pass a producer environment at their
owned execution boundary. They preserve the caller's existing setup hook and
initialize the sender before loading algorithm code, even when Ray was already
initialized. Without a producer policy, an owned worker does not inherit a
node's owner switch or start a gateway. Other user-created Ray tasks still
require explicit setup. Logs from an existing caller hook that executes before
the owned entry point are outside this initialization boundary.

Transport settings remain frozen per sender. A new explicit policy can bind
different operation/run/attempt identifiers for a model or training invocation
without changing the sender's process-wide identity. Each scope restores its
previous context, and absent identifiers are omitted rather than inherited
from another run. Model loading, prediction, cleanup and Train/Tune wrappers
use this scoped policy; cleanup does not initialize another sender.

## Local, remote Jobs and serving ownership

`run_local_entrypoint` starts the parent's gateway before its owned local Ray
runtime and child driver, and closes it after the child and Ray cleanup.
Worker hooks, Train wrappers and owned batch actors initialize process-local
senders, never another gateway or Doris probe reader.

A remote Ray Job is a separate execution domain. Its driver needs explicit
owner configuration and access to its own pre-provisioned binary and credential
references. The submitter's active SDK context is not a remote configuration;
its loopback endpoint is not propagated as an execution endpoint. A Job that
outlives submission must hold its own `reporting_session()` through completion.

For multiple nodes, use an explicitly reachable advertised sender endpoint and
receiver bind address. Plain HTTP across hosts requires `trusted_network=true`.
Use HTTPS and configured CA/client certificates across untrusted boundaries.
An explicit `headers_ref` resolves a JSON object of OTLP headers. Sender and
receiver TLS files must match. Referenced OTLP authorization may be propagated
to trusted worker runtime environments; owner SQL/write references and resolved
Doris credentials are excluded from the generated producer policy. Existing
credentials in a caller's node environment remain that caller's responsibility.

Managed Serve requires an actual persistent owner:

```python
from tributo.observability import reporting_session

with reporting_session(service_lifetime=True):
    # Deploy and serve while this owner remains alive.
    # Leaving the context deletes only newly owned applications, then drains.
    pass
```

A short-lived deployment CLI cannot own a detached service gateway. Use
`mode="external"` with an existing gateway for that case. External mode requires
the sender and read profile, starts no Collector, and never stops or changes
the external process. Managed ownership rejects pre-existing applications
instead of taking them over.

## Checks, data policy and failure semantics

Before business starts, the owner checks required configuration, sends a safe
unique probe through the same pipeline, and performs a bounded SQL readback.
Receiver health and OTLP ACK alone do not prove Doris ingestion. Producers
validate their frozen sender and OTLP ACK without receiving SQL credentials.
The local `reporting.checked` diagnosis belongs to an owner with SQL readback;
`producer.checked` means receiver acknowledgment only. These local diagnoses
include the role and are excluded from the reporting bridge itself.

Each new sender rejects nonempty generic/log-specific OTEL exporter endpoint,
headers, timeout, compression, certificate/key and credential-provider settings.
`OTEL_SDK_DISABLED=true` or invalid nonempty values also conflict. Unrelated
trace/metric configuration remains valid. Queue/resource/record limits are
explicit; the owned HTTP session does not read implicit proxies, netrc or CA
environment. Application environment and global telemetry objects are preserved.

The remote body is a validated event name, or `log.record` for an ordinary
message. Only safe bounded identifiers, numeric summaries and whitelisted status
fields are sent, marked `redacted=true`. Arbitrary message text, SQL, payloads,
exception messages and configuration dumps are excluded. Local logging retains
its existing formatting and levels; the reporting branch does not raise source
logger levels. SSE completion is emitted at iterator termination, with request
correlation held per request.

Streaming binds request context only while advancing the generator and while
recording its terminal event, then restores it before yielding to ASGI. Client
disconnects, send failures and cleanup in another task preserve the caller's
context. The underlying generator is closed under its request context; cleanup
errors are logged without replacing the disconnect or business result.

For a library logger with `propagate=False`, include that logger's exact
namespace explicitly. The bridge preserves this routing decision and avoids
duplicate handlers where records already reach a selected ancestor. It cannot
recover arbitrary records stopped by other descendant loggers.

Only HTTP 200 with valid protobuf is an OTLP success. Partial rejection retains
validated accepted/rejected counts and is never replayed as a whole batch.
Malformed/error bodies, 201/204 and redirects fail explicitly. Retryable
transport/status failures use the SDK's single bounded retry layer. Runtime
delivery failures make reporting visibly degraded and preserve business results.

`snapshot()` exposes offered records, adjacent OTLP acceptance/rejection,
failed batches, observed queue-overflow events and safe local diagnoses. It does
not claim per-record queue-loss precision or Doris commit totals. Finite flush
waits and owned-process shutdown bound caller waiting; pending in-memory logs
can be lost. Forced driver/node termination does not guarantee drain or absence
of an orphan. There is no exactly-once or durable-queue promise.

## Run the real acceptance suite

Provision a Collector matching Docker's platform, place it as
`otelcol-contrib` in a private binary directory, and run:

```bash
python -m tools.reporting_it \
  --binary-dir /absolute/preprovisioned-binaries \
  --log-dir /absolute/new-evidence-directory
```

The entrypoint uses a unique Compose project, shared locked runtime image,
two Ray nodes and isolated Doris FE/BE. It retains SQL evidence, exact source
snapshot identity, JUnit results and resource cleanup records. It needs Docker
access and must run outside a restrictive process/network sandbox.

## Run manually in GitHub Actions

After `log-reporting-e2e.yml` exists on the default branch, select **Log
Reporting E2E → Run workflow** and choose the branch to validate, or run:

```bash
gh workflow run log-reporting-e2e.yml --ref YOUR_BRANCH
```

The workflow provisions and checksum-verifies Contrib 0.163.2, then runs the
same eight cases through `tools.reporting_it --cases all` on Linux ARM64 with
Python 3.12. CI uses `--build-input-mode canonical`; local execution defaults
to `local-mirror`. Images remain digest-pinned in either mode.

Download the seven-day evidence artifact for JUnit, SQL records, service/build
logs and `result.json`. Success requires all cases without skips and no
`cleanup_errors`; failures retain diagnostics too. This workflow is manually
triggered only and is separate from the PR's required checks. It needs no
production database credentials.
