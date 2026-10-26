# Opt-in lifecycle log reporting with a managed OTLP Collector

| Attribute | Value |
| --- | --- |
| Status | Draft |
| Authors | @jiangxt2 |
| Decider | @jiangxt2 |
| Related issue | [#139](https://github.com/jiangxt2/Tributo/issues/139) |
| Supersedes | None |

> This proposal describes planned behavior. It does not establish that reporting, Collector management, or a tested dependency combination is available.

## Summary

Add an explicitly enabled, logs-only output path across Tributo's existing lifecycle: Python logging records flow through a process-local OpenTelemetry sender to an owned or explicitly external OTLP Collector, then through the upstream Doris Exporter to a pre-existing Doris log table. The default remains disabled. Local runs do not require a pre-existing Ray cluster or node-level file-tail agents.

The owner manages configuration, startup validation, gateway ownership, and bounded cleanup. Workers only send to the appropriate gateway. Reuse upstream protocol, batching, and Doris export components; do not implement another general-purpose Collector or a database data-plane writer in Tributo.

## Motivation and use cases

Users need to query logs while local SDK/CLI jobs, Ray Jobs, training/Tune workers, batch inference, and persistent Serve applications are running. A failure can occur before the first model operation, during data access or export, or during cancellation and cleanup. A cluster-file collection assumption misses local execution and runtimes provisioned through Tributo.

The initial design targets existing application trust boundaries and explicitly configured Doris access. Public, multi-tenant ingress, high availability, and forced-termination zero-loss requirements have not been established. Their absence does not remove credential isolation, bounded resource use, failure visibility, or explicit stop ownership.

## Goals

- Support an independent opt-in, required-configuration checks, and an end-to-end startup probe with Doris readback.
- Cover startup/configuration/plugins, Jobs/runtime, data I/O and ETL, Core algorithms, training/Tune/checkpoints, export/publication/hooks, registry/import, batch and HTTP/gRPC/SSE inference, explainability, indexing, failure/cancellation, and cleanup.
- Preserve application-owned logging, existing identities, business results, and Ray/Torch diagnostic settings.
- Default enabled runs to one lightweight managed Collector per outer execution domain, with explicit external reuse as an alternative.
- Make partial rejection, invalid OTLP success responses, configuration conflicts, queue overflow, and incomplete shutdown visible.

## Non-goals

- General data access, a dashboard, host/node collection, metrics or trace pipelines, and connector dependencies on Tributo.
- Capturing every native Ray/PyTorch daemon or arbitrary stdout/stderr message.
- Replacing OperationStore, Outbox, write receipts, or business status with log-delivery status.
- Automatic task-time downloads, dependency installation, compilation, image pulls, DDL, or persistent system-service changes.
- Exactly-once delivery, zero loss after SIGKILL, or a new HA/Kafka deployment.

## Current behavior and evidence

At the implementation baseline, the package logging helper supports text/JSON output and selected CLI/Job entry points configure it. It does not provide a reporting session, a managed Collector, a reporting environment policy, or a Doris end-to-end probe. The existing helper clears package handlers; preserving handler ownership and adding safe stage events are tracked separately in [#135](https://github.com/jiangxt2/Tributo/issues/135).

The main integration boundaries are CLI bootstrap, TributoClient submission, runtime environment construction, local child drivers, Core runtime adapters, training worker wrappers, inference actors, and Serve replica construction. A parent Ray initialization does not configure a subprocess's separate driver. Existing application runtimes and previously launched workers cannot be retroactively reconfigured.

The repository declares Ray 2.59.0. Under the owner's explicit local implementation exception recorded below, the reporting extra now locks OpenTelemetry SDK/API/proto and HTTP exporter 1.45.1 with Prometheus exporter 0.66b1. Real protocol, Collector/Doris and representative Ray validation has been completed locally. Collector Contrib 0.163.2 and a minimal OCB distribution use the selected upstream 0.163.0 components and 1.69.0 file/env providers. This Draft does not imply formal proposal acceptance or a published release; current Alpha contracts and boundaries are documented in the source, stability inventory and operations guide.

## Proposed design

### Sessions and ownership

Provide an Alpha explicit SDK reporting context for SDK-owned execution and invoke the same bootstrap from Tributo-owned task/CLI entry points. Ordinary imports remain free of sender threads, subprocesses, probes, or network activity. Nested operations borrow the existing owner. Each producer owns only its process-local sender.

Freeze the validated policy once per execution domain. Use a separate internal reporting policy, not the local logging-level policy, to carry enabled state, role, endpoint, service identity, bounded budgets, and credential references. Do not serialize active handlers, providers, connections, or subprocess objects. Resolve credentials through trusted deployment configuration; do not log or place resolved headers/passwords in ordinary task payloads.

| Mode | Owner | Producers |
| --- | --- | --- |
| Local SDK/CLI or directly owned Ray | The outer execution context owns a local Collector | Nested work, child drivers, and owned workers borrow its address |
| Local child entry point | Parent context owns the gateway until child and owned-resource cleanup finishes | Child configures its own driver and Ray Job hook |
| Remote Ray Job, including provisioned targets | Submission and remote execution contexts own separate resources | The remote driver validates its actual execution environment and advertises a worker-reachable address |
| Supported no-wait remote submission | The remote driver retains the execution gateway | Submitter exit releases only its local context; existing managed-runtime wait restrictions remain |
| Existing application runtime | Explicit current-process and later owned-entry integration | Do not re-init Ray or claim coverage of already-running external workers |
| Persistent Serve | A service-lifetime owner or an explicitly external gateway | Replicas borrow an endpoint and retain request-scoped context |

The existing deployment helper returns an application name; a temporary deployment context is not a persistent owner. Detached deployments without a service-lifetime owner must reject managed mode before deployment and require an explicit external selection. This still covers persistent service logs through the external mode. Introducing a new durable service host is deferred; it is not silently emulated by an orphaned Collector or a CLI-only reporting session.

### Configuration and startup

Propose independent TRIBUTO_LOG_REPORTING_ENABLED=0|1 and TRIBUTO_LOG_REPORTING_MODE=managed|external controls. Disabled is the default; mode defaults to managed only after enabling. DEBUG, an endpoint, connector usage, or unrelated OTEL environment variables must not enable the feature. Disabled runs ignore irrelevant reporting configuration and do not import optional reporting dependencies or start resources.

The public configuration will define bounded sender/Collector queues, record/request sizes, startup/send/retry/shutdown deadlines, logger selection, endpoints, and trusted credential references. Managed owners require a pre-provisioned verified binary and a Doris write profile. Both owner modes require a separate read-only Doris validation profile; producers require neither Doris credentials nor a local binary.

Validate database/table identifiers, HTTP/SQL endpoints, certificate and credential references, finite budgets, and role-specific requirements without echoing sensitive inputs. Log tables already exist and create_schema is false. Startup sends a unique safe probe through the same sender/receiver/exporter pipeline and performs a bounded parameterized readback from the target log table. Health/TCP/OTLP acceptance alone is insufficient. Failure prevents initial business execution and releases newly owned resources.

### Lightweight Collector delivery

First validate the pipeline using a pinned upstream Contrib binary. Then use OCB to assemble only the OTLP HTTP/protobuf receiver, Doris exporter, necessary memory/health components, and env/yaml configuration providers. The Contrib 0.163.2 manifest references the relevant components at 0.163.0 and env/yaml providers at 1.69.0; distribution and module versions must not be conflated.

Provision OS/architecture/checksum-verified binaries during installation, image preparation, or a separate build step. Runtime only checks and starts them. No user-side Go compiler, Docker requirement, automatic download, or system-service installation is introduced by task startup. Do not claim that deleting YAML pipeline entries shrinks an upstream binary.

Use loopback for local execution. Distributed execution requires an explicit bind/advertise distinction and a reachable address with the authentication/TLS settings appropriate to its actual trust boundary. Do not silently expose a public anonymous receiver or disable certificate verification. External mode never starts, reconfigures, or stops the external gateway.

### Frozen policy and ambient OpenTelemetry settings

For the referenced SDK, explicit headers do not eliminate automatically merged environment headers, unset certificate arguments fall back to environment settings, and even a separate LoggerProvider reads OTEL_SDK_DISABLED. Validate each process before constructing SDK objects.

Reject non-empty generic/log-specific OTEL exporter endpoint, headers, timeout, compression, CA, client certificate/key, and HTTP credential-provider environment settings for the Tributo sender. These values must instead come from its explicit frozen configuration and trusted credential references. Reject SDK_DISABLED=true or invalid non-empty values when reporting is enabled; false/unset/empty is allowed. Do not mutate global environment variables or private SDK fields.

Explicitly supply all batch budgets, Resource attributes, and global/log-specific record limits. Use a non-publishing meter provider rather than the application's global metrics pipeline. Create an owned requests Session with trust_env=false, explicit TLS/authentication and any explicitly needed proxy. Do not borrow application sessions or ambient netrc, CA bundle, or proxy settings. Unrelated traces/metrics-specific configuration remains the application's property. Disabled reporting does not perform these checks.

Initial conflicts fail before business execution. A conflict in a producer created during an already-running workload makes its reporting failure visible and prevents a false ready state, while retaining the default that runtime reporting failure does not cancel business execution.

### OTLP response adapter

Compose a LogRecordExporter with the upstream OTLPLogExporter using its public session parameter. Keep upstream serialization/compression and its bounded transport retries. A controlled Session validates the response before returning it to the SDK; do not replace private exporter client/transport fields or copy the upstream exporter.

The referenced SDK otherwise treats 200 through 399 as success without decoding ExportLogsServiceResponse. Require HTTP 200, application/x-protobuf, and a valid response within a decompressed 4 MiB parsing bound. Read in bounded chunks under transport/outer deadlines and close responses. An empty byte sequence is a valid empty Protobuf message. Preserve Protobuf unknown-field compatibility; successful parsing is not proof of Doris commit or of an honest receiving service.

| Response | Interpretation | Retry |
| --- | --- | --- |
| Valid 200, unset/empty partial_success | Entire batch accepted by the adjacent OTLP receiver | No replay |
| Valid 200, rejected_log_records greater than zero and no greater than batch size | Preserve accepted/rejected counts and report partial/all rejection | Never resend the whole batch |
| Valid 200, zero rejected records and a non-empty warning | Full acceptance with a safe warning category | No replay |
| Invalid 200 body/type, malformed/oversized Protobuf, invalid rejection count, 201/204/3xx | Protocol failure; acceptance is unknown | No replay or redirect following |
| Retryable HTTP/connection failure | Existing SDK classification under the frozen budget | One bounded retry layer only |

Use safe non-connection protocol exceptions at the Session boundary and retain a per-export result context. The SDK's binary FAILURE and failed-batch metrics do not mean that every record was rejected. Do not expose raw response bodies, arbitrary error_message text, headers, or endpoint credentials in diagnostics. Contract tests must exercise real exporter/Session calls and count HTTP requests.

### Lifecycle integration and safe records

Use standard logging and an explicit, idempotent sender handler for tributo and selected third-party Python namespaces. Do not raise source logger levels, override application formatters/filters, forward driver copies a second time, or route Collector/exporter diagnostics back into the same pipeline.

Preserve existing operation/run/attempt/submission/trial/request/worker/rank identifiers. Process service identity is a resource; changing run/request/stage context is record-scoped and reset at scope exit. Fixed safe events and approved simple attributes may be exported. Legacy/third-party messages that cannot be shown safe are represented by a redacted category, not arbitrary formatted messages or exception dumps. No new Dataset actions, database business queries, tensor conversions, collectives, algorithm calls, or publication operations are performed for logs.

Audit every goal's existing module boundary for producer initialization, safe context, and actual log evidence. A global handler alone is not proof of lifecycle coverage. Missing event-generation work remains visible and associated with #135; do not copy uncommitted work from that worktree or make this output path redefine business statuses.

## Public contracts and stability

The explicit SDK reporting context and typed reporting configuration are Alpha APIs. The owner explicitly waived the proposal PR prerequisite for this local implementation; the current signatures, configuration/profile representation, fixed limits and log-reporting extra are implemented and documented. Formal review may still request Alpha changes. Ordinary import behavior, existing stable execution APIs, connector independence, JSON-first configuration, and default local logging remain compatible.

The reporting policy is an internal bounded startup format, separate from business payloads and the local logging policy. Reject unsupported policy fields/versions safely rather than executing imports or accepting arbitrary logging configuration. Worker payloads contain only the producer subset and credential references. Logs are observational records, not a replacement persisted business contract.

A Doris table example and exporter mapping must match the actually verified exporter version. Provisioning/DDL is an explicit installation or test-fixture activity, never an implicit task-side operation.

## Failure and security semantics

Reporting state is disabled/starting/ready/degraded/draining/closed or failed, separate from business status. Bound memory, record length, retries, send time, startup, and cleanup. Runtime connection loss, gateway exit, partial rejection, or overflow produces bounded local diagnostic evidence and known counters; default training/inference results do not change. Unknown acceptance/unflushed counts stay unknown.

Drain producers after their business terminal/cleanup logs, release owned Ray resources, perform the final driver flush, then drain and stop only the owned Collector. Borrowers release only their sender; external gateways are never terminated. Nested/concurrent ownership is explicit, not inferred from a port or PID. Control configuration-file permissions and inject credentials through protected references/environment rather than command-line arguments or logged configuration.

The researched SDK's force_flush timeout is not an established deadline, and the batch processor does not replay rejected batches. Use an outer shutdown deadline and real hanging/disconnect tests. Ray/OS process-group cleanup and SIGKILL can prevent final draining; do not use detachment/setsid to evade Ray's lifecycle. Persistent Collector queues are optional only when restart recovery is required and do not recover records still in SDK memory.

## Compatibility, deprecation, and migration

No reporting resources exist by default and no existing connector gains a Tributo dependency. Users explicitly enable reporting and provision the required profiles, log table, and verified binary, or explicitly select an external gateway. Disabling reporting restores the existing execution path without migrating business data.

Resolve the optional reporting dependency set against the repository lock and Ray dependency closure before claiming a supported combination. Do not assume the researched SDK version is compatible with the currently locked Prometheus exporter. Do not remove the application-owned OpenTelemetry installation or alter unrelated dependency groups to force a resolution.

Use the same startup/record contracts for the upstream validation binary and eventual OCB build. A binary or SDK change invalidates only the relevant compatibility/protocol/resource evidence; it does not justify rerunning unaffected long suites.

## Alternatives considered

| Alternative | Benefit | Reason not selected as the default |
| --- | --- | --- |
| Node/file-tail agents | Collects existing cluster files | Requires cluster/node ownership and misses local/SDK execution; separate optional operations solution |
| External-only Collector | Reuses shared operations infrastructure | Does not meet the requested lightweight self-starting local path; retain as explicit mode |
| Direct Doris writes in every worker | Removes a Collector process | Distributes database credentials and duplicates batching/Stream Load failure logic |
| Custom in-process OTLP server | Can reduce packaging size | Adds protocol, security, buffering, and lifecycle maintenance; reuse upstream components |
| Ray Actor as the gateway | Uses a familiar execution primitive | Cannot bootstrap before Ray or cover non-cluster startup; unnecessary runtime dependency |
| No change | No added resources | Does not provide timely, correlated Doris log visibility across execution modes |

## Test plan and acceptance criteria

All entries below are planned. No implementation or infrastructure test is claimed by the proposal.

| Gate | Required evidence | Boundary and deduplication |
| --- | --- | --- |
| T-UP-ENABLE/CFG | Strict opt-in, defaults, missing/invalid configuration, role-specific credentials, handler/provider coexistence, no optional imports/threads/processes/network when disabled | Bounded unit/process tests |
| T-UP-OTLP | Real SDK/Session against controlled HTTP responses: valid empty Protobuf, partial rejection/warning, invalid 200 bodies/types/counts, oversized responses, 201/204 and 302/307/308, bounded 429/503 and connection failures | Count actual requests; no final-result mock or whole partial-batch retry |
| T-UP-ENV | Separate driver/worker environments for headers/certificates/credential provider/SDK_DISABLED; explicit budgets/resources/limits; unrelated telemetry and caller environment preserved | Construct actual SDK objects in separate processes |
| T-UP-LIFE | Safe actual records at every lifecycle boundary listed in Goals | Reuse minimal existing business cases; no second business execution for logging |
| T-UP-MODES | Local without an existing cluster, real child driver, Jobs submission/execution separation, no-wait, existing runtime, and persistent service ownership/external | Ray Data evidence comes from owned Docker Ray Jobs, not macOS Ray Client |
| T-UP-E2E | Real Collector and isolated Doris: probe write/readback, schema/permission/FE-BE routing, managed/external and startup failure | Receiver ACK or a fake Doris server is insufficient |
| T-UP-RUN/CLOSE | Overflow, rejection, disconnect/hang, gateway crash, fork/repeated/nested initialization, cancellation, external/borrower ownership, bounded flush and orphan evidence | Do not change original business exceptions or results |
| T-UP-LIGHT | Pinned component/binary/checksum inventory, task-time no installation, one Collector per owner, startup/size/RSS/CPU/queryable latency/cleanup measurements | Compare the same workload with reporting disabled/enabled; measure both upstream and OCB artifacts |
| Repository checks | Owned test inventory, applicable static/type/docs/API checks, final repository precheck | Run only suites justified by the changed code and fixed test matrix |

Before a long or external test, record its exact entry point, code/configuration/environment state, log location, purpose, and overlap in the implementation ledger. Run each final complete suite once per relevant state. Use unique owned Compose projects, build a custom runtime image once, start Compose with no-build, retain logs, and clean only precisely owned resources. Use pinned domestic registry mirrors for image operations with all proxy environment variants cleared. GitHub operations use the project's separate proxy policy.

## Open questions

Local implementation update: the SDK combination, bounded defaults, public Alpha session/configuration API and minimal OCB composition have been selected and validated under the recorded exception. The questions below remain formal review/longer-term compatibility questions; they are not an outstanding block on the authorized local development and do not mean that current implementation results are absent.

- Decider @jiangxt2: accept the exact Alpha SDK/session/configuration contract and complete profile schema before source implementation. This proposal defines the required behavior but does not infer a public API from the local planning document.
- Decider @jiangxt2: select the resolved SDK/HTTP exporter/Prometheus-compatible optional dependency combination; 1.45.1 is a reference until resolution and protocol contracts confirm the choice.
- Decider @jiangxt2: confirm supported binary platforms, distribution/checksum provisioning, default resource/timeout budgets, and the real Collector/Doris schema mapping. Measurements may tune budgets without weakening disabled/startup/ownership/credential guarantees.
- Deferred with a fixed boundary: a new durable detached Serve host, disk queue, HA, metrics/traces, and native-process capture are not required for this initial design; explicit external mode covers services without a persistent managed owner.

The first three questions block Accepted status. They do not authorize task-time dependency downloads, silent configuration fallback, a narrower lifecycle scope, or an unverified support claim.

## Decision outcome

The project owner explicitly waived the proposal-PR prerequisite and authorized local implementation and testing on 2026-10-23. Commit, push, and PR creation remain unauthorized. This document stays Draft as a design record; the local exception is not represented as a GitHub review or merged proposal. Implementation evidence is tracked separately from this workflow exception.

## Implementation tracking

Track the requested capability in [#139](https://github.com/jiangxt2/Tributo/issues/139), with event-generation/configuration foundations separately tracked in [#135](https://github.com/jiangxt2/Tributo/issues/135). No implementation pull request exists as part of this proposal. Architecture/API/support documentation must change only with verified implementation evidence.

## Review checklist

- [x] The proposal distinguishes implemented behavior from planned behavior.
- [x] Goals, non-goals, ownership boundaries, and proposed stability are explicit.
- [x] Bounded ingestion and unbounded streaming lifecycles remain separate; this proposal adds only an output path.
- [x] New bounded sources use Provider and Binding contracts without consumer source branches. N/A: no bounded source is added.
- [x] New inference outputs use ResultSink and WriteGateway contracts without a pipeline-owned data writer. N/A: no inference sink is added.
- [x] New algorithms declare their AlgorithmSpec, data-loading responsibility, execution strategy, and Bundle output. N/A: no algorithm is added.
- [x] Model formats identify exporter, validator, flavor, and loader responsibilities without format-string guessing. N/A: no model format is added.
- [x] Credential isolation and fail-closed startup behavior cover success and failure paths.
- [x] Compatibility, rollback, and required real-infrastructure evidence are defined.
- [x] Support claims require implementation, configuration entry points, and corresponding tests.
- [ ] Blocking public-contract, dependency, packaging, and budget questions are accepted by the decider.
- [ ] The maintainer records a formal design decision and required ADR before the implementation PR.

## References

- [Contribution workflow](../CONTRIBUTING.md) and [proposal lifecycle](README.md).
- [Ray RuntimeEnv](https://docs.ray.io/en/latest/ray-core/api/doc/ray.runtime_env.RuntimeEnv.html).
- [OpenTelemetry Python 1.45.1 HTTP client](https://github.com/open-telemetry/opentelemetry-python/blob/v1.45.1/exporter/opentelemetry-exporter-otlp-common/src/opentelemetry/exporter/otlp/common/http.py).
- [OpenTelemetry HTTP log exporter and public session parameter](https://github.com/open-telemetry/opentelemetry-python/blob/v1.45.1/exporter/opentelemetry-exporter-otlp-proto-http/src/opentelemetry/exporter/otlp/proto/http/_log_exporter/__init__.py).
- [OpenTelemetry headers/TLS resolver](https://github.com/open-telemetry/opentelemetry-python/blob/v1.45.1/exporter/opentelemetry-exporter-otlp-proto-http/src/opentelemetry/exporter/otlp/proto/http/_common/__init__.py).
- [OpenTelemetry LoggerProvider and record limits](https://github.com/open-telemetry/opentelemetry-python/blob/v1.45.1/opentelemetry-sdk/src/opentelemetry/sdk/_logs/_internal/__init__.py).
- [Collector Contrib distribution manifest](https://github.com/open-telemetry/opentelemetry-collector-releases/blob/v0.163.2/distributions/otelcol-contrib/manifest.yaml), [Doris Exporter](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/v0.163.0/exporter/dorisexporter/README.md), and [OCB](https://opentelemetry.io/docs/collector/extend/ocb/).
- [OTLP response and delivery semantics](https://opentelemetry.io/docs/specs/otlp/).
