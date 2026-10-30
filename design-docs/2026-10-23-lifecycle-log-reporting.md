# Opt-in lifecycle log reporting with a managed OTLP Collector

| Attribute | Value |
| --- | --- |
| Status | Draft |
| Authors | @jiangxt2 |
| Decider | @jiangxt2 |
| Related issue | [#139](https://github.com/jiangxt2/Tributo/issues/139) |
| Supersedes | None |

## Summary

Add an explicitly enabled logs output: selected Python logging namespaces → process-local OpenTelemetry sender → owned or explicitly external Collector → upstream Doris Exporter → existing log table. Local SDK/CLI execution must work before Ray starts. Tributo owns configuration and lifecycle integration; upstream components own protocol encoding, batching, retries and Doris writes.

## Motivation and use cases

Users need correlated logs while local tasks, Ray Jobs, training/Tune, batch inference and persistent Serve applications run. A node-file collection requirement misses local processes and runtimes started through Tributo. The scope follows existing application trust boundaries; public multi-tenant ingress and HA requirements have not been established.

## Goals

- Explicit default-off control, necessary configuration checks and a complete-path startup probe with bounded Doris readback.
- Existing startup/configuration/plugins, data/ETL, Core, training/checkpoints/Tune, export/hooks/registry, inference, explainability/indexing, failure and cleanup boundaries.
- Safe correlation, preserved caller logging/providers/Ray options, bounded resources and visible delivery failure without changing business results.

## Non-goals

General data access, host/native-process collection, metrics/traces, dashboards, another Collector implementation, task-time installation/download/build, implicit DDL, a durable detached-service host, HA/Kafka and forced-termination zero-loss guarantees. Event generation/configuration remains separately tracked in [#135](https://github.com/jiangxt2/Tributo/issues/135).

## Current behavior and evidence

The pre-feature baseline provides local text/JSON logging but no managed reporting session or Doris probe. Implementation [#144](https://github.com/jiangxt2/Tributo/pull/144) adds an Alpha path using the locked Ray 2.59.0 and OTel SDK/HTTP 1.45.1 dependency closure. The source, tests, [stability inventory](../docs/STABILITY.md) and [operations guide](../docs/operations/log-reporting.md) describe the implemented contract; this Draft is not formal acceptance or release evidence.

## Proposed design

An outer execution owner freezes ReportingConfig, validates trusted credential references, optionally starts one pre-provisioned Collector, creates its sender and verifies a unique safe probe in Doris before business starts. Nested operations borrow the session. Each worker owns only its process-local sender; serialized producer policy contains bounded sender configuration and safe invocation identities, never owner SQL profiles or live resources.

| Execution | Ownership |
| --- | --- |
| Local SDK/CLI, locally owned Ray and child driver | Outer context keeps the Collector alive through child/owned-runtime cleanup; each driver initializes its own sender |
| Remote Ray Job, including no-wait submission | Remote driver owns and validates its execution gateway; never propagate submitter loopback as a remote endpoint |
| Caller-initialized Ray | Preserve options/hooks and caller ownership; configure later owned entries without claiming to retrofit existing external workers |
| Persistent Serve | Explicit service-lifetime owner or external gateway; replicas preserve per-request context |

Reuse the pinned upstream Contrib distribution, or provision an OCB build containing only OTLP HTTP/protobuf, Doris, memory limiter, health check and file/env providers. Verify executable/platform/version/checksum/components before startup. Runtime does not require Go, Docker, downloads or compilation. External mode never manages the gateway. Distributed producers require an explicitly reachable advertised endpoint and authentication/TLS appropriate to the actual trust boundary.

## Public contracts and stability

The Alpha API is tributo.observability.reporting_session and its handle/configuration models. TRIBUTO_LOG_REPORTING_ENABLED accepts only 0/1 and defaults off; enabled mode defaults managed, with external explicitly selected. Disabled runs ignore irrelevant reporting configuration and create no sender, Collector, probe or reporting network activity.

Transport configuration stays frozen; record context changes per invocation and restores absent identifiers rather than inheriting another run. Model load/predict/close, Train/Tune and HTTP/gRPC/SSE share these rules. Cleanup must not create a sender. Managed Serve requires reporting_session(service_lifetime=True); exit deletes only newly registered owned applications before draining the gateway. A temporary deployment context cannot own a detached service.

## Failure and security semantics

Only approved bounded identifiers/numeric summaries are exported; unrestricted messages, exception text, SQL, credentials and samples are excluded. Preserve caller handlers, formatting, levels, global providers and environment. Prevent recursion and duplicate routing.

Each sender rejects nonempty generic/log-specific OTEL endpoint/header/timeout/compression/certificate/key and credential-provider overrides, plus conflicting OTEL_SDK_DISABLED values. Explicit SDK limits/resource/meter settings and an owned trust_env=False HTTP session prevent ambient settings from silently replacing policy. Unrelated trace/metric settings remain valid.

Compose the SDK through its public session parameter. Require HTTP 200, protobuf media type and a bounded valid ExportLogsServiceResponse. Empty protobuf is valid; partial rejection preserves verified counts and never replays the whole batch. Zero-rejection warnings remain successful. Invalid/error/oversized bodies, invalid rejection counts, 201/204 and redirects are protocol failures; redirects are not followed. Transient failures use only the SDK's bounded retry layer. OTLP acceptance is not a Doris commit.

Initial owner failure blocks business and releases newly owned resources. Dynamic producer or export failure remains visible without replacing business results/exceptions. Queues, requests, retries and caller flush/shutdown waits are finite; terminate only owned processes. SSE scopes bind during generator advancement/close, restore caller context and emit one terminal event across disconnect/cancellation/cleanup errors. In-memory loss and incomplete drain remain explicit limitations.

## Compatibility, deprecation, and migration

Reporting is an optional extra and an independent output branch. Existing local logging and business APIs remain usable when off; connectors acquire no Tributo dependency. Enabling requires existing Doris schema, separate owner read/write references and a verified binary or explicit external gateway. No business-data migration or new storage protocol is introduced.

## Alternatives considered

| Alternative | Assessment |
| --- | --- |
| Existing node/file agents | Useful where node ownership exists; insufficient for local SDK and self-provisioned execution |
| External-only Collector | Keep as explicit reuse; does not satisfy requested self-starting local mode |
| Direct worker Doris writes | Distributes database credentials and duplicates the upstream exporter |
| Custom OTLP server or Ray gateway Actor | Adds maintenance or a Ray bootstrap dependency; reuse Collector instead |
| No change | Does not provide timely cross-process Doris visibility |

## Test plan and acceptance criteria

| Gate | Required evidence |
| --- | --- |
| Enable/configuration/privacy | Default-off imports/resources, strict opt-in, owner/producer credentials, caller handler/provider/environment preservation, safe fields |
| OTLP/SDK environment | Actual SDK with real controlled HTTP: empty/full/partial/warning, invalid 200/types/counts/size, 2xx/3xx, bounded retries and request counts; separate driver/worker environment conflict tests |
| Lifecycle/context | Existing entry boundaries, nesting/concurrency, A/B/A and missing IDs, Train/Tune/kernel, SSE disconnect/cross-task close and one terminal |
| Real modes and Doris | No-Ray managed/external/probe failure/runtime loss, child driver, remote Jobs, two-node task/actor/Train/Tune/Serve and caller-initialized Core; actual SQL rows and owned cleanup |
| Lightweight behavior | Fixed component/binary inventory and same-workload off/on startup/RSS/CPU/queryability/close evidence; report upstream and OCB artifacts separately |
| Repository integration | Applicable formatting/type/policy/API/docs checks and required CI without skip-based success |

Retain scoped logs, SQL evidence and JUnit. For each relevant state, run one final complete IT with a uniquely owned Compose project, one shared runtime build and --no-build. Reuse valid unchanged evidence; retry failed/changed boundaries only. Measurements do not establish a general SLO.

## Open questions

@jiangxt2 owns formal Alpha-contract acceptance and supported platform/budget decisions. Platform and TLS deployment claims require actual evidence; current representative validation must not imply every model/runtime combination. Durable detached ownership, persistent queues, HA and other telemetry signals remain deferred.

## Decision outcome

The owner waived the design-PR prerequisite and authorized local implementation/testing on 2026-10-23, then authorized the existing implementation PR. The proposal remains Draft; that workflow exception is not a formal Accepted decision or ADR. The single-PR simplification preserves these contracts.

## Implementation tracking

[#139](https://github.com/jiangxt2/Tributo/issues/139) tracks delivery in [#144](https://github.com/jiangxt2/Tributo/pull/144); [#135](https://github.com/jiangxt2/Tributo/issues/135) tracks diagnostic generation/configuration. Support claims follow implementation/configuration/test evidence in the operations guide and support matrix.

## Review checklist

- [x] Baseline, implementation evidence and formal proposal status are distinguished.
- [x] Goals, non-goals, ownership and Alpha stability are explicit.
- [x] This adds an output path; bounded ingestion and unbounded consumption remain separate.
- [x] Provider/Binding contracts: N/A; no bounded source is added.
- [x] ResultSink/WriteGateway contracts: N/A; no inference sink is added.
- [x] AlgorithmSpec/data-loading/Bundle contracts: N/A; no algorithm is added.
- [x] Exporter/validator/flavor/loader contracts: N/A; no model format is added.
- [x] Credentials, fail-closed startup, compatibility, migration/rollback and real infrastructure criteria are defined.
- [x] Support claims require implementation, configuration and tests.
- [ ] Formal design acceptance/ADR is recorded by the decider.

## References

- [Contribution workflow](../CONTRIBUTING.md), [proposal lifecycle](README.md) and [operations contract](../docs/operations/log-reporting.md).
- [OTel 1.45.1 HTTP exporter](https://github.com/open-telemetry/opentelemetry-python/blob/v1.45.1/exporter/opentelemetry-exporter-otlp-proto-http/src/opentelemetry/exporter/otlp/proto/http/_log_exporter/__init__.py).
- [SDK HTTP response implementation](https://github.com/open-telemetry/opentelemetry-python/blob/v1.45.1/exporter/opentelemetry-exporter-otlp-common/src/opentelemetry/exporter/otlp/common/http.py), [environment resolver](https://github.com/open-telemetry/opentelemetry-python/blob/v1.45.1/exporter/opentelemetry-exporter-otlp-proto-http/src/opentelemetry/exporter/otlp/proto/http/_common/__init__.py) and [LoggerProvider](https://github.com/open-telemetry/opentelemetry-python/blob/v1.45.1/opentelemetry-sdk/src/opentelemetry/sdk/_logs/_internal/__init__.py).
- [Contrib manifest](https://github.com/open-telemetry/opentelemetry-collector-releases/blob/v0.163.2/distributions/otelcol-contrib/manifest.yaml), [Doris exporter](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/v0.163.0/exporter/dorisexporter), [OCB](https://opentelemetry.io/docs/collector/extend/ocb/) and [OTLP semantics](https://opentelemetry.io/docs/specs/otlp/).
