# Product Scope

Canonical definition of what Tributo is — and is not. This document is the
authoritative reference for scoping decisions. When a proposed feature falls
outside these boundaries, it requires either a scope amendment (new ADR) or
explicit deferral to a future platform cycle.

## Positioning

Tributo is a **Ray-native ML Framework/SDK** — not a multi-tenant ML Platform.

| Aspect | In Scope (Framework/SDK) | Out of Scope (Platform) |
|--------|--------------------------|------------------------|
| Runtime | Ray local runtime and Kubernetes-hosted Ray (Train, Data, Serve, Tune) | Kubernetes control plane, custom scheduler, Ray Standalone management |
| Configuration | Strict typed validation models; JSON is the built-in persisted format | Web UI, managed YAML pipelines, drag-and-drop |
| Model artifacts | Bundle (Manifest + files), `BundleReader` | Model Registry with approval workflows |
| Multi-tenancy | Not in this cycle | Project/Quota/RBAC, tenant isolation, audit |
| Training | Distributed XGBoost, DNN, PU Learning, constrained algorithm SPI, and selected sklearn MapReduce adapters | AutoML, managed notebooks, experiment tracking UI |
| Inference | Batch (Ray Data) + Online (Ray Serve/gRPC) | A/B testing, canary, shadow, auto-rollback |
| Data | Explicit Ray Data / Daft bounded ingestion; `StreamSource` for unbounded | Managed ETL, data catalog, schema registry |
| Observability | Framework-level metrics, logs, trace IDs | Centralized dashboard, alerting, cost attribution |

Ray is the execution runtime; Tributo provides the Framework-level contracts for
configuration, data access, training, model artifacts, inference, and serving.
Ray Job status is NOT the sole long-term source of truth — Bundle Manifests
carry their own integrity and provenance.

## Goals (This Cycle)

1. **Unify data source protocol**: Training, inference, and graph
   adapters consume typed bounded-ingestion handles behind one Gateway. Ray
   Data and Daft perform reads; Tributo owns only contracts, translation,
   routing, error normalization, and provenance. Bounded data and streaming
   remain separate protocols with shared schema/error/credential semantics.
2. **Unify model export kernel**: All trainers produce a single `Bundle` format
   consumed by `BundleReader`. Legacy exporters become compat adapters.
3. **Fix export failure semantics**: Required artifact failure → task failure.
   No silent "training succeeded but ONNX is missing".
4. **Establish Framework-level contracts**: Error model, version policy, API
   stability tiers, benchmark protocol, migration safety rules.
5. **Build CI gates**: End-to-end `training → bundle → inference → serving` in CI.
6. **Provide Ray-native training profiles**: The same constrained algorithm can
   run against an owned local Ray runtime or a Kubernetes-hosted Ray cluster.

## Non-Goals (Explicitly Deferred)

These are NOT in scope for the current architecture cycle. They are not
permanently excluded — each has a re-evaluation trigger and will be reconsidered
when that trigger fires.

| Non-Goal | Why Deferred | Re-evaluation Trigger |
|----------|-------------|----------------------|
| Multi-tenant control plane | No production multi-tenant demand | Multiple teams request isolated namespaces/quotas |
| Model Registry governance (approval, canary, rollback) | Out of scope for Framework SDK | MLflow registry used in production with > 10 models |
| Control plane HA / cross-region DR | Platform concern, not SDK concern | Tributo deployed as a service (not library) across regions |
| Unify all DAG DSLs (`_common.dag`, `pipeline.Pipeline`, `exporting.planner`) | Each serves a different domain; premature unification creates coupling | Two or more DSLs converge on identical semantics |
| Full PluginManager with lifecycle | A descriptor-only ingestion SPI does not require a platform manager | Third-party extensions require shared lifecycle, isolation, or dependency management |
| Complete streaming semantics (at-least-once, backpressure, dead-letter) | No production Kafka workload | Kafka source runs continuously ≥ 24 hours in production |
| Automatic distributed conversion of arbitrary Trainer or sklearn estimators | Distribution requires algorithm-specific state semantics | A new algorithm implements and proves one supported distribution strategy |
| TransformCompiler pushdown | No production Provider with benchmarked pushdown path | D1+D2 merged AND pushdown shows ≥ 20% improvement on ≥ 10 GB |

## Bounded Ingestion Boundary

The bounded-ingestion architecture follows the separation used by Arrow
Dataset, Ray Data, Daft, Iceberg, Lance, and database-specific distributed
connectors: Tributo describes *what* to read; an installed engine or Connector
owns file discovery, decoding, SQL, split planning, transport, and batch loops.

The current alpha implementation has real dual-engine Conformance evidence for
Local/S3 Parquet and CSV, Local/S3 Iceberg and Lance, plus PostgreSQL. HDFS has
a Ray Data adapter but still requires its cluster gate. ClickHouse and Doris
have optional thin adapters for independent connector packages but remain
unsupported until those packages are installable and pass real-infrastructure
tests. ORC and Hive external tables remain unsupported in the locked engine
versions. A configuration enum or adapter alone is never a support claim.

The narrow `tributo.ingestion_bindings` entry-point group discovers versioned
descriptors only. It has no plugin lifecycle, dependency installation, third
engine, or execution bypass, so it does not amend the Full PluginManager
non-goal.

Lance is represented as a versioned `TableScan`, not a generic file escape
hatch. Graph algorithms compose ordinary ingestion requests by semantic role;
the ingestion module does not implement graph sampling or expose a `GraphScan`.
This keeps future source and algorithm evolution independent of downstream
Training and Inference code.

## Decision Governance

Scope decisions follow this process:

1. **Proposal**: Any contributor opens an issue with `[SCOPE]` prefix, describing
   the proposed addition and how it relates to the current positioning.
2. **Evaluation**: Maintainer checks against this document. If it falls within
   Non-Goals, the re-evaluation trigger is checked.
3. **Decision**: Recorded in `docs/architecture/decision-log.md` with date,
   evidence, and rationale.
4. **Amendment**: If scope changes (e.g., product direction shifts to Platform),
   a new ADR is required — this document is updated and the old version is
   archived.

<!-- END -->
