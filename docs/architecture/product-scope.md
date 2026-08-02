# Product Scope

Canonical definition of what Tributo is — and is not. This document is the
authoritative reference for scoping decisions. When a proposed feature falls
outside these boundaries, it requires either a scope amendment (new ADR) or
explicit deferral to a future platform cycle.

## Positioning

Tributo is a **Ray-native ML Framework/SDK** — not a multi-tenant ML Platform.

| Aspect | In Scope (Framework/SDK) | Out of Scope (Platform) |
|--------|--------------------------|------------------------|
| Runtime | Ray (Train, Data, Serve, Tune) | Kubernetes control plane, custom scheduler |
| Configuration | JSON `SourceConfig`, `TrainingConfig`, `JobConfig` | Web UI, YAML pipelines, drag-and-drop |
| Model artifacts | Bundle (Manifest + files), `BundleReader` | Model Registry with approval workflows |
| Multi-tenancy | Not in this cycle | Project/Quota/RBAC, tenant isolation, audit |
| Training | XGBoost, DNN, PU Learning on Ray Train | AutoML, managed notebooks, experiment tracking UI |
| Inference | Batch (Ray Data) + Online (Ray Serve/gRPC) | A/B testing, canary, shadow, auto-rollback |
| Data | `DataSourceProvider` for bounded reads; `StreamSource` for unbounded | Managed ETL, data catalog, schema registry |
| Observability | Framework-level metrics, logs, trace IDs | Centralized dashboard, alerting, cost attribution |

Ray is the execution runtime; Tributo provides the Framework-level contracts for
configuration, data access, training, model artifacts, inference, and serving.
Ray Job status is NOT the sole long-term source of truth — Bundle Manifests
carry their own integrity and provenance.

## Goals (This Cycle)

1. **Unify data source protocol**: Training, inference, and embeddings share
   one `DataSourceProvider` contract. Bounded data and streaming use separate
   protocols with shared schema/error/credential semantics.
2. **Unify model export kernel**: All trainers produce a single `Bundle` format
   consumed by `BundleReader`. Legacy exporters become compat adapters.
3. **Fix export failure semantics**: Required artifact failure → task failure.
   No silent "training succeeded but ONNX is missing".
4. **Establish Framework-level contracts**: Error model, version policy, API
   stability tiers, benchmark protocol, migration safety rules.
5. **Build CI gates**: End-to-end `training → bundle → inference → serving` in CI.

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
| Full PluginManager with lifecycle | No third-party plugin consumers exist | Third-party PyPI package registers Tributo entry points |
| Complete streaming semantics (at-least-once, backpressure, dead-letter) | No production Kafka workload | Kafka source runs continuously ≥ 24 hours in production |
| Multi-worker / DDP for all trainers | No real dataset exceeds single-worker capacity | A training task OOMs on single worker or multi-GPU requested |
| TransformCompiler pushdown | No production Provider with benchmarked pushdown path | D1+D2 merged AND pushdown shows ≥ 20% improvement on ≥ 10 GB |

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
