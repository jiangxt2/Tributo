# System landscape

![Tributo system landscape showing users, framework workloads and contracts, the Ray execution runtime, external systems, and platform concerns that are outside the framework boundary.](../images/tributo-system-landscape.svg)

Tributo is a Ray-native ML Framework/SDK. It owns portable contracts for job
submission, bounded ingestion, algorithm execution, model bundles, inference,
and serving. Ray 2.55.1 remains the distributed execution runtime for Jobs,
Data, Train, Tune, Serve, tasks, actors, and scheduling.

## Reading the landscape

The main flow runs from top to bottom:

- Users and applications use the CLI or Python API. Persisted configuration
  uses strict JSON, while `TributoClient` wraps submission, status, logs, and
  stop operations over the Ray Jobs API.
- Scenario workloads cover training and tuning, embeddings, batch inference,
  online serving, and streaming input. Their stability is not uniform: legacy
  trainers remain Beta compatibility paths, portable algorithm execution is
  Alpha, and Kafka is an Alpha `StreamSource` rather than a complete service
  loop.
- The portable contract layer separates bounded ingestion, algorithm dispatch,
  and Bundle-based model delivery. Providers, engine bindings, exporters,
  validators, model importers, flavors, hooks, runtime adapters, and sinks
  extend these contracts without changing the core orchestration path.
- Ray executes the resolved work. Tributo does not replace Ray's scheduler or
  distributed runtime.

## External systems

Data systems, Bundle stores, MLflow, and Kafka remain outside the framework
boundary:

- Data systems supply bounded inputs. Support varies by source and engine;
  adapter presence alone is not a support claim.
- Bundle stores persist immutable artifacts and manifests. Current publication
  supports local paths, `file://`, and S3; HDFS Bundle storage is not
  implemented.
- MLflow is an optional model-import, tracking, provenance, and registry
  integration. It is not the source of truth for Bundle readability.
- Kafka provides a fail-closed microbatch input protocol. Tributo does not ship
  a built-in Kafka-to-inference long-running service loop.

The [support matrix](../reference/support-matrix.md) is the authoritative view
of verified, Alpha, Beta, adapter-only, and unsupported paths. The diagram
groups architectural responsibilities; it does not promote every extension
contract to a supported product capability.

## Framework boundary

Tributo is intentionally not a multi-tenant ML platform. Kubernetes control
planes, custom scheduling, tenant isolation, quota and RBAC, approval and
canary workflows, and a centralized operations UI are outside the current
framework scope. See [Product Scope](product-scope.md) for the complete set of
goals, non-goals, and re-evaluation triggers.
