# API stability inventory

`@PublicAPI` annotations are the canonical symbol-level stability source. The
[generated API reference](reference/index.md) covers every annotated object and
CI compares each imported object's runtime stability with the source inventory.
This page provides module-level guidance and deprecation notes.

## Stability Levels

| Level | Meaning | Backward Compatibility |
|-------|---------|----------------------|
| `stable` | Public API with SemVer guarantee | Breaking changes require major version bump |
| `beta` | Public API under active development | May change with deprecation notice (≥ 2 minor versions) |
| `alpha` | Experimental public API | May change without notice |
| `deprecated` | Will be removed; use replacement | Deprecation window per ADR 001 |
| `prototype` | Validation-only code; not for production use | No compatibility promise |
| `developer` | Internal implementation detail | May change in any release |
| `legacy` | Old path kept as compat adapter | Will be removed per migration plan |

## Module Inventory

### Core (tributo.*)

| Module | Level | Notes |
|--------|-------|-------|
| `tributo.config` — `JobConfig` | `stable` | Ray Job submission config |
| `tributo.config` — `AlgorithmExecutionConfig` and nested algorithm execution models | `alpha` | Strict JSON envelope shared by owned-local and attached-cluster Ray execution |
| `tributo.job` — `TributoClient` | `stable` | Primary Ray Jobs client |
| `tributo.job` — `RayJob` | `stable` annotation with runtime deprecation warning | Use `TributoClient`; the annotation and warning conflict is documented without changing the public contract in this documentation update |
| `tributo.ray_jobs` | `alpha` | Workload-neutral submission identity, ambiguous-submit reconciliation, status, logs, and stop helpers |
| `tributo.exceptions` — core exceptions | `stable` | ``TributoError`` and 16 common subtypes |
| `tributo.exceptions` — `ResultMaterializationError` | `alpha` | Credential-safe lazy inference action failure |
| `tributo.exceptions` — Bundle/Plugin exceptions | `beta` | ``BundleExportError``, ``BundleCommitBusyError``, ``AliasConflict``, ``UnsupportedArtifactFormat``, ``PostPublishCallbackError``, ``PluginLoadIssue`` |
| `tributo.exceptions` — Streaming exceptions | `beta` | ``StreamSourceError``, ``KafkaCommitError``, ``KafkaPoisonMessageError`` |
| `tributo.exceptions` — `EngineNotAvailableError` | `alpha` | Candidate bounded-ingestion error |
| `tributo.cli` | `beta` | Command-line interface |

### Runtime image tooling

The image builder and emitted files are repository tooling rather than public
Python API symbols. Their contract is intentionally Alpha and is
covered by the external `runtime-image` suite.

| Surface | Level | Notes |
|--------|-------|-------|
| `tools/build_tributo_image.py` | `alpha` | JSON-only Buildx builder for the pinned image validated for CPU execution by default, with explicit `linux/amd64`/`linux/arm64` targeting; it performs dependency-closure discovery, manifest sealing, and fail-closed import checks |
| `tools/tributo-runtime-full.json` | `alpha` | Multi-architecture pinned Ray/uv image references, native-platform default, complete first-party runtime-extra closure including locked v1.0 ClickHouse/Doris connectors and the external `ray-hive` package, and optional external wheelhouse support |
| `manifest.json` / `image-profile.json` | `alpha` | Build attestations consumed for immutable image selection and algorithm artifact compatibility; not a registry or deployment API |

### Training (tributo.training.*)

Callbacks without a public `failure_policy` are best-effort in every normal
lifecycle phase, including `on_setup_start`. Callbacks that must abort training
need to declare `failure_policy = "required"`; this is a Beta behavior change
from the legacy setup-only propagation rule.

| Module | Level | Notes |
|--------|-------|-------|
| `tributo.training.config` — `TrainingDataConfig` and trainer-specific config models | `beta` | Training data and trainer configuration contracts |
| `tributo.training.base` — `BaseTrainer`, `TrainerSpec` | `beta` | First-party trainers default to an explicit-destination Bundle; raw artifacts require `legacy_export=True` |
| `tributo.training.results` — `TrainingResult` and status enums | `beta` | Closed training, Bundle, Hook, URI, and execution identity result contract |
| `tributo.training.checkpoint` | `beta` | Resume checkpoint contract |
| `tributo.training.xgboost_trainer` | `deprecated` | Production implementations moved to the official `tributo-algorithms` Wheels |
| `tributo.training.dnn_trainer` | `deprecated` | Production implementations moved to the official `tributo-algorithms` Wheels |
| `tributo.training.pu_trainer` | `deprecated` | Production implementations moved to the official `tributo-algorithms` Wheels |
| `tributo.training.graph_trainer` | `alpha` | Early-stage graph training |
| `tributo.training.causal_estimator` | `beta` | Causal effect estimation (docstring was alpha; aligned to @PublicAPI) |
| `tributo.training.algorithm_spec` | `beta` | Algorithm capability declarations |
| `tributo.training.catalog` | `beta` | Algorithm registry |
| `tributo.training.data_loader` | `beta` | Ray compatibility adapter over IngestionGateway |
| `tributo.training.tune_config` | `beta` | Hyperparameter tuning config |
| `tributo.training.tune_runner` | `beta` | Tune execution |
| `tributo.training.portable_tune` | `alpha` | Portable distributed fit-only Tune execution |
| `tributo.training.tune_space` | `beta` | Search space definitions |
| `tributo.training.priors` | `beta` | Class prior estimation |
| `tributo.training.flavor` | `beta` | Model flavor adapter |
| `tributo.training.onnx_exporter` | `deprecated` | Use `tributo.exporting` instead |
| `tributo.training.exporters.*` | `deprecated` | Use `tributo.exporting` / `tributo.integrations.exporters` instead |
| `tributo.training.registry` | `beta` | Trainer registration API |
| `tributo.training.job_submitter` | `beta` | Job submission helpers |
| `tributo.training.local_runner` | `beta` | Local training runner |
| `tributo.training.features` | `beta` | Feature declarations and transformations |
| `tributo.training.losses` | `beta` | Training loss implementations |
| `tributo.training.models` | `beta` | First-party model definitions |
| `tributo.training.exporters.artifact_protocol` | `deprecated` | Legacy artifact protocol compatibility |

### Portable algorithm execution (tributo.algorithms.*)

| Module | Level | Notes |
|--------|-------|-------|
| `tributo.algorithms.api.models` | `alpha` | Portable registration, request, plan, result, environment, runtime, input, and artifact value objects |
| `tributo.algorithms.api.artifacts` | `alpha` | Algorithm Wheel/Bundle distribution and immutable image Profile contracts |
| `tributo.algorithms.api.distribution` | `alpha` | Versioned distributed strategy, profile, resource, and coordination declarations |
| `tributo.algorithms.api.execution` | `alpha` | Formal execution request and immutable worker/node/shard receipt evidence |
| `tributo.algorithms.api.descriptor` | `alpha` | Trusted installed-package distributed algorithm descriptor API v1 |
| `tributo.algorithms.api.context` — `UserExecutionContext` | `alpha` | Restricted context for trusted module-qualified Worker functions |
| `tributo.algorithms.api.errors` | `alpha` | Portable execution error taxonomy |
| `tributo.algorithms.api.support` | `alpha` | Trusted Wheel support evidence, execution semantics, expiry, and revocation |
| `tributo.algorithms.conformance` | `alpha` | Descriptor-only and installed algorithm Wheel Conformance Testkit |
| `tributo.algorithms.builtin.*` | `deprecated` | Production algorithms moved to the official `tributo-algorithms` Wheels; Core retains only public SPI and Ray runtimes |
| `tributo.algorithms.core.builder` — `AlgorithmBuilder` | `alpha` | Provisional sklearn and Custom Ray Function registration builders |
| `tributo.algorithms.core.runtime` | `alpha` | Owned local Ray lifecycle and deployment-neutral attached-cluster connection |
| `tributo.algorithms.composition` | `alpha` | Default formal Dispatcher composition root |
| `tributo.algorithms.spi.execution` | `alpha` | Provisional operation and Runtime execution protocols |
| `tributo.algorithms.spi.contracts` | `alpha` | Executable algorithm contract validator protocol |
| `tributo.algorithms.spi.input` | `alpha` | Two-stage input resolution and Driver/Worker ownership contracts |
| `tributo.algorithms.spi.torch` | `alpha` | Narrow model/loss/optimizer/metric recipe contract lowered to Ray Train |

### Data (tributo.data.*)

| Module | Level | Notes |
|--------|-------|-------|
| `tributo.data.source_config` — `SourceConfig` | `beta` | Strict in-memory contract; JSON is the built-in persisted format |
| `tributo.data.provider` — `DataSourceProvider` | `beta` | Logical normalization/planning contract; `normalize()` and `plan()` drive canonical ingestion, while `open()`/`DatasetHandle` remain an independent Provider SPI |
| `tributo.data.transform_ir` | `alpha` | Versioned engine-neutral ETL contract |
| `tributo.data.transform_compiler` | `developer` | Internal Ray/Daft expression translation |
| `tributo.data.scan_plan` | `developer` | Internal engine-neutral scan SPI; downstream consumers use `IngestionGateway` |
| `tributo.data.ingestion` | `alpha` | Two-stage Gateway, explicit request, typed handles, and receipt |
| `tributo.data.handle_adapters` | `alpha` | Explicit native-handle conversions with conversion evidence; never a routing fallback |
| `tributo.data.contracts.handles` | `alpha` | Typed Ray and Daft handle contracts shared by ingestion and writing |
| `tributo.data.contracts.modes` | `beta` | Canonical shared `WriteMode` contract; `tributo.data.base` remains a narrow re-export |
| `tributo.data.contracts.storage` | `beta` | Canonical shared `S3Config` contract; credentials are hidden from repr and identity material |
| `tributo.data.writing` | `alpha` | Unified bounded-write Gateway package |
| `tributo.data.writing.capabilities` | `alpha` | Native writer capability declarations |
| `tributo.data.writing.contracts` | `alpha` | Credential-safe write requests, descriptors, receipts, and errors |
| `tributo.data.writing.gateway` | `alpha` | Target planning, capability negotiation, and native write delegation |
| `tributo.data.engine_binding` | `developer` | Third-party extension SPI; not exported from the consumer-facing `tributo.data` root |
| `tributo.data.binding_plugins` | `developer` | Descriptor-only `tributo.ingestion_bindings` discovery SPI |
| `tributo.data.provider_plugins` | `developer` | Versioned descriptor-only `tributo.ingestion_providers` discovery SPI |
| `tributo.data.bindings.*` | `developer` | Thin adapters over public Ray Data, Daft, or installed engine APIs |
| `tributo.data.graph` | `beta` | Graph data abstraction (GNN; @PublicAPI says beta) |
| `tributo.data.base` | `legacy` | Narrow compatibility re-export for `S3Config` and `WriteMode`; no reader, writer, registry, or Connector class |
| `tributo.data.provider_registry` | `beta` | Data source provider registry |
| `tributo.data.refs` | `beta` | Data reference value objects |

### Distributed Lance vector indexing (tributo.vector_index.*)

| Module | Level | Notes |
|--------|-------|-------|
| `tributo.vector_index.contracts` | `alpha` | Credential-free build, search, maintenance, result-delivery, and receipt contracts |
| `tributo.vector_index.index_job` | `alpha` | Planning-baseline distributed index build orchestration and coverage evidence |
| `tributo.vector_index.search` | `alpha` | Fixed-version Lance-Ray Top-K query orchestration |
| `tributo.vector_index.maintenance` | `alpha` | Lance-Ray index optimization and distributed compaction orchestration |

### Exporting / Bundle (tributo.exporting.*)

| Module | Level | Notes |
|--------|-------|-------|
| `tributo.exporting.service` — `BundleExportService` | `beta` | Primary export orchestration |
| `tributo.exporting.models` — all public models | `beta` | Configuration, artifact, canonical Bundle reference, and exact publication result models |
| `tributo.exporting.protocols` — all protocols | `beta` | Exporter/Validator/SourceProvider contracts |
| `tributo.exporting.manifest` — `ExportManifest` | `beta` | Bundle manifest (schema v1) |
| `tributo.exporting.bundle_reader` — `BundleReader` | `beta` | Repository-routed Bundle consumption with exact manifest-byte and artifact verification |
| `tributo.exporting.planner` | `beta` | Export plan builder |
| `tributo.exporting.executor` | `beta` | Export executor |
| `tributo.exporting.publisher` | `beta` | Bundle publisher |
| `tributo.exporting.validators` | `beta` | Artifact validator runner |
| `tributo.exporting.registries` | `beta` | Exporter/validator registries |
| `tributo.exporting.options` | `beta` | Compatibility re-exports; schemas are owned by integration exporters |
| `tributo.exporting.records` | `beta` | Export record types; `PublicationAttempt` is read-only legacy compatibility and receives no new writes |
| `tributo.exporting.gc` | `beta` | Bundle GC |
| `tributo.exporting.events` | `beta` | Immutable publication event contract |
| `tributo.exporting.hooks` | `beta` | Adapter and committed-artifact access contracts |
| `tributo.exporting.dispatch` | `beta` | Inline Hook dispatch policy |
| `tributo.exporting.capabilities` | `beta` | Exporter/Flavor-derived capability declarations |
| `tributo.exporting.repository` | `beta` | Bundle repository and alias store ports |
| `tributo.exporting.runtime` | `beta` | Bundle model runtime and Flavor protocol |
| `tributo.exporting.conftest` | `beta` | Public plugin conformance test kit |

### Explainability (tributo.explainability.*)

| Module | Level | Notes |
|--------|-------|-------|
| `tributo.explainability.conformance` | `alpha` | Adapter SPI structural conformance validation |
| `tributo.explainability.contracts` | `alpha` | Explainability request, descriptor, attribution, receipt and policy contracts |
| `tributo.explainability.executor` | `alpha` | Ray Data batch executor, lease heartbeat and attempt isolation |
| `tributo.explainability.export` | `alpha` | Bundle export companion-artifact preparation |
| `tributo.explainability.job_runner` | `alpha` | Ray Jobs submission entry point for batch explanations |
| `tributo.explainability.planner` | `alpha` | Adapter selection and resource preflight planning |
| `tributo.explainability.protocols` | `alpha` | Adapter SPI, resolved model binding, serializable model-session factory, result-store port, model context and support decision protocols |
| `tributo.explainability.reference` | `alpha` | Reference/background data provider protocol and file provider |
| `tributo.explainability.registry` | `alpha` | Adapter registry and entry-point discovery |
| `tributo.explainability.shap` | `alpha` | First-party SHAP adapter (tree and model-agnostic backends) |

### Integrations (tributo.integrations.*)

| Module | Level | Notes |
|--------|-------|-------|
| `tributo.integrations.algorithm_inputs` | `alpha` | Production IngestionGateway bridge with invocation-scoped request refs and explicit Ray/Daft Worker adapters |
| `tributo.integrations.algorithm_inputs.ingestion` | `alpha` | Ingestion input bridge |
| `tributo.integrations.algorithm_runtimes.legacy_descriptors` | `developer` | Internal lightweight descriptors for the bounded Trainer compatibility bridge |
| `tributo.integrations.algorithm_runtimes.legacy_trainer` | `developer` | Internal Worker-only execution adapter; not a native first-party runtime |
| `tributo.integrations.algorithm_runtimes.collective` | `developer` | Internal Ray Train collective runtime adapter |
| `tributo.integrations.algorithm_runtimes.framework_native` | `developer` | Internal framework-native distributed runtime adapter |
| `tributo.integrations.algorithm_runtimes.map_reduce` | `developer` | Internal bounded tree-MapReduce runtime adapter |
| `tributo.integrations.exporters.*` | `beta` | Built-in exporter implementations |
| `tributo.integrations.exporters.x_learner` | `alpha` | Fixed X-Learner model and causal-report exporter adapters |
| `tributo.integrations.flavors` | `beta` | Built-in runtime flavor package |
| `tributo.integrations.flavors.onnx_runtime` | `beta` | ONNX Runtime flavor implementation |
| `tributo.integrations.validators` | `beta` | Built-in validator package |
| `tributo.integrations.validators.*` | `beta` | Built-in validator implementations |
| `tributo.integrations.sources` | `beta` | Built-in source provider package |
| `tributo.integrations.sources.*` | `beta` | Built-in source providers |
| `tributo.integrations.sources.ray_torch_recipe` | `alpha` | Generic trusted Torch recipe checkpoint provider |
| `tributo.integrations.storage` | `beta` | Built-in storage adapter package |
| `tributo.integrations.storage.*` | `beta` | Built-in storage backends |
| `tributo.integrations.hooks` | `beta` | Built-in Hook package |
| `tributo.integrations.hooks.*` | `beta` | Built-in hooks (MLflow etc.) |
| `tributo.integrations.flavors.xgboost_native` | `alpha` | Safe native JSON/UBJ XGBoost runtime flavor |
| `tributo.integrations.flavors.x_learner` | `alpha` | Safe batch-only fixed-composition X-Learner runtime flavor |
| `tributo.integrations.sources.ray_x_learner` | `alpha` | Five-checkpoint X-Learner ExportSource provider |
| `tributo.integrations.model_importers.*` | `alpha` | Canonical ModelImporter protocol/registry plus explicit MLflow and typed artifact-to-Bundle implementations |
| `tributo.integrations.model_runtimes.*` | `alpha` | Bundle-backed model reference, prediction-kernel, and explainability capability adapters |
| `tributo.integrations.sinks.parquet` | `alpha` | Parquet inference ResultSink adapter |
| `tributo.integrations.sinks.lance` | `alpha` | Generic Lance inference ResultSink adapter |
| `tributo.integrations.sinks.data_write` | `alpha` | Generic data-module-backed inference ResultSink adapter |
| `tributo.integrations.sinks.explainability` | `alpha` | Explainability ResultSink, inspection, receipt and cleanup adapter |
| `tributo.integrations.broker` | `alpha` | Minimal transport-neutral Broker API v1; transport implementations and consume loops are external |
| `tributo.integrations.broker_registry` | `alpha` | Lazy broker discovery and explicit provider resolution |

### Inference (tributo.inference.*)

| Module | Level | Notes |
|--------|-------|-------|
| `tributo.inference.base` — `BasePredictor` | `beta` | Batch predictor contract |
| `tributo.inference.batch_predictor` | `beta` | Batch predictor implementation |
| `tributo.inference.pipeline` | `beta` | Inference pipeline; data loading delegates the Ray Gateway adapter |
| `tributo.inference.job_runner` | `beta` | Inference job runner |
| `tributo.inference.contracts` | `alpha` | Compatibility request/result contracts plus stripped prepared execution and sink ports |
| `tributo.inference.api` | `alpha` | Bundle-aware resolve and execute entry points |
| `tributo.inference.importers` | `alpha` | Compatibility re-export; new code uses `tributo.integrations.model_importers` |
| `tributo.inference.input_resolver` | `alpha` | Public IngestionGateway to RayDataHandle adapter |
| `tributo.inference.resolver` | `alpha` | Fail-closed immutable inference-plan resolver |
| `tributo.inference.bundle_predictor` | `alpha` | Named tensor binding Ray actor |
| `tributo.inference.executor` | `alpha` | RayMapBatchesExecutor |
| `tributo.inference.kernel` | `alpha` | Format-neutral prediction Kernel, Kernel Factory, provider, and Ray batch binding contracts |
| `tributo.inference.post_training` | `alpha` | Training-result entry adapter; no Training implementation dependency |
| `tributo.runtime` | `beta` | Deployment-neutral RuntimeTarget contracts plus DeveloperAPI lazy composition factories |

### Serving (tributo.serving.*)

| Module | Level | Notes |
|--------|-------|-------|
| `tributo.serving.serve_runner` | `beta` | Ray Serve management |
| `tributo.serving.model_deployment` | `beta` | Model deployment config |
| `tributo.serving.grpc_deployment` | `beta` | gRPC serving deployment |
| `tributo.serving.grpc_runner` | `beta` | gRPC runner |
| `tributo.serving.streaming_deployment` | `alpha` | Streaming serving |
| `tributo.serving.streaming_runner` | `beta` | Streaming runner |
| `tributo.serving.composition` | `beta` | Composite model inference |
| `tributo.serving.schema` | `beta` | Serving schema types |
| `tributo.serving.proto.*` | `developer` | Generated protobuf code |
| `tributo.serving.proto` | `developer` | Generated protobuf package |

### Streaming (tributo.streaming.*)

| Module | Level | Notes |
|--------|-------|-------|
| `tributo.streaming.protocol` — `StreamSource` | `beta` | Streaming protocol |
| `tributo.streaming.kafka_source` | `alpha` | Kafka source (fail-closed safety baseline: commit retention, poison-message stop, uncommitted-batch barrier) |

### Pipeline (tributo.pipeline.*)

| Module | Level | Notes |
|--------|-------|-------|
| `tributo.pipeline.core` | `alpha` | Pipeline orchestration |

### Registry (tributo.registry.*)

| Module | Level | Notes |
|--------|-------|-------|
| `tributo.registry.model_registry` | `beta` | Model registry client |
| `tributo.registry.callback` | `beta` | Training callback |
| `tributo.registry.schema` | `beta` | Registry schema types |
| `tributo.registry.mlflow_util` | `developer` | Internal MLflow utilities |

### Plugin (tributo.plugin)

| Module | Level | Notes |
|--------|-------|-------|
| `tributo.plugin` — all `discover_*` functions | `beta` | Plugin discovery (no PluginManager until PL1+PL2) |
| `tributo.plugin.validate_distributed_algorithm_descriptor` | `alpha` | Conformance validation for constrained distributed algorithm entry points |

### Utilities (tributo.util.*)

| Module | Level | Notes |
|--------|-------|-------|
| `tributo.util.annotations` — `PublicAPI`, `DeveloperAPI` | `stable` | Stability annotation system |

### Common (tributo._common.*)

| Module | Level | Notes |
|--------|-------|-------|
| `tributo._common.storage_profiles` | `beta` | Storage profile resolution |
| `tributo._common.dependencies` | `beta` | Unified dependency probing layer |
| `tributo._common` | `developer` | Internal shared package |
| All other `tributo._common.*` | `developer` | Internal shared utilities |

## Deprecation Schedule

| Old API | Replacement | Deprecated Since | Removal Window |
|---------|------------|-----------------|----------------|
| `tributo.training.exporters.*` | `tributo.exporting.*` + `tributo.integrations.exporters.*` | v1.0.0 | ≥ 2 minor versions (E4) |
| `tributo.training.onnx_exporter` | `tributo.exporting.service.BundleExportService` | v1.0.0 | ≥ 2 minor versions (E4) |
| `BaseTrainer.run(..., legacy_export=True)` | Default Bundle publication with an explicit `BundleOutputConfig.bundle_uri` | v1.0.0 | ≥ 2 minor versions (E4); emits `DeprecationWarning` per invocation |
| `tributo.exporting.protocols.SourceProvider` (name) | `ExportSourceProvider` (E1) | After E1 merge | 2 minor versions with DeprecationWarning |
| Legacy flat data config | `CanonicalSourceInput` / `IngestionRequest` | v1.0.0 | Conversion-only adapter retained for its deprecation window; direct dispatch removed |
| `TRIBUTO_DATA_BACKEND=legacy` | Default Provider/Gateway path | v1.0.0 | Selector is accepted with `FutureWarning` during the compatibility window; it no longer restores direct dispatch |
| Third-party Provider `normalize()+open()` SPI | `plan()` + `EngineBinding` | v1.0.0 | Provider `open()` remains a standalone SPI; canonical Gateway never falls back and the removed Ray adapter warning is no longer emitted |
| `InferenceConfig.s3_config` | Independent source/model/sink storage profiles | Inference architecture P0 | One compatibility window with `DeprecationWarning` |
| `tributo.job.RayJob` | `tributo.job.TributoClient` | Runtime warning exists in v1.0.0 | Removal version is not declared; the `stable` annotation remains authoritative until a separate API decision changes it |

## Unannotated Code

Code not listed above defaults to `developer` (internal). If a symbol is missing
from this inventory but should be public API, file a PR to add it to this table
and annotate it with `@PublicAPI`.

<!-- END -->

## Stability-Aware Module Docstrings

The following modules have stability levels stated in their file-level docstrings.
This is informative only — `STABILITY.md` is the canonical reference.

### Marked as prototype

- None.

### Marked as deprecated

- `tributo.training.exporters` — "Deprecated re-exports from tributo.exporting"

### Marked as alpha

- `tributo.data.transform_ir` — versioned engine-neutral ETL contract
- `tributo.data.ingestion` — candidate dual-engine ingestion API
- `tributo.pipeline.core` — "Alpha; lightweight in-process DAG executor"
- `tributo.training.graph_trainer` — "Alpha; GNN training"
- `tributo.serving.streaming_deployment` — "Alpha; streaming inference service"
- `tributo.streaming.kafka_source` — "Alpha; Kafka source"

### Marked as beta

- All other modules not listed above are `beta` if they appear in the
  Module Inventory table, or `developer` if they don't.
