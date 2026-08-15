# Support matrix

This page distinguishes implemented paths from extension contracts and
prototypes.

Capability status is independent from GitHub Actions placement. A real
environment Gate may provide support evidence while remaining an external
validation; `ci/test-suites.json` is authoritative for execution tier and
automation eligibility.

## Validation evidence

| Area | Implementation and configuration | Bounded CI evidence | External environment evidence |
| --- | --- | --- | --- |
| Bounded ingestion | Data Gateway, Provider/Binding descriptors, and engine options | `unit`, `unit-integration-contracts`, and the ephemeral `s3-contract` suite | `data-ingestion-cluster` |
| Model export and Bundle publication | Export planner, exporters, validators, publisher, and storage profiles | `unit`, `unit-integration-contracts`, and `s3-contract` | `model-export-cluster` |
| Batch and online inference | Inference plans, model flavors, sinks, and serving configuration | `unit` and `unit-integration-contracts` | `inference-cluster` |
| Lance vector index | Vector-index API, storage profiles, and Ray Jobs request contract | `unit` contracts | `lance-vector-cluster` |
| Tune and explainability | Capability declarations and typed job configuration | `unit` contracts | `tune-cluster` and `explainability-cluster` |
| Distributed algorithms | Algorithm descriptors, execution profiles, and portable receipts | `unit` and `unit-integration-contracts` | `distributed-algorithm-cluster` |

External suite names describe required evidence, not GitHub Actions jobs.
Quarantined MLflow, serving, streaming, and legacy standalone tests do not
count as capability evidence until their lifecycle and reliability conditions
are repaired.

For distributed algorithms, compatible profiles are declared by the static
contract. Validated profiles have additionally passed the corresponding real
environment Gate. The project-wide distributed algorithm Gate uses Ray Jobs on
an isolated Docker cluster with two worker nodes to verify sharding, state
coordination, cross-node receipts, and Bundle atomicity. Kubernetes remains a
deployment profile managed by KubeRay, not a separate Tributo control plane.
No real KubeRay environment Gate has been completed yet: `kubernetes` is a
compatible profile, while the generated `Validated profiles` column remains
`local` until a RayJob on KubeRay supplies direct environment evidence.

## Data

| Capability | Status | Boundary |
| --- | --- | --- |
| Local/S3 Parquet and CSV reads | Verified | Native Ray Data or Daft handle through one Gateway |
| Local/S3 Iceberg reads | Verified | Built-in bindings use PyIceberg `>=0.11.1,<0.12.0` with `PyArrowFileIO`; Ray may push `row_filter` into the scan, Daft applies it as a lazy residual filter, and empty-table schema is preserved from Iceberg metadata; broader Catalog/delete-file matrix remains gated |
| Local/S3 Lance reads | Verified | Native Ray Data or Daft table reader; numeric versions and tags are supported, Daft also supports as-of, and Iceberg snapshot references fail closed |
| PostgreSQL structured table reads | Verified | Ray uses a single public SQL read and fails closed on parallel shard requirements; Daft may use native partition hints |
| ClickHouse/Doris raw SQL | Unsupported | Legacy shapes return a credential-free migration error; use structured table input or execute SQL outside Tributo ingestion |
| HDFS Parquet/CSV reads | Adapter only | Ray binding exists; real HDFS/JVM/worker gate is pending |
| ClickHouse reads | Adapter only | Requires unpublished `daft-olap-connectors` and real-database Conformance; provider partition discovery is distinct from engine auto-routing |
| Doris reads | Adapter only | Requires unpublished `ray-doris` or `daft-olap-connectors` and real-database Conformance; tablet planning remains provider/binding-owned |
| ORC and Hive external-table reads | Not implemented | Locked Ray/Daft versions expose no validated public reader |
| Third-party ingestion Provider/Binding SPI | Implemented | Installed packages use `tributo.ingestion_providers` plus `tributo.ingestion_bindings`; bad plugins are isolated, duplicate routes never replace built-ins, and Binding selection can constrain filesystem, catalog, and storage format |
| Lance output | Implemented as a generic ResultSink path | User Predictor owns vector semantics; the sink does not pool, normalize, or automatically invoke the separate vector-index workflow |
| Native bounded writes | Basic local/S3 native round-trips are implemented through `WriteGateway` for Ray/Daft Parquet, CSV, Iceberg, and Lance. Ray Lance delegates to locked `lance-ray==0.5.0`/PyLance 9; Daft delegates to `DataFrame.write_lance`. Mode support comes from the selected native Binding capability. Existing-target `CREATE`, missing-target `APPEND`, schema evolution, and empty writes remain provider-owned, are not Tributo guarantees, and are outside the current Gate | Tributo owns control-plane validation only; Ray Data, Daft, or an official native integration owns data-plane writes |
| Custom Hugging Face Predictor | User-provided Ray Data/Jobs extension point | Tokenization, task semantics, output interpretation, pooling, normalization, and metadata remain user-owned |
| Database inference sinks | Extension point | No built-in ClickHouse or Doris sink |
| Ray/Daft transform compiler | Alpha | Portable bounded ETL subset with dual-engine Conformance |
| File glob empty-match behavior | Engine-defined | Tributo delegates discovery; exception type and timing follow the selected Ray/Daft reader |
| Bounded SQL empty result | Valid | Ingestion returns an empty native handle; Training, Inference, or another consumer decides whether empty input is acceptable |

## Training

### Registered algorithms

<!-- BEGIN GENERATED: TRIBUTO ALGORITHM SUPPORT -->
<!-- Generated by tools/generate_algorithm_support_matrix.py; do not edit. -->
| Algorithm | Lifecycle | Stability | Availability | Tested | Supported | Execution | Implementations | Topology | Distribution | Compatible profiles | Validated profiles | Input views | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <code>dnn</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>yes</code> | <code>yes</code> | <code>fit</code> | <code>tributo.dnn.legacy_trainer</code>, <code>tributo.dnn.ray_train_collective</code> | <code>ray_train_collective</code>, <code>single_worker</code> | <code>ray_train_collective</code> | <code>kubernetes</code>, <code>local</code> | <code>local</code> | <code>ray_data</code> | <code>CPU/Gloo is supported; GPU/NCCL requires a separate gate.</code>, <code>Multi-worker BatchNorm is rejected until synchronized BatchNorm is gated.</code>, <code>Automatic worker retries are rejected until failure injection is gated.</code> |
| <code>multinomial_nb</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>yes</code> | <code>yes</code> | <code>fit</code> | <code>tributo.multinomial_nb.map_reduce</code> | <code>ray_map_reduce</code> | <code>ray_map_reduce</code> | <code>kubernetes</code>, <code>local</code> | <code>local</code> | <code>ray_data</code> | — |
| <code>pu</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>yes</code> | <code>yes</code> | <code>fit</code> | <code>tributo.pu.legacy_trainer</code>, <code>tributo.pu.ray_train_collective</code> | <code>ray_train_collective</code>, <code>single_worker</code> | <code>ray_train_collective</code> | <code>kubernetes</code>, <code>local</code> | <code>local</code> | <code>ray_data</code> | <code>CPU/Gloo is supported; GPU/NCCL requires a separate gate.</code>, <code>Multi-worker BatchNorm is rejected until synchronized BatchNorm is gated.</code>, <code>Automatic worker retries are rejected until failure injection is gated.</code> |
| <code>xgboost</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>yes</code> | <code>yes</code> | <code>fit</code> | <code>tributo.xgboost.framework_native</code>, <code>tributo.xgboost.legacy_trainer</code> | <code>framework_managed</code>, <code>framework_native</code> | <code>framework_native</code> | <code>kubernetes</code>, <code>local</code> | <code>local</code> | <code>ray_data</code> | <code>CPU distributed training is supported; GPU requires a separate gate.</code>, <code>Multi-worker checkpoint resume is not supported.</code>, <code>Automatic worker retries are rejected until failure injection is gated.</code> |
<!-- END GENERATED: TRIBUTO ALGORITHM SUPPORT -->

### Training infrastructure and planned capabilities

| Capability | Status | Boundary |
| --- | --- | --- |
| Ray Tune | Beta | Capability-gated algorithms only |
| Legacy managed sklearn and Custom Ray Function | Alpha compatibility | Joblib and legacy `data_parallel` remain compatibility mechanisms and do not prove distributed model training |
| Portable distributed execution | Alpha | Explicit collective, framework-native, and bounded tree-MapReduce strategies; local and Kubernetes profiles share one contract |
| Constrained algorithm descriptor SPI | Alpha | Trusted packages from the selected image or a validated Job artifact; no arbitrary dependency resolution, isolation, hot reload, or PluginManager lifecycle |
| Algorithm Wheel distribution | Alpha | Image Profiles plus code-only `py_modules` Wheels by default; opt-in offline Wheelhouse installs use `--no-index`, an attested manifest, and the existing entry-point registry. No online dependency resolution or untrusted-code sandbox is provided |
| Graph training | Alpha skeleton | No built-in PyG/DGL trainer |
| Causal estimation | Extension contract | No concrete estimator is bundled |
| Streaming user recovery decisions | Not implemented | Kafka source remains separate; no recovery algorithm or source-to-sink runtime is planned by the algorithm-module refactor |

The tabular DNN, PU, and XGBoost legacy Trainer implementations remain
available to Beta compatibility APIs while their native distributed
registrations are the default formal implementations. `multinomial_nb` is a
first-party sklearn-backed MapReduce example; it is not a claim of complete
sklearn support. Graph neural networks, Transformers, and `RLlib`
policies do not yet have built-in trainers. The CLI algorithm catalog and this
matrix are generated from the same Registry projection.

## Bundle and inference

| Capability | Status | Boundary |
| --- | --- | --- |
| Local and `file://` bundle publication | Beta | Manifest and digest validation |
| S3 bundle publication | Beta | Manifest-last and alias compare-and-set |
| HDFS bundle publication | Not implemented | Storage backend extension |
| Ray Data batch inference | Beta | Actor-based model reuse |
| Batch output to local/S3 Parquet | Implemented | Database sinks are separate |
| Batch explainability | Alpha | Optional SHAP adapter over a declared Bundle role; batch-only Ray ingestion and bounded Parquet results |

Artifact capabilities are independent. "Readable" means `BundleReader` can
verify and expose the artifact; it does not imply that Tributo can execute it.

| Flavor | Exportable | Bundle readable | Batch inference | Online serving | Boundary |
| --- | --- | --- | --- | --- | --- |
| `onnx-runtime-v1` | Yes | Yes | Yes | Yes | Typed signature and safe ONNX Runtime loader |
| `xgboost-native-v1` | Yes | Yes | Yes | Yes | Canonical `ubj` and `xgboost-json` formats share this safe Booster runtime; canonical `float_input` binding |
| `safetensors-v1` | Yes | Yes | No | No | Weights-only; no trusted architecture loader |
| `torch-export-v1` | Yes | Yes | No | No | PT2 loader and version/device contract are pending |
| `hf-onnx-v1` | Yes | Yes | No | No | Dedicated runtime compatibility gate is pending |
| `onnx-int8-v1` | Yes | Yes | No | No | Quantized numerical compatibility gate is pending |

The ONNX entries describe execution of validated tensors. DNN/PU Bundle
publication includes digest-protected preprocessing state, and
`IdentityPredictor` consumes it for raw-feature online inference. In v1.0.0,
batch inference requires callers to bind already-preprocessed
tensors; it does not apply DNN/PU preprocessing implicitly.

## Serving and streaming

| Capability | Status | Boundary |
| --- | --- | --- |
| ONNX HTTP serving | Beta | Ray Serve |
| gRPC serving | Beta | Install the `grpc` extra |
| LLM SSE serving | Alpha | Streaming service contract |
| Kafka source | Alpha | Fail-closed microbatch source |
| Kafka-to-inference service loop | Not built in | Requires explicit orchestration and sink |

## Vector indexing

| Capability | Status | Boundary |
| --- | --- | --- |
| Lance vector-index build | Alpha, verified | IVF_FLAT and IVF_PQ through Lance-Ray with l2, cosine, or dot distance |
| Lance vector search | Alpha, verified | Fixed-version global Top-K with bounded inline or explicit Parquet delivery |
| Lance index optimization | Alpha, verified | Active dataset version only; indexes appended fragments |
| Lance file compaction | Alpha, verified | Active dataset version only; Lance-Ray owns file compaction |
| Vector-index namespace mode | Alpha | Namespace properties resolve from a named worker environment variable; no inline credentials |

## Configuration and compatibility

| Contract | Status |
| --- | --- |
| Python | `>=3.12,<3.14` |
| Configuration files | JSON |
| YAML configuration | Rejected |
| Public stability source | `@PublicAPI` and the stability inventory |
| Versioning | Semantic Versioning |
| Third-party Provider `normalize()+open()` | Independent Provider SPI retained; canonical Gateway requires `plan()` plus `EngineBinding`, and does not fallback or emit the removed adapter warning |
| Third-party bounded source | `ProviderSourceConfig` plus versioned Provider/Binding descriptors; no consumer-module source branches |

For symbol-level compatibility promises, consult the
[API stability inventory](../STABILITY.md).
