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
| Full runtime image for CPU validation | Pinned Ray/uv base, locked first-party extras, Alpha import closure, and `ImageProfile` attestation | `unit` contracts | `runtime-image` |

External suite names describe required evidence, not GitHub Actions jobs.
Quarantined MLflow, serving, streaming, and legacy standalone tests do not
count as capability evidence until their lifecycle and reliability conditions
are repaired.

For distributed algorithms, compatible profiles are declared by the static
contract. Validated profiles have additionally passed the corresponding real
environment Gate. The project-wide distributed algorithm Gate uses Ray Jobs on
an isolated Docker cluster with two worker nodes to verify sharding, state
coordination, cross-node receipts, and Bundle atomicity. This verifies the
deployment-neutral `cluster` execution profile on an existing Ray cluster.
Kubernetes remains an external substrate managed by KubeRay or another Ray
platform, not a Tributo profile or control plane. Tributo's distributed
evidence is the deployment-neutral `cluster` profile on an isolated Docker Ray
cluster. KubeRay deployment and lifecycle validation are outside the Tributo IT
scope; users may attach the same workload to an externally provided Ray Jobs
endpoint.

## Data

| Capability | Status | Boundary |
| --- | --- | --- |
| Local/S3 Parquet and CSV reads | Verified | Native Ray Data or Daft handle through one Gateway |
| Local/S3 Iceberg reads | Verified | Built-in bindings use PyIceberg `>=0.11.1,<0.12.0` with `PyArrowFileIO`; Ray may push `row_filter` into the scan, Daft applies it as a lazy residual filter, and empty-table schema is preserved from Iceberg metadata; broader Catalog/delete-file matrix remains gated |
| Local/S3 Lance reads | Verified | Native Ray Data or Daft table reader; numeric versions and tags are supported, Daft also supports as-of, and Iceberg snapshot references fail closed |
| PostgreSQL structured table reads | Verified | Ray uses a single public SQL read and fails closed on parallel shard requirements; Daft may use native partition hints |
| ClickHouse/Doris raw SQL | Unsupported | Legacy shapes return a credential-free migration error; use structured table input or execute SQL outside Tributo ingestion |
| HDFS Parquet/CSV reads | Adapter only | Ray binding exists; real HDFS/JVM/worker gate is pending |
| ClickHouse reads | Adapter only | Uses locked `daft-clickhouse==1.0` through `tributo[clickhouse]`; real-database Conformance is still required, and provider partition discovery is distinct from engine auto-routing |
| Doris reads | Adapter only | Ray routes use locked `ray-doris==1.0`; Daft routes use locked `daft-doris==1.0`; real-database Conformance is still required and tablet planning remains provider/binding-owned |
| ORC and Hive external-table reads | Not implemented | The full image includes external `ray-hive==1.0` for HiveServer2 reads; Tributo Provider/Binding routing and real Hive Conformance remain pending |
| Third-party ingestion Provider/Binding SPI | Implemented | Installed packages use `tributo.ingestion_providers` plus `tributo.ingestion_bindings`; bad plugins are isolated, duplicate routes never replace built-ins, and Binding selection can constrain filesystem, catalog, and storage format |
| Lance output | Implemented as a generic ResultSink path | Declared fixed-shape Ray/Arrow tensor columns are strictly validated and normalized to Lance `FixedSizeList`; the user Predictor still owns vector semantics, and the sink does not pool, mathematically normalize embedding values, change dtype, or automatically invoke the separate vector-index workflow |
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
| <code>difference_in_means_ate</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.causal.difference_in_means</code> | <code>ray_map_reduce</code> | <code>ray_map_reduce</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>dnn</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.tabular_torch.dnn</code> | <code>ray_train_recipe_v2</code> | <code>ray_train_recipe_v2</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>doubly_robust_ate</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.causal_dr.aipw</code> | <code>framework_native</code> | <code>framework_native</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>dowhy_linear_refutation</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.causal_dowhy.linear_refutation</code> | <code>framework_native</code> | <code>framework_native</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>extra_trees</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.extra_trees.joblib</code>, <code>tributo.official.extra_trees.native_ensemble</code> | <code>ray_joblib_estimator</code>, <code>ray_parallel_ensemble</code> | <code>ray_joblib_estimator</code>, <code>ray_parallel_ensemble</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>gcm_root_cause</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.causal_dowhy.gcm_root_cause</code> | <code>framework_native</code> | <code>framework_native</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>graphsage_node_classifier</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.graph_pyg.graphsage</code> | <code>framework_native</code> | <code>framework_native</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>jagged_embedding_recommender</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.recsys_torch.jagged_embedding</code> | <code>framework_native</code> | <code>framework_native</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>linear_dml_ate</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.causal.linear_dml</code> | <code>ray_map_reduce</code> | <code>ray_map_reduce</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>linear_iv_ate</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.causal.linear_iv</code> | <code>ray_map_reduce</code> | <code>ray_map_reduce</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>linear_regression</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.linear_regression.squared_l2</code> | <code>ray_iterative_optimization</code> | <code>ray_iterative_optimization</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>logistic_regression</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.logistic_regression.binary_l2</code> | <code>ray_iterative_optimization</code> | <code>ray_iterative_optimization</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>multinomial_nb</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.multinomial_nb.map_reduce</code> | <code>ray_map_reduce</code> | <code>ray_map_reduce</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>pc_stability_discovery</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.causal_discovery.pc_stability</code> | <code>ray_map_reduce</code> | <code>ray_map_reduce</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>pretrain_finetune_classifier</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.multistage_torch.pretrain_finetune</code> | <code>framework_native</code> | <code>framework_native</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>pu</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.tabular_torch.pu</code> | <code>ray_train_recipe_v2</code> | <code>ray_train_recipe_v2</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>random_forest</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.random_forest.joblib</code>, <code>tributo.official.random_forest.native_ensemble</code> | <code>ray_joblib_estimator</code>, <code>ray_parallel_ensemble</code> | <code>ray_joblib_estimator</code>, <code>ray_parallel_ensemble</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>rgcn_node_classifier</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.graph_pyg.rgcn</code> | <code>framework_native</code> | <code>framework_native</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>tabular_autoencoder</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.representation.tabular_autoencoder</code> | <code>ray_train_recipe_v2</code> | <code>ray_train_recipe_v2</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>teacher_student_distillation</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.multistage_torch.distillation</code> | <code>framework_native</code> | <code>framework_native</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>temporal_conv_classifier</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.timeseries.temporal_conv</code> | <code>ray_train_recipe_v2</code> | <code>ray_train_recipe_v2</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>token_transformer_classifier</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.transformer.token_classifier</code> | <code>ray_train_recipe_v2</code> | <code>ray_train_recipe_v2</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>two_tower_recommender</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.recsys_torch.two_tower</code> | <code>ray_train_recipe_v2</code> | <code>ray_train_recipe_v2</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>x_learner</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.causal_xlearner.xgboost</code> | <code>framework_native</code> | <code>framework_native</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
| <code>xgboost</code> | <code>ready</code> | <code>alpha</code> | <code>available</code> | <code>no</code> | <code>no</code> | <code>fit</code> | <code>tributo.official.boosting.xgboost</code> | <code>framework_native</code> | <code>framework_native</code> | <code>cluster</code>, <code>local</code> | — | <code>ray_data</code> | — |
<!-- END GENERATED: TRIBUTO ALGORITHM SUPPORT -->

### Training infrastructure and planned capabilities

| Capability | Status | Boundary |
| --- | --- | --- |
| Ray Tune | Beta | Capability-gated algorithms only |
| Low-code PyTorch recipe | Alpha | Trusted dense-tabular packages define model/loss/optimizer/metric factories; Tributo lowers them to Ray Train and the existing ONNX Bundle path |
| Legacy managed sklearn and Custom Ray Function | Alpha compatibility | Joblib and legacy `data_parallel` remain compatibility mechanisms and do not prove distributed model training |
| Portable distributed execution | Alpha | Explicit collective, framework-native, and bounded tree-MapReduce strategies; owned local and attached cluster profiles share one contract |
| Constrained algorithm descriptor SPI | Alpha | Trusted packages from the selected image or a validated Job artifact; no arbitrary dependency resolution, isolation, hot reload, or PluginManager lifecycle |
| Algorithm Wheel distribution | Alpha | Image Profiles plus code-only `py_modules` Wheels by default; opt-in offline Wheelhouse installs use `--no-index`, an attested manifest, and the existing entry-point registry. No online dependency resolution or untrusted-code sandbox is provided |
| Graph training | Alpha skeleton | No built-in PyG/DGL trainer |
| X-Learner causal estimation | Alpha, conformance-tested | Binary treatment/outcome, numeric tabular features, deterministic 5-fold cross-fitting over five native Ray Train XGBoost stages, causal report, and batch CATE inference; multi-node Gate pending |
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

| Flavor | Exportable | Bundle readable | Batch inference | Online serving | Native attribution | Boundary |
| --- | --- | --- | --- | --- | --- | --- |
| `onnx-runtime-v1` | Yes | Yes | Yes | Yes | No | Typed signature and safe ONNX Runtime loader; model-agnostic attribution requires an explicit reference |
| `xgboost-native-v1` | Yes | Yes | Yes | Yes | Conditional TreeSHAP | Canonical `ubj` and `xgboost-json` formats share this safe Booster runtime; runtime validation proves prediction parity and eligible tree models prove exact margin reconstruction |
| `x-learner-v1` | Yes | Yes | Yes | No | No | Fixed five-Booster X-Learner composition with named CATE/component outputs and integer quadrant codes |
| `report` | Yes | Yes | No | No | No | JSON causal report role; readable but non-executable |
| `safetensors-v1` | Yes | Yes | No | No | No | Weights-only; no trusted architecture loader |
| `torch-export-v1` | Yes | Yes | No | No | No | PT2 loader and version/device contract are pending |
| `hf-onnx-v1` | Yes | Yes | No | No | No | Dedicated runtime compatibility gate is pending |
| `onnx-int8-v1` | Yes | Yes | No | No | No | Quantized numerical compatibility gate is pending |

Executable decisions come from the plugin-derived capability registry. The
frozen support matrix remains a compatibility/documentation view and is not
used by inference core to branch on model formats.

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

## Runtime images

| Capability | Status | Boundary |
| --- | --- | --- |
| Full Tributo runtime image for CPU validation | Alpha, directly buildable | Linux `arm64`/`amd64`, defaulting to the native host architecture; Python 3.12, Ray 2.55.1, locked dependency closure, and all first-party runtime extras including Alpha modules. Linux PyTorch resolution may include transitive CUDA/NVIDIA distributions, but no GPU support is claimed |
| Custom connector wheelhouse variant | Alpha, optional extension | An external wheelhouse remains available for packages outside the locked v1.0 connector set; it is not required for the canonical ClickHouse/Doris/Ray Hive image and does not change the Tributo lockfile |
| Runtime image attestation | Alpha | `manifest.json`, `image-profile.json`, normalized distribution inventory, and sealed `org.tributo.manifest-sha256` label |
| Runtime image Ray Jobs gate | Alpha, validation gate | Requires a unique two-node Docker Ray cluster on a native host architecture; verifies driver/worker imports, Ray Data, Jobs API submission, and v1.0 ClickHouse/Doris/Ray Hive package presence. `linux/amd64` requires a matching native host |
| GPU runtime image | Not implemented | The Linux dependency closure may contain transitive CUDA/NVIDIA distributions from PyTorch, but no GPU driver, scheduling, NCCL, or GPU compatibility contract has been validated |
| HDFS/ORC/Hive runtime additions | External package only | `ray-hive==1.0` is locked into the full image; no validated Tributo-native Provider/Binding or Hive runtime contract is claimed |

For symbol-level compatibility promises, consult the
[API stability inventory](../STABILITY.md).
