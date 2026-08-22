# Training on Ray

Run XGBoost, X-Learner, DNN, and PU through real distributed state coordination.
XGBoost and X-Learner use framework-native coordination; formal DNN lowers a first-party
`TorchTrainingRecipe` to the common Ray-owned worker loop, while PU retains its
specialized Ray Train/PyTorch DDP kernel. A one-worker local run is supported
but is not reported as distributed training.

Formal algorithms choose an explicit `local` or `cluster` execution profile.
The local profile owns an in-process Ray runtime. The cluster profile attaches
with `address="auto"`; KubeRay RayJob, Cluster Launcher, or an external manager
owns cluster creation and cleanup. The legacy JSON value `kubernetes` remains
an input alias for `cluster`, but new output is deployment neutral.
`DistributionSpec.supported_execution_profiles` declares implementation
compatibility; the separately reported validated profiles are the environments
that have direct environment evidence. Tributo's algorithm Gate runs through
the Ray Jobs API on an isolated Docker cluster with two worker nodes. It proves
deployment-independent sharding, state coordination, receipt, and Bundle
semantics without making Docker, Kubernetes, or VM providers execution profiles.

Ordinary third-party PyTorch models should implement `TorchTrainingRecipe`.
Advanced packages may implement `CollectiveAlgorithm`, `MapReduceAlgorithm`,
or `FrameworkNativeAlgorithm` and publish a
`DistributedAlgorithmDescriptor` through the `tributo.algorithms` entry-point
group. Ray Joblib is not a formal distribution strategy. Arbitrary sklearn
estimators are not automatically converted to distributed models.

## Default Bundle Publication

First-party XGBoost, DNN, and PU trainers publish an immutable Bundle by
default. Formal X-Learner runs also require an explicit Bundle destination.
Tributo never writes a default Bundle into the current directory or a temporary
directory.

```python
from tributo.exporting.models import BundleOutputConfig

summary = trainer.run(
    bundle_config=BundleOutputConfig(
        bundle_uri="s3://your-bucket/models/fraud-detector",
        storage_profile="production",
    )
)

print(summary["training_status"])
print(summary["bundle_status"])
print(summary["hook_status"])
print(summary["bundle_uri"])
print(summary["execution_id"])
```

Omitting `targets` selects the trainer's standard artifacts:

| Trainer | Default artifacts | Inference role |
|---------|-------------------|----------------|
| XGBoost | ONNX opset 12 and native UBJ in the same Bundle | `onnx-model` |
| DNN | ONNX opset 18 | `onnx-model` |
| PU | ONNX opset 18 | `onnx-model` |

DNN and PU ONNX artifacts also contain the checkpoint's
`preprocessor.json`, recorded with the Bundle file role `preprocessor` and
covered by the artifact digest. `IdentityPredictor` requires that state and
checks it against the model feature configuration and Manifest input
signature before accepting raw-feature requests.

The returned mapping includes the stable `TrainingResult` fields
`model_uri`, `bundle_uri`, `metrics`, `legacy_artifact_uri`,
`training_status`, `bundle_status`, `hook_status`, and `execution_id`.
Required Bundle or Hook failures are raised as typed errors and expose the
same terminal contract as `error.training_result`.

The old raw-artifact hooks are available only through
`trainer.run(output_path=..., legacy_export=True)`. That opt-in emits a
`DeprecationWarning` and must not be used for formal integrations.

## XGBoost Training

The following example reads Parquet through the canonical ingestion gateway,
trains XGBoost, and lets the first-party defaults publish ONNX and UBJ together:

```python
from tributo.data import (
    IngestionRequest,
    ParquetSourceConfig,
    RayDataHandle,
    open_ingestion,
)
from tributo.exporting.models import BundleOutputConfig
from tributo.training.xgboost_trainer import XGBoostTrainerImpl

train_input = open_ingestion(
    IngestionRequest(
        source=ParquetSourceConfig(path="s3://your-bucket/train/*.parquet"),
        engine="ray",
    )
)
assert isinstance(train_input.handle, RayDataHandle)

try:
    trainer = XGBoostTrainerImpl(
        datasets={"train": train_input.handle.dataset},
        config={
            "data": {"label_col": "label"},
            "model": {
                "objective": "binary:logistic",
                "max_depth": 6,
                "eta": 0.1,
            },
            "training": {"num_rounds": 200, "val_size": 0.2},
            "ray": {"num_workers": 4, "use_gpu": False},
        },
    )
    summary = trainer.run(
        bundle_config=BundleOutputConfig(
            bundle_uri="s3://your-bucket/models/xgboost",
            storage_profile="production",
        )
    )
    print(summary["bundle_uri"])
finally:
    train_input.close()
```

Tributo validates in-memory configuration with strict Pydantic contracts. JSON
is the built-in persisted format; deployment-specific parsers may convert other
formats to a mapping before validation.

Dictionary/JSON trainer entry points use `output.bundle_uri` for the default
Bundle lifecycle. The older `output.onnx_path`, `output.onnx_opset`, and
`output.metrics_path` fields belong to the deprecated raw-artifact lifecycle;
supplying `onnx_path` without `bundle_uri` selects that legacy path and emits a
`DeprecationWarning`. Combining `bundle_uri` with explicitly configured legacy
output fields is rejected; the two destinations are never interpreted as
aliases for one another.

## X-Learner Causal Training

Use the formal `x_learner` algorithm for binary-treatment, binary-outcome
uplift modeling. It composes five native Ray Train XGBoost fits, reports CATE,
four quadrants, Uplift, Qini, AUUC, and ATE, and publishes a batch-inference
Bundle. See [Train an X-Learner uplift model](x-learner.md) for its input roles,
fixed formula, configuration, output semantics, and limitations.

## DNN and PU Training

DNN and PU use the same Bundle call after their trainer has been constructed:

```python
from tributo.exporting.models import BundleOutputConfig

summary = dnn_trainer.run(
    bundle_config=BundleOutputConfig(bundle_uri="s3://your-bucket/models/dnn")
)
print(summary["bundle_uri"])
```

Set `ray.num_workers` to one for local single-model development or to two or
more for DDP. Every worker consumes a distinct Ray Dataset shard; metrics and
early stopping use global reductions and rank zero owns the consolidated
checkpoint. Formal DNN streams transformed batches through Ray Data; its
sparse vocabulary, normalization, preprocessor artifact, and split semantics
remain in a first-party thin adapter. Historical `DNN loss.type="nnpu"`
configuration is a compatibility alias routed to the canonical PU adapter, so
Tributo maintains only one PU training kernel.

`multinomial_nb` is the first sklearn-backed distributed example. It maps
disjoint shards to bounded `class_count` and `feature_count` statistics, merges
them in a balanced Ray reduction tree, finalizes with sklearn's public
`partial_fit` API, and publishes a validated ONNX Bundle. The runtime also
cross-checks the Driver's bounded input count against the total rows consumed
by all map workers before declaring complete coverage. Sparse feature batches
remain sparse until the bounded class-by-feature statistics are emitted. This
does not imply support for the complete sklearn library.

Use `tributo algo run --config execution.json` for the formal path. The JSON
envelope selects `profile: "local"` or `profile: "cluster"`; both profiles
use the same algorithm configuration and implementation. `local` creates an
owned Ray runtime in the calling process. A cluster invocation attaches to the
Ray runtime created by KubeRay, Cluster Launcher, or an external manager; it
does not submit a second job.

Start from `examples/distributed_algorithm_execution.json`. For a KubeRay job,
set only the envelope profile to `"cluster"`, keep the algorithm payload
unchanged, bake the JSON and installed Python packages into the image (or
mount them), and use `examples/kuberay/distributed-algorithm-rayjob.yaml` as
the deployment skeleton. The RayJob manifest is Kubernetes YAML; Tributo's
persisted algorithm configuration remains JSON. KubeRay owns deployment and
cluster lifecycle; Tributo owns the algorithm contract already exercised by
the Docker multi-node Ray Jobs Gate. KubeRay 1.6.0 RayJob has also passed the
common provision-substrate Gate on an isolated kind cluster; this verifies
cluster creation, submission, status/logs, and native cleanup rather than
validating every algorithm again on Kubernetes.
Maintainers can reproduce that substrate evidence with
`scripts/run_kuberay_algorithm_it.sh`; application users continue to invoke
KubeRay directly.

## Bundle Checkpoint Compatibility

DNN and PU checkpoints produced before the E2 export contract do not contain
the required `ExportCheckpointV1` metadata and cannot be exported through
`BaseTrainer.run(bundle_config=...)`. Regenerate those checkpoints with the
E2 trainer implementation before using Bundle export.

The legacy `export_model()` path remains available for existing checkpoints
only through the explicit `legacy_export=True` compatibility switch.

## S3 Authentication

Three methods, in order of preference:

1. **IAM Role** (recommended for production) — no config needed; workers assume the role.
2. **Environment variables** — set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`.
3. **Explicit config** — pass `s3` dict in the data config with `access_key_id`, `secret_access_key`, `endpoint`.

## Resource Tuning

| Parameter | Guidance |
|---|---|
| `worker_count` | Formal execution envelope: one for supported single-machine development, two or more for a receipt-verified distributed model. |
| `ray.num_workers` | Compatibility field for DNN/PU/XGBoost; when present it must equal formal `worker_count`. |
| `use_gpu` | Requires GPU workers and a GPU-capable framework build: XGBoost for tree training or PyTorch for DNN/PU. |
| `num_cpus_per_worker` | XGBoost option; default 1. Increase when each worker has spare CPU. |

## Resource Budget

The PU and Beta compatibility Trainer paths enforce an unconditional worker
materialization budget. Over-budget inputs **fail fast** (a
`ResourceBudgetExceededError` is raised before the unbounded concat) — data is
never silently truncated. Formal DNN uses the streaming Ray Data Recipe path
and does not materialize the full worker shard.

| Config field | Default | Meaning |
|---|---|---|
| `resource.max_batch_bytes` | 64 MiB | Per-batch size guard. |
| `resource.max_worker_materialization_bytes` | 1 GiB | Total bytes materialized per worker across all splits (includes the concat-copy peak). |
| `resource.max_input_rows_per_worker` | `null` (disabled) | Optional per-worker row guard; exceeding it fails fast instead of slicing. |

```json
"resource": {
  "max_batch_bytes": 67108864,
  "max_worker_materialization_bytes": 1073741824,
  "max_input_rows_per_worker": 1000000
}
```

> **Note**: `external_memory` and `data_iter` are reserved XGBoost
> parameters — they would bypass this budget contract and are rejected
> by the config.

## See Also

- Example: `examples/xgboost_s3_training.py`
