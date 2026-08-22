# Ray-first torch recipes

`TorchTrainingRecipe` is the low-code path for scalar-column dense tabular
PyTorch models.
An algorithm package defines model, loss, optimizer, and metric factories.
Tributo lowers that recipe to the existing formal collective runtime and keeps
deployment, data movement, distributed coordination, checkpoint upload, and
Bundle publication out of user code.

## Ray reuse audit

The implementation is bound to Ray 2.55.1 and its default Train V2 path. In
this release `RAY_TRAIN_V2_ENABLED` defaults to true; Tributo does not override
it to force V1. Source conclusions are based on the `ray-2.55.1`
implementation and its public annotations, not unversioned latest docs.

| Concern | Ray capability reused | Tributo responsibility |
|---|---|---|
| Worker group and process group | Stable `TorchTrainer` and `TorchConfig` | Validate descriptor topology and select the declared backend |
| Dataset assignment | Stable, subclassable `DataConfig`; public `Dataset.streaming_split()` | A version-locked `ExactCoverageDataConfig` selects `equal=False` because Ray 2.55.1 hard-codes `equal=True` in `DataConfig.configure()` |
| Torch batches | Public `DataIterator.iter_torch_batches()` | Validate feature/label roles and pass bounded batch options |
| Device and DDP | Stable `ray.train.torch.prepare_model()` | Reject unvalidated BatchNorm and GPU claims |
| Metrics | PyTorch tensors and `torch.distributed` collectives | Apply the descriptor's existing `MetricReduction`; do not infer reducers from names |
| Checkpoint transfer | Ray `Checkpoint`, `train.report()`, `RunConfig`, and retention | Write bounded model/optimizer/RNG metadata and validate epoch-boundary resume identity |
| Run identity | V2 `RunConfig.name` and storage context | Derive a unique Ray run name from the existing Tributo `run_id` so independent runs cannot inherit each other's checkpoints |
| Export and serving | Existing Torch ONNX exporter, ONNX Runtime validator, Bundle publisher, reader, batch inference, and Serve flavor | Reconstruct the trusted recipe model and create one `ExportSource` |
| Local execution | `ray.init(address="local")` | Own and close only the runtime created by Tributo |
| Existing or provisioned cluster | `ray.init(address="auto")`, Ray Jobs, KubeRay RayJob, or Cluster Launcher | Attach or expose the same entrypoint; never provision or delete the cluster |

Ray's public API has no `DataConfig(equal=False)` option in 2.55.1. The custom
`configure()` override is therefore isolated under
`tributo.integrations.algorithm_runtimes`, covered by fixed-version tests, and
changes only the `equal` argument while preserving Ray execution options,
training-resource exclusion, and locality hints. A future Ray public option
should replace this adapter.

V2 rejects a non-null `resume_from_checkpoint` argument as deprecated. Worker
recovery within one Ray run still uses `train.get_checkpoint()`. For an
explicit new-run resume, Tributo passes the validated, worker-visible epoch
checkpoint path to the recipe and restores model, optimizer, and RNG state in
the worker. This does not replace Ray's checkpoint upload, retention, or
failure-recovery controller.

## Uneven input decision

Exact coverage creates unequal shard lengths. PyTorch DDP Join can shadow DDP
forward/backward collectives, but the default recipe also needs the global
active sample count on every optimization step to produce a sample-weighted
gradient. Join does not mirror that additional collective.

The first implementation therefore uses a narrow dynamic alignment protocol:

- every rank must receive at least one batch;
- all ranks exchange only the active row count before each step;
- exhausted ranks run a zero-row DDP forward/backward and contribute zero
  gradient while active ranks retain every observed row;
- loss scaling uses the global active row count, so uneven final batches are
  sample weighted;
- no row is dropped or replayed;
- all ranks report once per epoch and only rank zero attaches the replicated
  checkpoint.

This protocol supplements domain correctness. It does not replace Ray Dataset
sharding, PyTorch DDP, process-group setup, or Ray checkpoint transport. Empty
shards remain unsupported until a separate gate proves their semantics.

## User and framework boundaries

The default recipe consumes a pre-bound `train` Ray Dataset. Validation and
test datasets are optional roles when the input adapter can provide them;
the recipe does not split data. The first-party DNN recipe keeps its existing
split behavior in a thin adapter before the common worker loop.

The first `tabular_batch` profile accepts scalar numeric feature columns. The
worker stacks them into one dense tensor, while the export adapter restores the
original column names as separate ONNX inputs so Bundle inference does not lose
the declared `InputBinding` contract. Vector, sequence, jagged, and graph inputs
remain later profiles rather than implicit shape inference.

Advanced models may override the bounded `forward()` and `compute_loss()`
methods. Models that need custom training steps, multiple optimizers, framework
callbacks, graph sampling, sharded embeddings, or framework-owned checkpoints
use a framework adapter or the existing full worker-loop SPI.

Capabilities not described by this contract are not implied by the recipe,
dependency set, or framework installation.

## Evidence status

Unit evidence covers descriptor lowering, exact-coverage configuration,
streaming batch consumption, typed metric reduction, resumable checkpoint
contents, trusted model reconstruction, ONNX Bundle publication, execution
profile compatibility, and isolated wheel construction. The isolated Recipe
wheel has also passed owned-local single/two-worker and existing Docker Ray
cluster multi-node Ray Jobs gates, including 65-row uneven input and Bundle
publication. KubeRay 1.6.0 has also passed the public provision-substrate gate
on an isolated kind cluster: RayJob created the RayCluster, submitted the same
Tributo `cluster` workload, exposed status and logs, completed Bundle
publication, and removed the RayCluster through
`shutdownAfterJobFinishes`. This is substrate evidence and does not imply that
every algorithm was rerun on Kubernetes.
The reproducible test-only entrypoint is
`scripts/run_kuberay_algorithm_it.sh`; it owns one uniquely named kind cluster
and delegates RayCluster creation and cleanup to KubeRay.
