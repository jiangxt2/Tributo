# Ray-first torch recipes

Tributo exposes one versioned `TorchRecipe` for Core-owned training loops and
one `RayTorchAdapter` for framework-owned loops. Both are selected by the
`RAY_TRAIN_TORCH` strategy and executed by `tributo.ray_train_torch`.
`TorchPolicy.execution_plan` is the single source of truth for routing, Stage
order, checkpoint dependencies, state layout and final export Stage.

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
| Checkpoint transfer | Ray `Checkpoint`, `train.report()`, `RunConfig`, and retention | Write bounded model, optimizer, scheduler, gradient scaling, and RNG metadata at complete optimizer-window boundaries and validate Stage identity |
| Run identity | V2 `RunConfig.name` and storage context | Derive a unique Ray run name from `run_id`, `invocation_id`, `stage_id`, and the Policy/execution-plan identity digest so retries reuse only the matching Stage directory |
| Export and serving | Existing Torch ONNX exporter, ONNX Runtime validator, Bundle publisher, reader, batch inference, and Serve flavor | Reconstruct the trusted recipe model and create one `ExportSource` |
| Local execution | `ray.init(address="local")` | Own and close only the runtime created by Tributo |
| Existing or provisioned cluster | `ray.init(address="auto")`, Ray Jobs, KubeRay RayJob, or Cluster Launcher | Attach or expose the same entrypoint; workload code never provisions or deletes the cluster |

Ray's public API has no `DataConfig(equal=False)` option in 2.55.1. The custom
`configure()` override is therefore isolated under
`tributo.integrations.algorithm_runtimes`, covered by fixed-version tests, and
changes only the `equal` argument while preserving Ray execution options,
training-resource exclusion, and locality hints. A future Ray public option
should replace this adapter.

Ray Train V2 rejects a non-null `resume_from_checkpoint` argument as deprecated.
The Core Runtime never supplies it: failure retry uses `train.get_checkpoint()`;
cross-Stage and cross-Run recovery use a credential-free Core Worker control
envelope. A full `TorchRecoveryEnvelope` records completed Stage locators and an
optional active Stage; completed Stage evidence is persisted in the validated
checkpoint sidecar so a recovery-only invocation can produce the same Receipt.
A typed progress manifest preserves per-rank coverage/metric prefixes, reducer
observation, Dataset cursor and whether the current epoch's Scheduler step has
already run.  Remote Stage payloads remain unavailable under a deterministic
staging prefix until a matching commit marker is written.
A Torch-only preflight and invocation-local one-shot lease complete before Ray
or input resources are opened.

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
- loss scaling uses the algorithm-declared global normalizer for the complete
  accumulation window, so uneven final batches remain mathematically weighted
  by the declared unit (rows, sample weights, or valid tokens);
- no row is dropped or replayed;
- all ranks report at the declared `checkpoint_interval_windows` cadence (one
  complete optimizer window by default), and only the checkpoint-owner rank
  attaches the replicated checkpoint.

Optional validation and test roles use the same rank-alignment rule.  When a
present role has fewer rows than Workers, exhausted ranks execute a typed
zero-row validation step and contribute zero metrics; all ranks still perform
the same metric collectives.  A role that is absent remains explicitly absent,
while an explicitly empty role is rejected by the route contract.

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

Advanced models implement the typed `TorchRecipe` hooks. Models that need
custom training steps, framework callbacks, graph sampling, sharded embeddings,
or framework-owned checkpoints use `RayTorchAdapter`; the Adapter still cannot
create a nested Trainer or declare a second execution plan.

An Adapter declares its `TorchArtifactPlan` through the Core Provider.  The
Provider attaches that plan to the Adapter's `ExportSource` before invoking the
single Core Bundle exporter, and rejects conflicting plan metadata.  Adapter
worker configuration receives algorithm-owned values plus credential-free input
binding metadata; Core Ray paths, recovery locators, and output publication
URIs are never accepted in that JSON payload.

Capabilities not described by this contract are not implied by the recipe,
dependency set, or framework installation.

## Evidence status

Unit evidence covers descriptor lowering, exact-coverage configuration,
streaming batch consumption, typed metric reduction, resumable checkpoint
contents, trusted model reconstruction, ONNX Bundle publication, execution
profile compatibility, and isolated wheel construction. The isolated Recipe
wheel has also passed owned-local single/two-worker and existing Docker Ray
cluster multi-node Ray Jobs gates, including 65-row uneven input and Bundle
publication. Kubernetes and KubeRay remain external deployment substrates; the
deployment-neutral Docker Ray multi-node gate is the Tributo evidence for the
common `cluster` workload. This does not claim that Tributo owns or validates
Kubernetes control-plane lifecycle.
