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
| Checkpoint transfer | Ray `Checkpoint`, `train.report()`, `RunConfig`, and retention | Report one completed-Stage checkpoint for Stage dependencies and export; retry replays the Stage |
| Run identity | V2 `RunConfig.name` and storage context | Derive a unique Ray run name from the logical run, Stage, and Policy/execution-plan identity |
| Export and serving | Existing Torch ONNX exporter, ONNX Runtime validator, Bundle publisher, reader, batch inference, and Serve flavor | Reconstruct the trusted recipe model and create one `ExportSource` |
| Local execution | `ray.init(address="local")` | Own and close only the runtime created by Tributo |
| Existing or provisioned cluster | `ray.init(address="auto")`, Ray Jobs, KubeRay RayJob, or Cluster Launcher | Attach or expose the same entrypoint; workload code never provisions or deletes the cluster |

Ray's public API has no `DataConfig(equal=False)` option in 2.55.1. The custom
`configure()` override is therefore isolated under
`tributo.integrations.algorithm_runtimes`, covered by fixed-version tests, and
changes only the `equal` argument while preserving Ray execution options,
training-resource exclusion, and locality hints. A future Ray public option
should replace this adapter.

Ray Train V2 rejects a non-null `resume_from_checkpoint` argument as deprecated,
so the Core Runtime never supplies it. Ray owns Worker failure retry and
Checkpoint persistence. Runtime API v1 replays the current Recipe or Adapter
Stage from its Dataset beginning instead of exposing Ray's retry Checkpoint to
algorithm code; a replayed component Stage still receives its predecessor Stage
Checkpoint. Core does not add a progress cursor or remote commit protocol.
Completed Stage checkpoints move directly through Ray objects to dependent
Stages and the Bundle exporter. Cross-Run recovery is not supported. Torch-only
environment validation still runs before Ray or input resources are opened,
without a separate token or lease state machine.

## Uneven input decision

Exact coverage creates unequal shard lengths. PyTorch DDP Join can shadow DDP
forward/backward collectives, but the default recipe also needs the global
active sample count on every optimization step to produce a sample-weighted
gradient. Join does not mirror that additional collective.

The first implementation therefore uses a narrow dynamic alignment protocol:

- every rank must receive at least one batch;
- all ranks exchange only an active-batch flag before each step;
- exhausted ranks run a zero-row DDP forward/backward and contribute zero
  gradient while active ranks retain every observed row;
- loss scaling uses the algorithm-declared global normalizer for the complete
  accumulation window, so uneven final batches remain mathematically weighted
  by the declared unit (rows, sample weights, or valid tokens);
- no row is dropped or replayed within a successful Stage attempt;
- all ranks report once when the Stage completes, and only the checkpoint-owner
  rank attaches the replicated checkpoint.

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

Recipe implementations construct their declared dense, fixed-window, token or
ID tensors from named scalar columns. LSTM and GRU export a two-dimensional
`[batch, window]` input and add the singleton channel inside the model, keeping
the existing format-neutral inference binding unchanged. Jagged and graph
shapes remain explicit Adapter contracts rather than implicit Core inference.

Advanced models implement the typed `TorchRecipe` hooks. Models that need
custom training steps, framework callbacks, graph sampling, sharded embeddings,
or framework-owned checkpoints use `RayTorchAdapter`; the Adapter still cannot
create a nested Trainer or declare a second execution plan.

An Adapter declares its `TorchArtifactPlan` through the Core Provider.  The
Provider attaches that plan to the Adapter's `ExportSource` before invoking the
single Core Bundle exporter, and rejects conflicting plan metadata.  Adapter
worker configuration receives algorithm-owned values plus credential-free input
binding metadata; Core Ray paths and output publication URIs are never accepted
in that JSON payload.

Capabilities not described by this contract are not implied by the recipe,
dependency set, or framework installation.

## Evidence status

Local contract evidence covers descriptor lowering, exact-coverage
configuration, streaming batch consumption, typed metric reduction, completed
Stage checkpoints, trusted model reconstruction, Bundle publication, and
execution-profile compatibility. Candidate-Wheel conformance and the final
Docker Ray multi-node gate remain required before the migrated algorithms are
declared validated. Kubernetes and KubeRay remain external deployment
substrates; Tributo does not own or validate their control-plane lifecycle.
Portable Torch Tune trials report the target metric only; the selected
parameters must be used for a separate formal training run that publishes the
final Bundle.
