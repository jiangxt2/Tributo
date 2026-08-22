# X-Learner causal training architecture

## Purpose and boundary

The first-party X-Learner turns the existing local XGBoost demonstration into a
formal Tributo algorithm without introducing a generic causal DAG, checkpoint
system, composite-model platform, or cluster manager. It is a narrow consumer
of the existing Algorithm, Ray Data, Ray Train, Bundle, and batch-inference
contracts.

## Execution chain

```text
Ingestion Provider/Binding
  -> canonical Ray Dataset
  -> seeded identity-hash train/validation/test split
  -> mu0 + mu1 Ray XGBoost fits
  -> Ray Data pseudo outcomes
  -> tau0 + tau1 Ray XGBoost fits
  -> propensity Ray XGBoost fit
  -> Ray Data holdout predictions and stable CATE ordering
  -> uplift/Qini/AUUC/ATE and quadrant summary
  -> staged framework evidence
  -> ExportSource
  -> x-learner-v1 model + causal report
  -> existing Bundle publisher and batch inference
```

The Driver retains configuration, stage checkpoints, Booster bytes needed to
seed downstream prediction actors, and bounded metric summaries. It never
collects the complete training or prediction table.

Each transformed stage input is materialized through Ray Data before its Ray
Train worker group starts. Ray Object Store and spill own that intermediate;
this prevents predictor actors from competing with a fully reserved training
worker group and never moves rows into the Driver process. Because stages are
strictly sequential, one additional Ray control-plane CPU is sufficient beyond
the worker CPUs. Owned-local execution validates that headroom before opening
the stage inputs; attached clusters continue to rely on Ray's native resource
demand and Autoscaler.

## Framework ownership

| Concern | Owner |
| --- | --- |
| Source resolution, projection, receipts | Tributo Provider/Binding control plane |
| Identity-hash split, filtering, sorting, batch actors | Ray Data |
| XGBoost worker groups, Rabit, sharding, checkpoints | Ray Train and XGBoost |
| Exact uneven-row coverage | Existing `ExactCoverageDataConfig` thin adapter |
| Algorithm selection and local/cluster profile | Existing Tributo Dispatcher and runtime |
| Artifact planning, validation, publication, reading | Existing Tributo Bundle lifecycle |
| Cluster provisioning, autoscaling, cleanup | Ray-native external substrate |

Daft remains available for upstream ETL with an explicit versioned-storage
handoff. The algorithm requires a Ray Dataset and does not add an implicit or
streaming Daft-to-Ray bridge.

## Fixed causal composition

The algorithm declares exactly `mu0`, `mu1`, `tau0`, `tau1`, and `propensity`.
Stages execute sequentially because later pseudo-outcomes consume earlier
models. Tributo does not expose stage dependencies as a user-defined graph.

The final CATE formula is fixed to
`propensity*tau0+(1-propensity)*tau1`. A composition digest binds this formula,
feature order, threshold, propensity clip, and input-binding digest. The common
framework-native runtime combines it with the five synchronized model digests
and row counts.

## Evidence and delivery

`FrameworkNativePolicy.component_stages` is the only shared execution-contract
extension. An empty tuple preserves the existing one-model validator. A staged
algorithm must provide the exact declared stage set, complete per-stage worker
evidence, one full-input anchor stage, and a canonical composition digest. The
receipt stores bounded stage digests, rows, worker counts, and node counts.

The source provider opens exactly five Ray XGBoost checkpoints. The model
exporter writes five native UBJ files plus `x_learner.json`; the Bundle manifest
supplies file digests and path containment. The safe flavor rejects unknown
metadata versions, formulas, quadrant mappings, or component sets before
loading Boosters. The existing causal-report implementation is reused through
a thin exporter-API adapter, leaving its Beta compatibility API unchanged.

## Operational semantics

The same algorithm configuration runs with:

- `local`, where Tributo owns one local Ray runtime;
- `cluster`, where Tributo attaches to the Ray runtime already created for the
  Ray Job.

Every stage requires at least one training row per requested worker and must
prove exact row coverage. Missing treatment arms, invalid roles, duplicate
identities, malformed evidence, missing checkpoints, required export failure,
or an invalid Bundle all fail closed. Automatic retries and X-Learner training
resume are not part of this release.
