# Train an X-Learner uplift model

Use the first-party `x_learner` algorithm for binary-treatment, binary-outcome
uplift modeling with XGBoost base learners. The implementation runs the same
five Ray Train stages in the owned-local and attached-cluster profiles.

## Prepare the data

The input must contain:

- one unique non-empty integer or string identity column;
- a treatment column whose values are exactly `0` or `1`;
- an outcome column whose values are exactly `0` or `1`;
- one or more numeric feature columns.

The generic execution envelope projects only `input.features` plus
`input.label`. Include treatment and identity in `input.features` so they reach
the causal algorithm, but list only predictive features in
`algorithm_config.data.feature_columns`. Treatment and identity are never
passed to the five Boosters as model features.

## Run the formal algorithm

```{literalinclude} ../examples/doc_code/x_learner_execution.json
:language: json
:caption: x_learner_execution.json
```

Run against an owned local Ray runtime:

```bash
tributo algo run --config x_learner_execution.json
```

To run the same workload on a Ray cluster, change only `profile` to `cluster`
and submit the command through Ray Jobs. Tributo attaches with
`ray.init(address="auto")`; KubeRay, Cluster Launcher, or another Ray-native
provider owns cluster creation and cleanup.

## Understand the model

The implementation fits five native `ray.train.xgboost.XGBoostTrainer` stages:

- `mu0` and `mu1` estimate control and treated response probabilities;
- `tau0` fits `mu1(X) - Y` on control rows;
- `tau1` fits `Y - mu0(X)` on treated rows;
- `propensity` estimates `P(T=1 | X)` on the full training split.

The fixed combination is:

```text
cate = propensity * tau0 + (1 - propensity) * tau1
```

Tributo uses Ray Data's streaming identity-hash split, with the configured seed
included in the hash key. Pseudo-outcome and holdout prediction use stateful Ray
Data batch actors so each actor loads its Booster state once. Ray Train owns
worker groups, Rabit coordination, data shards, and stage checkpoints.
Before each stage starts, Ray Data materializes its transformed input into the
Object Store/spill layer so preprocessing does not require undeclared CPU while
the XGBoost worker group is reserved. Sequential stages also use Ray Train's
native coordinator; owned-local execution therefore requires one additional
Ray control-plane CPU beyond `worker_count`. Attached clusters expose the same
demand to Ray Autoscaler instead of using a Tributo cluster adapter.

## Interpret the output

Training publishes one Bundle with two roles:

- `inference`: a safe `x-learner-v1` artifact containing exactly five UBJ
  Boosters and fixed composition metadata;
- `causal_report`: the existing JSON causal report artifact.

The result reports model-mean CATE as ATE, an uplift curve, AUUC, a Qini curve,
raw Qini area, random-baseline-adjusted Qini, and four quadrant counts. Curves
are computed only on the independent test split and ties are ordered by
identity.

The quadrant rule compares `mu0` and `mu1` with `response_threshold`:

| Control response | Treated response | Quadrant | Bundle code |
| --- | --- | --- | --- |
| Low | High | `persuadable` | `1` |
| High | High | `sure_thing` | `3` |
| Low | Low | `lost_cause` | `0` |
| High | Low | `sleeping_dog` | `2` |

Batch inference uses the existing `InferenceRequest` and
`BundleBatchPredictor` contracts. Bind ordered feature columns to the
`float_input` tensor and select any of `mu0`, `mu1`, `tau0`, `tau1`,
`propensity`, `cate`, or `quadrant` as named outputs. The flavor is intentionally
batch-only; no online-serving support is claimed.

## Know the first-release boundary

This implementation uses deterministic five-fold cross-fitting by default;
`training.cross_fit_folds` may select a value from 2 to 20. Every fold must
contain treated and control examples. Sample weights, continuous treatment or
outcome, multiple treatments, automatic retries, and training resume remain
outside this adapter. It is an X-Learner, not a doubly robust or orthogonal
estimator.
