# Train with positive-unlabeled data

Train classifiers when you have confirmed positive examples and an unlabeled
population, as in fraud detection, churn prediction, and identity resolution.

## Why PU learning

In many real-world scenarios, you have:

- A small set of **confirmed positives**, such as known fraud cases.
- A large set of **unlabeled** samples that are not confirmed negatives.

Standard supervised learning treats the unlabeled population as negative,
which produces biased models. PU learning estimates risk without making that
assumption.

## Run formal distributed PU training

Use one execution envelope for the formal collective implementation:

```{literalinclude} ../examples/doc_code/pu_execution.json
:language: json
:caption: pu_execution.json
```

Run the request on a local Ray runtime:

```bash
tributo algo run --config pu_execution.json
```

The formal PU implementation uses the shared DNN/PyTorch DDP kernel. It splits
positive and unlabeled rows into global stratified train and validation
datasets before Ray Train shards each split. Risk and metrics are reduced
across workers, and rank zero owns the checkpoint. Every positive and unlabeled
split must contain enough rows to give each requested worker a non-empty shard.
Invalid input fails before training.

The `kubernetes` profile uses the same envelope inside an existing KubeRay
RayJob. Only the `local` profile accepts `local_runtime`.

## Use the compatibility Trainer

`PUTrainerImpl` is a Beta compatibility entry that uses the same shared
DNN/PU DDP worker kernel and accepts `ray.num_workers` greater than one. Direct
callers provide datasets and Trainer configuration instead of the formal
execution envelope, and pass the Bundle destination separately with
`BundleOutputConfig`. `output.onnx_path` belongs only to the deprecated
raw-artifact path and is not a Bundle URI.

## Choose a loss

| Loss | Description | When to use |
|---|---|---|
| `nnpu` | Non-negative PU learning loss. More stable, recommended default. | Most cases. |
| `upu` | Unbiased PU learning loss. Can produce negative loss values. | When prior is accurately known. |

Each optimization batch must contain both confirmed-positive and unlabeled
examples. The canonical PU adapter enforces this through the shared DNN/PU
kernel's deterministic paired sampler. Direct callers of `PULoss` must enforce
the same invariant; otherwise the loss fails closed instead of producing a
biased partial risk.
When validation is enabled, Tributo shuffles and proportionally splits P and U
separately before worker assignment. Both classes must therefore have enough
rows for every worker in both resulting splits. Set `training.val_size` to `0`
when validation must be disabled for a very small dataset.

## Set the class prior

An explicit `pu.class_prior` must be in the open interval `(0, 1)`. It is the
estimated proportion of true positives in the population, not the fraction of
observed labeled rows. When `class_prior_method="label_frequency"` is selected
and the explicit value is omitted, Tributo computes the observed-label
frequency with a global reduction. Callers should use this method only when its
domain assumption is valid.

The standalone helpers in `tributo.training.priors` can support an upstream
estimation workflow. Their result must be reviewed and passed explicitly to
the trainer. The compatibility field `class_prior_method` records provenance;
it does not trigger estimation during training. A non-default legacy value
emits a migration warning, but the explicit `class_prior` remains the only
value used by the loss.

```{warning}
Do not use the raw labeled-positive frequency unless the labeling mechanism
makes that value a defensible population prior.
```

## Interpret metrics

The formal and compatibility training kernels report the following facts
through Ray Train and in the final result:

| Metric | Description |
|---|---|
| `epoch` | Completed training epoch. |
| `train_loss` | Complete training-split uPU risk or Eq. 6 nnPU risk. |
| `train_optimization_objective` | Algorithm 1 optimization surrogate used for backpropagation. It can differ from `train_loss` in the nnPU correction region. |
| `train_observed_label_accuracy` | Diagnostic agreement with the observed positive/unlabeled indicator. It is not population classification accuracy. |
| `train_acc` | Deprecated compatibility alias for `train_observed_label_accuracy`; it is not population classification accuracy. |
| `val_loss` | Complete validation-split uPU risk or Eq. 6 nnPU risk, when validation is enabled. |
| `val_observed_label_accuracy` | Validation diagnostic against observed indicators, when validation is enabled. |
| `val_acc` | Deprecated compatibility alias for `val_observed_label_accuracy`, when validation is enabled. |
| `class_prior` | Explicit population positive-class prior used by the loss. |

Standard accuracy is not a valid quality measure for PU learning. Tributo also
provides standalone post-training evaluation helpers:

| Metric | Description |
|---|---|
| `pu_precision` | Precision adjusted for the unlabeled set. |
| `pu_recall` | Recall against known positives. |
| `pu_f1` | Harmonic mean of PU precision and recall. |
| `pu_auc` | Area under the PU-ROC curve. |

These `pu_*` metrics are computed by `tributo.training.pu_metrics`; the Trainer
does not emit them automatically in v1.0.0.

## See also

- `src/tributo/training/priors.py`: standalone prior estimation utilities.
- `src/tributo/training/pu_metrics.py`: PU evaluation metrics.
- `src/tributo/training/pu_trainer.py`: formal worker kernel and compatibility
  Trainer implementation.
