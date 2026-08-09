# PU Learning (Positive-Unlabeled)

Train classifiers when you only have positive labels — common in fraud detection, churn prediction, and identity resolution.

## Why PU Learning?

In many real-world scenarios, you have:
- A small set of **confirmed positives** (known fraud cases, known churners).
- A large set of **unlabeled** samples (users not yet flagged — they aren't confirmed negatives).

Standard supervised learning treats unlabeled as negative, which produces biased models. PU Learning handles this correctly.

## Quick Start

```python
from tributo.training.pu_trainer import run_pu_training_with_config

result = run_pu_training_with_config(
    {
        "data": {
            "source": {
                "provider": "tributo.parquet",
                "uri": "s3://bucket/training-data/*.parquet",
            }
        },
        "features": [
            {"name": "account_age", "type": "dense"},
            {"name": "monthly_usage", "type": "dense"},
        ],
        "label_col": "is_confirmed_positive",
        "model": {"dnn_hidden_units": [128, 64, 32], "dnn_dropout": 0.3},
        "pu": {"loss_type": "nnpu", "class_prior": 0.3},
        "training": {"epochs": 50, "batch_size": 1024},
        "ray": {"num_workers": 1},
        "output": {"onnx_path": "/models/pu-fraud"},
    }
)
print(f"ONNX model: {result['onnx_path']}")
```

## Configuration

```json
{
  "data": {
    "source": {
      "provider": "tributo.parquet",
      "uri": "s3://bucket/training-data/*.parquet"
    }
  },
  "features": [
    {"name": "account_age", "type": "dense"},
    {"name": "monthly_usage", "type": "dense"}
  ],
  "label_col": "is_confirmed_positive",
  "model": {
    "dnn_hidden_units": [128, 64, 32],
    "dnn_dropout": 0.3
  },
  "pu": {
    "loss_type": "nnpu",
    "class_prior": 0.3
  },
  "training": {
    "epochs": 50,
    "batch_size": 1024,
    "learning_rate": 0.001
  },
  "ray": {
    "num_workers": 1
  },
  "output": {
    "onnx_path": "/models/pu-fraud"
  }
}
```

```{note}
PU training currently runs on a single worker. DDP multi-worker training is
not yet supported; set `num_workers` to 1.
```

## Loss Functions

| Loss | Description | When to Use |
|---|---|---|
| `nnpu` | Non-negative PU learning loss. More stable, recommended default. | Most cases. |
| `upu` | Unbiased PU learning loss. Can produce negative loss values. | When prior is accurately known. |

Each optimization batch must contain both confirmed-positive and unlabeled
examples. The built-in DNN and PU trainers enforce this with a deterministic
paired sampler. Direct callers of `PULoss` must enforce the same invariant;
otherwise the loss fails closed instead of producing a biased partial risk.
When validation is enabled, the input therefore needs at least two examples
from each observed group so both training and validation retain P/U coverage.
Set `training.val_size` to `0` when validation must be disabled for a very small
dataset.

## Class Prior

Training requires `pu.class_prior` in the open interval `(0, 1)`. It is the
estimated proportion of true positives in the population, not the fraction of
currently labeled rows. Tributo deliberately does not infer it from the
observed label frequency because labeled positives are normally a selected
subset of all positives.

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

## Evaluation Metrics

The PU Trainer reports the following training facts through Ray Train and in
the final result:

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
does not currently emit them automatically.

## See Also

- `src/tributo/training/priors.py` — standalone prior estimation utilities
- `src/tributo/training/pu_metrics.py` — PU evaluation metrics
- `src/tributo/training/pu_trainer.py` — PUTrainer implementation
