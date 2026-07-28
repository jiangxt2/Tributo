# PU Learning (Positive-Unlabeled)

Train classifiers when you only have positive labels — common in fraud detection, churn prediction, and identity resolution.

## Why PU Learning?

In many real-world scenarios, you have:
- A small set of **confirmed positives** (known fraud cases, known churners).
- A large set of **unlabeled** samples (users not yet flagged — they aren't confirmed negatives).

Standard supervised learning treats unlabeled as negative, which produces biased models. PU Learning handles this correctly.

## Quick Start

```python
from tributo.training.pu_trainer import PUTrainerImpl, PUTrainingConfig

config = PUTrainingConfig.from_json("pu_config.json")
trainer = PUTrainerImpl(config)
result = trainer.fit()
print(f"ONNX model: {result.metrics['onnx_path']}")
```

## Configuration

```json
{
  "data": {
    "type": "s3",
    "uri": "s3://bucket/training_data/*.parquet",
    "format": "parquet"
  },
  "model": {
    "label_col": "is_fraud",
    "pu": {
      "loss": "nnpu",
      "prior": 0.3,
      "auto_prior": true
    },
    "nn": {
      "hidden_dims": [128, 64, 32],
      "dropout": 0.3,
      "activation": "relu"
    }
  },
  "train": {
    "num_workers": 1,
    "num_epochs": 50,
    "batch_size": 1024,
    "learning_rate": 0.001
  },
  "export": {
    "onnx_output": "s3://bucket/models/pu_fraud_model.onnx"
  }
}
```

!!! note "num_workers"
    PU training currently runs on a single worker. DDP multi-worker training is not yet supported;
    set `num_workers` to 1.

## Loss Functions

| Loss | Description | When to Use |
|---|---|---|
| `nnpu` | Non-negative PU learning loss. More stable, recommended default. | Most cases. |
| `upu` | Unbiased PU learning loss. Can produce negative loss values. | When prior is accurately known. |

## Class Prior Estimation

Tributo provides three methods for estimating the positive class prior:

| Method | Description |
|---|---|
| `label_frequency` | Simple frequency-based estimate. Fast; assumes labeled positives are representative. |
| `histogram_match` | Matches score distributions between labeled and unlabeled. More accurate. |
| `em` | Expectation-Maximization. Most accurate; slower. |

Set `auto_prior: true` and specify the method with `prior_method`.

## Evaluation Metrics

Standard accuracy is meaningless for PU learning. Use PU-specific metrics:

| Metric | Description |
|---|---|
| `pu_precision` | Precision adjusted for the unlabeled set. |
| `pu_recall` | Recall against known positives. |
| `pu_f1` | Harmonic mean of PU precision and recall. |
| `pu_auc` | Area under the PU-ROC curve. |

Metrics are reported in real time via the training event stream.

## See Also

- `src/tributo/training/priors.py` — class prior estimation
- `src/tributo/training/pu_metrics.py` — PU evaluation metrics
- `src/tributo/training/pu_trainer.py` — PUTrainer implementation
