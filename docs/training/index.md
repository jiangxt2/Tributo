# Training

Tributo trainers execute through Ray Train and publish validated model
bundles. Trainer implementations own configuration validation, training,
checkpoint handling, and bundle export.

```{toctree}
:maxdepth: 1

../how-to/training
../how-to/pu-learning
```

Use the training guide for distributed XGBoost and DNN workflows. PU learning
has additional prior-estimation and worker-count constraints documented in its
dedicated guide.
