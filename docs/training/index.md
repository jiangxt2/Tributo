# Trainer compatibility guides

Tributo trainers execute through Ray Train and publish validated model
bundles. Trainer implementations own configuration validation, training,
checkpoint handling, and bundle export.

Use the [Algorithms and training](../algorithms/index.md) component as the main
entry point. These pages retain the legacy Trainer organization and published
URLs for compatibility.

- [Configure distributed training](../how-to/training.md).
- [Train a PU model](../how-to/pu-learning.md).
- [Add a distributed algorithm](../how-to/custom-distributed-algorithms.md).
