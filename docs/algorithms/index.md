# Algorithms and training

Tributo provides a formal, framework-neutral algorithm execution contract and
a compatibility Trainer lifecycle. First-party formal implementations cover
distributed XGBoost, DNN, positive-unlabeled learning, and Multinomial Naive
Bayes.

## Start with a task

```{toctree}
:maxdepth: 1

getting-started
key-concepts
../how-to/training
../how-to/pu-learning
../how-to/custom-distributed-algorithms
../training/index
```

Use the formal `tributo algo run` path for algorithm integrations. Use
legacy Trainer APIs only when maintaining an existing integration. Ray Tune
trials perform setup and fit; they do not publish the production Bundle.

See the [generated algorithm matrix](../reference/support-matrix.md#registered-algorithms)
for implementation, topology, profile, and validation evidence.
See the generated
[Algorithms and training API](../reference/api/algorithms-training.md) for
public signatures and stability labels.
