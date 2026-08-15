# Algorithm key concepts

## Registration and selection

`AlgorithmRegistration` combines an `AlgorithmSpec`, implementation descriptor,
environment contract, runtime binding, and default-selection flag. Selection is
deterministic. Duplicate or incompatible candidates fail closed.

## Distribution strategies

- Collective algorithms coordinate state across a fixed worker group.
- Framework-native algorithms delegate distribution to a framework such as
  XGBoost.
- MapReduce algorithms emit bounded partial state and reduce it through a tree.

Ray Joblib and arbitrary Ray tasks do not become formal distributed algorithms
without a matching descriptor and receipt contract.

## Input ownership

Input resolution happens before worker execution. A runtime adapter turns the
resolved input into worker payloads. Receipts record worker, node, and shard
evidence so a successful result can prove which data and topology ran.

## Training and Bundle publication

Fit produces a checkpoint or bounded artifact draft. A formal production run
publishes a validated Bundle once. Ray Tune trials are fit-only selection runs;
the caller starts a separate formal run with the chosen parameters.

## Compatibility Trainer lifecycle

`BaseTrainer` remains Beta for existing integrations. Formal first-party training
uses `BundleExportService`; `export_artifacts()` and `export_model()` are
compatibility hooks rather than extension points.
