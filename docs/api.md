# API Reference

This is a curated public entry point, not an inventory of every implementation
module. Signatures and docstrings come from the installed source tree.
Stability labels come from `@PublicAPI`.

See the [stability inventory](STABILITY.md) for the complete module-level
classification.

## Stable core API

```{autoclass} tributo.TributoClient
:members:
```

```{autoclass} tributo.RayJob
:members:
```

```{autoclass} tributo.JobConfig
:members:
```

### Core exceptions

```{autoexception} tributo.TributoError
```

```{autoexception} tributo.JobSubmissionError
```

```{autoexception} tributo.JobExecutionError
```

```{autoexception} tributo.JobConfigurationError
```

```{autoexception} tributo.JobTimeoutError
```

```{autoexception} tributo.ModelExportError
```

```{autoexception} tributo.DataSourceError
```

## Data provider API

This is the logical source-extension contract. Physical reads are delegated to
Ray Data, Daft, or an installed Connector through the bounded ingestion API.

```{autoclass} tributo.data.provider.DataSourceProvider
:members:
```

```{autoclass} tributo.data.provider.ResolvedSource
:members:
```

```{autoclass} tributo.data.provider.DatasetHandle
:members:
```

## Bounded ingestion API

New bounded-read code explicitly selects Ray Data or Daft and receives a typed
native handle plus a credential-free plan receipt. The Gateway never silently
switches engines.

```{autoclass} tributo.data.IngestionRequest
:members:
```

```{autoclass} tributo.data.IngestionGateway
:members:
```

```{autoclass} tributo.data.IngestionOpenResult
:members:
```

```{autoclass} tributo.data.IngestionPlanReceipt
:members:
```

```{autofunction} tributo.data.open_ingestion
```

## Training API

```{autoclass} tributo.training.BaseTrainer
:members:
```

```{autoclass} tributo.training.AlgorithmSpec
:members:
```

## Model bundle API

```{autoclass} tributo.exporting.ExportSpec
:members:
```

```{autoclass} tributo.exporting.ExportTarget
:members:
```

```{autofunction} tributo.exporting.export
```

```{autofunction} tributo.exporting.load_bundle
```

## Inference API

```{autoclass} tributo.inference.InferenceConfig
:members:
```

```{autofunction} tributo.inference.run_batch_inference
```

## Embedding and serving API

```{autoclass} tributo.embeddings.ModelSpec
:members:
```

```{autofunction} tributo.embeddings.submit_embedding_job
```

```{autofunction} tributo.serving.start_serving
```

## Streaming API

`StreamSource` is an unbounded input protocol. It does not return a finite Ray
Dataset.

```{autoclass} tributo.streaming.StreamSource
:members:
```

## Alpha API

The in-process pipeline is alpha. Retry, timeout, checkpoint, and cache
semantics are not part of its current contract.

```{autoclass} tributo.pipeline.Pipeline
:members:
```
