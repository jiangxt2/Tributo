# Inference and serving API

```{important}
This page is generated from top-level `@PublicAPI` annotations. Do not edit it
by hand. Run `python tools/generate_public_api_reference.py` after changing
a public annotation or moving a public object.
```

Stable, Beta, and Alpha objects appear because Ray-style API policy requires
documentation for every public stability tier.

## `tributo.explainability.conformance`

```{autofunction} tributo.explainability.conformance.validate_adapter_conformance
```


## `tributo.explainability.contracts`

```{autoclass} tributo.explainability.contracts.ExplainabilityConfig
:no-members:
```

```{autoclass} tributo.explainability.contracts.ExplainabilityDescriptor
:no-members:
```

```{autoclass} tributo.explainability.contracts.ExplainabilityLimits
:no-members:
```

```{autoclass} tributo.explainability.contracts.ExplainabilityOperationRecord
:no-members:
```

```{autoclass} tributo.explainability.contracts.ExplainabilityReceipt
:no-members:
```

```{autoclass} tributo.explainability.contracts.ExplainabilityRequest
:no-members:
```

```{autoclass} tributo.explainability.contracts.FeatureAttribution
:no-members:
```

```{autoclass} tributo.explainability.contracts.ReferenceBinding
:no-members:
```

```{autoclass} tributo.explainability.contracts.ResourcePolicy
:no-members:
```

```{autoclass} tributo.explainability.contracts.ResultPolicy
:no-members:
```


## `tributo.explainability.executor`

```{autoclass} tributo.explainability.executor.ExplainabilityBatchWorker
:no-members:
```

```{autofunction} tributo.explainability.executor.run_batch_explainability
```


## `tributo.explainability.export`

```{autofunction} tributo.explainability.export.prepare_bundle_output_config
```


## `tributo.explainability.job_runner`

```{autofunction} tributo.explainability.job_runner.submit_explainability_job
```


## `tributo.explainability.planner`

```{autoclass} tributo.explainability.planner.ExplainabilityPlan
:no-members:
```

```{autoclass} tributo.explainability.planner.ExplainabilityPlanner
:no-members:
```


## `tributo.explainability.protocols`

```{autoclass} tributo.explainability.protocols.ExplainableModelContext
:no-members:
```

```{autoclass} tributo.explainability.protocols.ExplainerAdapter
:no-members:
```

```{autoclass} tributo.explainability.protocols.PreparedExplainer
:no-members:
```

```{autoclass} tributo.explainability.protocols.SupportDecision
:no-members:
```


## `tributo.explainability.reference`

```{autoclass} tributo.explainability.reference.FileReferenceProvider
:no-members:
```

```{autoclass} tributo.explainability.reference.ReferenceProvider
:no-members:
```

```{autoclass} tributo.explainability.reference.ResolvedReference
:no-members:
```


## `tributo.explainability.registry`

```{autoclass} tributo.explainability.registry.ExplainerRegistry
:no-members:
```

```{autofunction} tributo.explainability.registry.get_default_explainer_registry
```


## `tributo.explainability.shap`

```{autoclass} tributo.explainability.shap.ShapAdapter
:no-members:
```


## `tributo.inference.api`

```{autofunction} tributo.inference.api.resolve_inference
```

```{autofunction} tributo.inference.api.run_inference
```

```{autofunction} tributo.inference.api.run_resolved_inference
```


## `tributo.inference.base`

```{autoclass} tributo.inference.base.BasePredictor
:no-members:
```


## `tributo.inference.batch_predictor`

```{autoclass} tributo.inference.batch_predictor.XGBoostONNXPredictor
:no-members:
```


## `tributo.inference.bundle_predictor`

```{autoclass} tributo.inference.bundle_predictor.BundleBatchPredictor
:no-members:
```


## `tributo.inference.contracts`

```{autoclass} tributo.inference.contracts.ArtifactModelReference
:no-members:
```

```{autoclass} tributo.inference.contracts.BundleModelReference
:no-members:
```

```{autoclass} tributo.inference.contracts.FailureDiagnostic
:no-members:
```

```{autoclass} tributo.inference.contracts.InferenceExecutor
:no-members:
```

```{autoclass} tributo.inference.contracts.InferenceRequest
:no-members:
```

```{autoclass} tributo.inference.contracts.InferenceResult
:no-members:
```

```{autoclass} tributo.inference.contracts.InputBindingSpec
:no-members:
```

```{autoclass} tributo.inference.contracts.LanceResultSinkRequest
:no-members:
```

```{autoclass} tributo.inference.contracts.LanceVectorColumnSpec
:no-members:
```

```{autoclass} tributo.inference.contracts.OutputBindingSpec
:no-members:
```

```{autoclass} tributo.inference.contracts.ParquetResultSinkRequest
:no-members:
```

```{autoclass} tributo.inference.contracts.RayExecutionPolicy
:no-members:
```

```{autoclass} tributo.inference.contracts.RegistryModelReference
:no-members:
```

```{autoclass} tributo.inference.contracts.ResolvedInference
:no-members:
```

```{autoclass} tributo.inference.contracts.ResolvedInputSelection
:no-members:
```

```{autoclass} tributo.inference.contracts.ResolvedModelSelection
:no-members:
```

```{autoclass} tributo.inference.contracts.ResultSink
:no-members:
```

```{autoclass} tributo.inference.contracts.ResultSinkReceipt
:no-members:
```

```{autoclass} tributo.inference.contracts.TensorInputBinding
:no-members:
```

```{autoclass} tributo.inference.contracts.TensorOutputBinding
:no-members:
```


## `tributo.inference.executor`

```{autoclass} tributo.inference.executor.RayMapBatchesExecutor
:no-members:
```


## `tributo.inference.input_resolver`

```{autoclass} tributo.inference.input_resolver.IngestionGatewayInputResolver
:no-members:
```

```{autoclass} tributo.inference.input_resolver.InputResolverPort
:no-members:
```

```{autoclass} tributo.inference.input_resolver.OpenedInferenceInput
:no-members:
```


## `tributo.inference.job_runner`

```{autoclass} tributo.inference.job_runner.InferenceJobAttempt
:no-members:
```

```{autoclass} tributo.inference.job_runner.InferenceJobResult
:no-members:
```

```{autofunction} tributo.inference.job_runner.map_ray_job_status
```

```{autofunction} tributo.inference.job_runner.submit_inference_job
```

```{autofunction} tributo.inference.job_runner.submit_inference_request
```

```{autofunction} tributo.inference.job_runner.submit_inference_request_with_identity
```

```{autofunction} tributo.inference.job_runner.submit_inference_request_with_retry
```

```{autofunction} tributo.inference.job_runner.submit_resolved_inference
```

```{autofunction} tributo.inference.job_runner.submit_resolved_inference_with_identity
```

```{autofunction} tributo.inference.job_runner.wait_for_job
```


## `tributo.inference.pipeline`

```{autoclass} tributo.inference.pipeline.InferenceConfig
:no-members:
```

```{autofunction} tributo.inference.pipeline.run_batch_inference
```

```{autofunction} tributo.inference.pipeline.run_inference_from_json
```


## `tributo.inference.post_training`

```{autoclass} tributo.inference.post_training.PostTrainingInferenceAction
:no-members:
```

```{autofunction} tributo.inference.post_training.run_post_training_inference
```

```{autofunction} tributo.inference.post_training.submit_post_training_inference
```


## `tributo.inference.resolver`

```{autoclass} tributo.inference.resolver.InferenceResolver
:no-members:
```


## `tributo.integrations.flavors.onnx_runtime`

```{autoclass} tributo.integrations.flavors.onnx_runtime.ONNXRuntimeFlavor
:no-members:
```


## `tributo.integrations.flavors.xgboost_native`

```{autoclass} tributo.integrations.flavors.xgboost_native.XGBoostNativeFlavor
:no-members:
```


## `tributo.integrations.model_importers.artifact`

```{autoclass} tributo.integrations.model_importers.artifact.ArtifactImportOptions
:no-members:
```

```{autoclass} tributo.integrations.model_importers.artifact.ArtifactModelImporter
:no-members:
```


## `tributo.integrations.model_importers.mlflow`

```{autoclass} tributo.integrations.model_importers.mlflow.MLflowImportOptions
:no-members:
```

```{autoclass} tributo.integrations.model_importers.mlflow.MLflowModelImporter
:no-members:
```


## `tributo.integrations.model_importers.registry`

```{autoclass} tributo.integrations.model_importers.registry.ModelImporter
:no-members:
```

```{autoclass} tributo.integrations.model_importers.registry.ModelImporterRegistry
:no-members:
```

```{autofunction} tributo.integrations.model_importers.registry.build_default_model_importer_registry
```


## `tributo.integrations.sinks.lance`

```{autoclass} tributo.integrations.sinks.lance.LanceResultSink
:no-members:
```


## `tributo.integrations.sinks.parquet`

```{autoclass} tributo.integrations.sinks.parquet.ParquetResultSink
:no-members:
```


## `tributo.serving.composition`

```{autoclass} tributo.serving.composition.ModelRunner
:no-members:
```


## `tributo.serving.grpc_deployment`

```{autoclass} tributo.serving.grpc_deployment.gRPCInferenceService
:no-members:
```


## `tributo.serving.grpc_runner`

```{autofunction} tributo.serving.grpc_runner.get_grpc_serving_status
```

```{autofunction} tributo.serving.grpc_runner.start_grpc_serving
```

```{autofunction} tributo.serving.grpc_runner.stop_grpc_serving
```


## `tributo.serving.model_deployment`

```{autoclass} tributo.serving.model_deployment.ONNXModel
:no-members:
```


## `tributo.serving.schema`

```{autoclass} tributo.serving.schema.PredictInput
:no-members:
```

```{autoclass} tributo.serving.schema.PredictRequest
:no-members:
```

```{autoclass} tributo.serving.schema.PredictResponse
:no-members:
```


## `tributo.serving.serve_runner`

```{autofunction} tributo.serving.serve_runner.get_serving_status
```

```{autofunction} tributo.serving.serve_runner.start_serving
```

```{autofunction} tributo.serving.serve_runner.stop_serving
```


## `tributo.serving.streaming_deployment`

```{autoclass} tributo.serving.streaming_deployment.LLMStreamingService
:no-members:
```

```{autoclass} tributo.serving.streaming_deployment.StreamingInferenceService
:no-members:
```


## `tributo.serving.streaming_runner`

```{autofunction} tributo.serving.streaming_runner.get_streaming_serving_status
```

```{autofunction} tributo.serving.streaming_runner.start_streaming_serving
```

```{autofunction} tributo.serving.streaming_runner.stop_streaming_serving
```
