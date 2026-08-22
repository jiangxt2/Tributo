# Model lifecycle API

```{important}
This page is generated from top-level `@PublicAPI` annotations. Do not edit it
by hand. Run `python tools/generate_public_api_reference.py` after changing
a public annotation or moving a public object.
```

Stable, Beta, and Alpha objects appear because Ray-style API policy requires
documentation for every public stability tier.

## `tributo.exporting`

```{autoclass} tributo.exporting.ExportSpec
:no-members:
```

```{autofunction} tributo.exporting.export
```

```{autofunction} tributo.exporting.load_bundle
```


## `tributo.exporting.bundle_reader`

```{autoclass} tributo.exporting.bundle_reader.BundleReader
:no-members:
```


## `tributo.exporting.capabilities`

```{autoclass} tributo.exporting.capabilities.ArtifactCapability
:no-members:
```

```{autoclass} tributo.exporting.capabilities.CapabilityRegistry
:no-members:
```

```{autofunction} tributo.exporting.capabilities.get_default_capability_registry
```


## `tributo.exporting.conftest`

```{autoclass} tributo.exporting.conftest.ExportSourceProviderConformanceTest
:no-members:
```

```{autoclass} tributo.exporting.conftest.ExporterConformanceTest
:no-members:
```

```{autoclass} tributo.exporting.conftest.ValidatorConformanceTest
:no-members:
```


## `tributo.exporting.dispatch`

```{autoclass} tributo.exporting.dispatch.InlineHookDispatcher
:no-members:
```


## `tributo.exporting.events`

```{autoclass} tributo.exporting.events.OperationEvent
:no-members:
```


## `tributo.exporting.executor`

```{autoclass} tributo.exporting.executor.ExportManager
:no-members:
```


## `tributo.exporting.gc`

```{autoclass} tributo.exporting.gc.BundleGarbageCollector
:no-members:
```


## `tributo.exporting.hooks`

```{autoclass} tributo.exporting.hooks.HookOutcome
:no-members:
```

```{autoclass} tributo.exporting.hooks.PublicationHook
:no-members:
```

```{autoclass} tributo.exporting.hooks.PublicationRunner
:no-members:
```


## `tributo.exporting.manifest`

```{autoclass} tributo.exporting.manifest.ExportManifest
:no-members:
```

```{autoclass} tributo.exporting.manifest.ExportManifestV2
:no-members:
```

```{autoclass} tributo.exporting.manifest.ManifestExecution
:no-members:
```

```{autoclass} tributo.exporting.manifest.ManifestExecutionNode
:no-members:
```

```{autoclass} tributo.exporting.manifest.ManifestSchemaRegistry
:no-members:
```

```{autoclass} tributo.exporting.manifest.ManifestSignature
:no-members:
```

```{autoclass} tributo.exporting.manifest.ManifestSourceInfo
:no-members:
```

```{autoclass} tributo.exporting.manifest.SignatureField
:no-members:
```

```{autofunction} tributo.exporting.manifest.compute_bundle_digest
```


## `tributo.exporting.models`

```{autoclass} tributo.exporting.models.AliasConfig
:no-members:
```

```{autoclass} tributo.exporting.models.ArtifactDraft
:no-members:
```

```{autoclass} tributo.exporting.models.ArtifactFile
:no-members:
```

```{autoclass} tributo.exporting.models.ArtifactRef
:no-members:
```

```{autoclass} tributo.exporting.models.BundleOutputConfig
:no-members:
```

```{autoclass} tributo.exporting.models.BundleRef
:no-members:
```

```{autoclass} tributo.exporting.models.BundleResult
:no-members:
```

```{autoclass} tributo.exporting.models.CheckpointField
:no-members:
```

```{autoclass} tributo.exporting.models.DraftFile
:no-members:
```

```{autoclass} tributo.exporting.models.ExportCheckpointV1
:no-members:
```

```{autoclass} tributo.exporting.models.ExportContext
:no-members:
```

```{autoclass} tributo.exporting.models.ExportExecutionResult
:no-members:
```

```{autoclass} tributo.exporting.models.ExportSource
:no-members:
```

```{autoclass} tributo.exporting.models.ExportTarget
:no-members:
```

```{autoclass} tributo.exporting.models.FailureInfo
:no-members:
```

```{autoclass} tributo.exporting.models.HookBinding
:no-members:
```

```{autoclass} tributo.exporting.models.HookReceipt
:no-members:
```

```{autoclass} tributo.exporting.models.HookStatus
:no-members:
```

```{autoclass} tributo.exporting.models.LogicalArtifact
:no-members:
```

```{autoclass} tributo.exporting.models.NodeResult
:no-members:
```

```{autoclass} tributo.exporting.models.PlannedTarget
:no-members:
```

```{autoclass} tributo.exporting.models.PluginLoadDiagnostic
:no-members:
```

```{autoclass} tributo.exporting.models.ProducerInfo
:no-members:
```

```{autoclass} tributo.exporting.models.SupportRequest
:no-members:
```

```{autoclass} tributo.exporting.models.SupportResult
:no-members:
```

```{autoclass} tributo.exporting.models.UpstreamRequirement
:no-members:
```

```{autoclass} tributo.exporting.models.ValidationResult
:no-members:
```

```{autoclass} tributo.exporting.models.ValidatorBinding
:no-members:
```


## `tributo.exporting.planner`

```{autoclass} tributo.exporting.planner.ExportPlan
:no-members:
```

```{autoclass} tributo.exporting.planner.ExportPlanner
:no-members:
```


## `tributo.exporting.protocols`

```{autoclass} tributo.exporting.protocols.ExportSourceProvider
:no-members:
```

```{autoclass} tributo.exporting.protocols.ExportValidator
:no-members:
```

```{autoclass} tributo.exporting.protocols.ModelExporter
:no-members:
```

```{autoclass} tributo.exporting.protocols.ModelFactory
:no-members:
```


## `tributo.exporting.publisher`

```{autoclass} tributo.exporting.publisher.Publisher
:no-members:
```


## `tributo.exporting.records`

```{autoclass} tributo.exporting.records.DeliveryRecord
:no-members:
```

```{autoclass} tributo.exporting.records.ExecutionRecord
:no-members:
```

```{autoclass} tributo.exporting.records.PublicationAttempt
:no-members:
```


## `tributo.exporting.registries`

```{autoclass} tributo.exporting.registries.ExportRegistry
:no-members:
```

```{autoclass} tributo.exporting.registries.FlavorRegistry
:no-members:
```

```{autoclass} tributo.exporting.registries.ModelFactoryRegistry
:no-members:
```

```{autoclass} tributo.exporting.registries.SourceProviderRegistry
:no-members:
```

```{autoclass} tributo.exporting.registries.ValidatorRegistry
:no-members:
```


## `tributo.exporting.repository`

```{autoclass} tributo.exporting.repository.BundleAliasStore
:no-members:
```

```{autoclass} tributo.exporting.repository.BundleRepository
:no-members:
```

```{autoclass} tributo.exporting.repository.ReaderResourceLimits
:no-members:
```


## `tributo.exporting.runtime`

```{autoclass} tributo.exporting.runtime.BundleModel
:no-members:
```

```{autoclass} tributo.exporting.runtime.BundleModelFlavor
:no-members:
```

```{autoclass} tributo.exporting.runtime.BundleModelLoader
:no-members:
```

```{autoclass} tributo.exporting.runtime.BundleModelRuntime
:no-members:
```

```{autoclass} tributo.exporting.runtime.BundleReaderLike
:no-members:
```

```{autoclass} tributo.exporting.runtime.FlavorSupportEntry
:no-members:
```


## `tributo.exporting.service`

```{autoclass} tributo.exporting.service.BundleExportService
:no-members:
```

```{autofunction} tributo.exporting.service.bundle_id_for_request
```


## `tributo.exporting.validators`

```{autoclass} tributo.exporting.validators.StructureValidator
:no-members:
```

```{autoclass} tributo.exporting.validators.StructureValidatorOptions
:no-members:
```


## `tributo.integrations.exporters.hf_onnx`

```{autoclass} tributo.integrations.exporters.hf_onnx.HuggingFaceONNXExporter
:no-members:
```


## `tributo.integrations.exporters.onnx_quantizer`

```{autoclass} tributo.integrations.exporters.onnx_quantizer.ONNXQuantizer
:no-members:
```


## `tributo.integrations.exporters.options`

```{autoclass} tributo.integrations.exporters.options.HFONNXOptions
:no-members:
```

```{autoclass} tributo.integrations.exporters.options.ONNXQuantizerOptions
:no-members:
```

```{autoclass} tributo.integrations.exporters.options.SafetensorsOptions
:no-members:
```

```{autoclass} tributo.integrations.exporters.options.TorchONNXOptions
:no-members:
```

```{autoclass} tributo.integrations.exporters.options.XGBoostJSONOptions
:no-members:
```

```{autoclass} tributo.integrations.exporters.options.XGBoostNativeOptions
:no-members:
```

```{autoclass} tributo.integrations.exporters.options.XGBoostONNXOptions
:no-members:
```

```{autoclass} tributo.integrations.exporters.options.XGBoostUBJOptions
:no-members:
```


## `tributo.integrations.exporters.prebuilt_onnx`

```{autoclass} tributo.integrations.exporters.prebuilt_onnx.PrebuiltONNXExporter
:no-members:
```

```{autoclass} tributo.integrations.exporters.prebuilt_onnx.PrebuiltONNXOptions
:no-members:
```


## `tributo.integrations.exporters.torch_export`

```{autoclass} tributo.integrations.exporters.torch_export.TorchExportExporter
:no-members:
```

```{autoclass} tributo.integrations.exporters.torch_export.TorchExportOptions
:no-members:
```


## `tributo.integrations.exporters.torch_onnx`

```{autoclass} tributo.integrations.exporters.torch_onnx.TorchONNXExporter
:no-members:
```


## `tributo.integrations.exporters.torch_safetensors`

```{autoclass} tributo.integrations.exporters.torch_safetensors.TorchSafetensorsExporter
:no-members:
```


## `tributo.integrations.exporters.x_learner`

```{autoclass} tributo.integrations.exporters.x_learner.XLearnerCausalReportExporter
:no-members:
```

```{autoclass} tributo.integrations.exporters.x_learner.XLearnerExporter
:no-members:
```


## `tributo.integrations.exporters.xgboost_native`

```{autoclass} tributo.integrations.exporters.xgboost_native.XGBoostJSONExporter
:no-members:
```

```{autoclass} tributo.integrations.exporters.xgboost_native.XGBoostNativeExporter
:no-members:
```

```{autoclass} tributo.integrations.exporters.xgboost_native.XGBoostUBJExporter
:no-members:
```


## `tributo.integrations.exporters.xgboost_onnx`

```{autoclass} tributo.integrations.exporters.xgboost_onnx.XGBoostONNXExporter
:no-members:
```


## `tributo.integrations.hooks.mlflow_hook`

```{autoclass} tributo.integrations.hooks.mlflow_hook.MLflowHookOptions
:no-members:
```

```{autoclass} tributo.integrations.hooks.mlflow_hook.MLflowPostPublishHook
:no-members:
```


## `tributo.integrations.storage.gc`

```{autoclass} tributo.integrations.storage.gc.S3BundleGarbageCollector
:no-members:
```


## `tributo.integrations.storage.json_operation_store`

```{autoclass} tributo.integrations.storage.json_operation_store.JsonFileOperationStore
:no-members:
```


## `tributo.integrations.validators.onnx_runtime`

```{autoclass} tributo.integrations.validators.onnx_runtime.ONNXRuntimeValidator
:no-members:
```


## `tributo.registry.callback`

```{autoclass} tributo.registry.callback.MLflowTrackingCallback
:no-members:
```


## `tributo.registry.model_registry`

```{autoclass} tributo.registry.model_registry.ModelRegistry
:no-members:
```


## `tributo.registry.schema`

```{autoclass} tributo.registry.schema.ExperimentInfo
:no-members:
```

```{autoclass} tributo.registry.schema.ModelVersion
:no-members:
```

```{autoclass} tributo.registry.schema.RunMetrics
:no-members:
```
