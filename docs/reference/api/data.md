# Data API

```{important}
This page is generated from top-level `@PublicAPI` annotations. Do not edit it
by hand. Run `python tools/generate_public_api_reference.py` after changing
a public annotation or moving a public object.
```

Stable, Beta, and Alpha objects appear because Ray-style API policy requires
documentation for every public stability tier.

## `tributo.data.base`

```{autoclass} tributo.data.base.DataConnector
:no-members:
```

```{autoclass} tributo.data.base.S3Config
:no-members:
```


## `tributo.data.contracts.handles`

```{autoclass} tributo.data.contracts.handles.DaftDataFrameHandle
:no-members:
```

```{autoclass} tributo.data.contracts.handles.RayDataHandle
:no-members:
```


## `tributo.data.contracts.modes`

```{autoclass} tributo.data.contracts.modes.WriteMode
:no-members:
```


## `tributo.data.csv`

```{autoclass} tributo.data.csv.CsvDataConnector
:no-members:
```


## `tributo.data.graph`

```{autoclass} tributo.data.graph.GraphDataBundle
:no-members:
```

```{autoclass} tributo.data.graph.GraphSchema
:no-members:
```


## `tributo.data.handle_adapters`

```{autoclass} tributo.data.handle_adapters.HandleConversionReceipt
:no-members:
```

```{autoclass} tributo.data.handle_adapters.RayHandleAdaptation
:no-members:
```

```{autofunction} tributo.data.handle_adapters.adapt_daft_result_to_ray
```


## `tributo.data.iceberg`

```{autoclass} tributo.data.iceberg.IcebergDataConnector
:no-members:
```


## `tributo.data.ingestion`

```{autoclass} tributo.data.ingestion.DistributionVersionEvidence
:no-members:
```

```{autoclass} tributo.data.ingestion.HandleOwnership
:no-members:
```

```{autoclass} tributo.data.ingestion.IngestionDescriptor
:no-members:
```

```{autoclass} tributo.data.ingestion.IngestionGateway
:no-members:
```

```{autoclass} tributo.data.ingestion.IngestionOpenResult
:no-members:
```

```{autoclass} tributo.data.ingestion.IngestionPlanReceipt
:no-members:
```

```{autoclass} tributo.data.ingestion.IngestionRequest
:no-members:
```

```{autoclass} tributo.data.ingestion.IngestionRuntimeContext
:no-members:
```

```{autoclass} tributo.data.ingestion.PhysicalSplitSummary
:no-members:
```

```{autoclass} tributo.data.ingestion.ReadHint
:no-members:
```

```{autoclass} tributo.data.ingestion.ReadOptions
:no-members:
```

```{autoclass} tributo.data.ingestion.SchemaContract
:no-members:
```

```{autoclass} tributo.data.ingestion.TransformDecision
:no-members:
```

```{autofunction} tributo.data.ingestion.describe_ingestion
```

```{autofunction} tributo.data.ingestion.open_ingestion
```

```{autofunction} tributo.data.ingestion.ray_worker_distribution_probe
```


## `tributo.data.lance`

```{autoclass} tributo.data.lance.LanceDataConnector
:no-members:
```


## `tributo.data.parquet`

```{autoclass} tributo.data.parquet.ParquetDataConnector
:no-members:
```


## `tributo.data.provider`

```{autoclass} tributo.data.provider.DataSourceProvider
:no-members:
```

```{autoclass} tributo.data.provider.DatasetHandle
:no-members:
```

```{autoclass} tributo.data.provider.ResolvedSource
:no-members:
```


## `tributo.data.provider_registry`

```{autofunction} tributo.data.provider_registry.list_providers
```

```{autofunction} tributo.data.provider_registry.register_provider
```

```{autofunction} tributo.data.provider_registry.resolve_provider
```

```{autofunction} tributo.data.provider_registry.unregister_provider
```


## `tributo.data.refs`

```{autoclass} tributo.data.refs.DatasetRef
:no-members:
```

```{autofunction} tributo.data.refs.compute_ref_id
```

```{autofunction} tributo.data.refs.digest
```

```{autofunction} tributo.data.refs.schema_fingerprint
```


## `tributo.data.registry`

```{autofunction} tributo.data.registry.get_connector
```

```{autofunction} tributo.data.registry.list_connectors
```

```{autofunction} tributo.data.registry.register_connector
```


## `tributo.data.source_config`

```{autoclass} tributo.data.source_config.CsvSourceConfig
:no-members:
```

```{autoclass} tributo.data.source_config.IcebergSourceConfig
:no-members:
```

```{autoclass} tributo.data.source_config.LegacyConfigNormalizer
:no-members:
```

```{autoclass} tributo.data.source_config.LegacySourceInput
:no-members:
```

```{autoclass} tributo.data.source_config.ParquetSourceConfig
:no-members:
```

```{autoclass} tributo.data.source_config.ProviderSourceConfig
:no-members:
```

```{autoclass} tributo.data.source_config.RawSourceConfig
:no-members:
```

```{autoclass} tributo.data.source_config.SqlPartitioning
:no-members:
```

```{autoclass} tributo.data.source_config.SqlSourceConfig
:no-members:
```

```{autofunction} tributo.data.source_config.apply_source_projection
```

```{autofunction} tributo.data.source_config.source_projection
```


## `tributo.data.transform_ir`

```{autoclass} tributo.data.transform_ir.CastColumn
:no-members:
```

```{autoclass} tributo.data.transform_ir.ColumnRename
:no-members:
```

```{autoclass} tributo.data.transform_ir.DropColumns
:no-members:
```

```{autoclass} tributo.data.transform_ir.FillNull
:no-members:
```

```{autoclass} tributo.data.transform_ir.FilterComparison
:no-members:
```

```{autoclass} tributo.data.transform_ir.FilterEq
:no-members:
```

```{autoclass} tributo.data.transform_ir.FilterIsIn
:no-members:
```

```{autoclass} tributo.data.transform_ir.FilterNotEq
:no-members:
```

```{autoclass} tributo.data.transform_ir.FilterNotNull
:no-members:
```

```{autoclass} tributo.data.transform_ir.FilterNull
:no-members:
```

```{autoclass} tributo.data.transform_ir.FilterRange
:no-members:
```

```{autoclass} tributo.data.transform_ir.Limit
:no-members:
```

```{autoclass} tributo.data.transform_ir.RenameColumns
:no-members:
```

```{autoclass} tributo.data.transform_ir.SelectColumns
:no-members:
```

```{autoclass} tributo.data.transform_ir.TransformPipeline
:no-members:
```

```{autofunction} tributo.data.transform_ir.transform_ir_digest
```


## `tributo.data.writing.builtins`

```{autofunction} tributo.data.writing.builtins.default_write_gateway
```


## `tributo.data.writing.capabilities`

```{autoclass} tributo.data.writing.capabilities.WriteCapability
:no-members:
```


## `tributo.data.writing.contracts`

```{autoexception} tributo.data.writing.contracts.WriteBindingError
```

```{autoexception} tributo.data.writing.contracts.WriteCapabilityError
```

```{autoclass} tributo.data.writing.contracts.WriteDescriptor
:no-members:
```

```{autoexception} tributo.data.writing.contracts.WriteError
```

```{autoclass} tributo.data.writing.contracts.WriteReceipt
:no-members:
```

```{autoclass} tributo.data.writing.contracts.WriteRequest
:no-members:
```


## `tributo.data.writing.gateway`

```{autoclass} tributo.data.writing.gateway.WriteGateway
:no-members:
```


## `tributo.integrations.sources.huggingface`

```{autoclass} tributo.integrations.sources.huggingface.HuggingFaceSourceProvider
:no-members:
```


## `tributo.integrations.sources.ray_dnn`

```{autoclass} tributo.integrations.sources.ray_dnn.RayDnnSourceProvider
:no-members:
```


## `tributo.integrations.sources.ray_pu`

```{autoclass} tributo.integrations.sources.ray_pu.RayPUSourceProvider
:no-members:
```


## `tributo.integrations.sources.ray_xgboost`

```{autoclass} tributo.integrations.sources.ray_xgboost.RayXGBoostSourceProvider
:no-members:
```


## `tributo.streaming.kafka_source`

```{autoclass} tributo.streaming.kafka_source.KafkaStreamSource
:no-members:
```


## `tributo.streaming.protocol`

```{autoclass} tributo.streaming.protocol.StreamSource
:no-members:
```
