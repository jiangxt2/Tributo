"""Distributed batch inference pipeline: S3 Parquet → ONNX inference → S3 Parquet.

End-to-end Ray Data streaming; the Driver only orchestrates, data does not pass through the Head node.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from tributo.data.base import S3Config
    from tributo.inference.base import BasePredictor

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tributo.data import source_projection
from tributo.data.source_config import (
    CanonicalSourceInput,
    CsvSourceConfig,
    IcebergSourceConfig,
    LegacyConfigNormalizer,
    ParquetSourceConfig,
    ProviderSourceConfig,
    RawSourceConfig,
    apply_source_projection,
)
from tributo.exceptions import JobConfigurationError
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


def _build_s3_config(raw: dict[str, str]) -> "S3Config | None":
    """Build S3Config from a raw dict, returning None if empty.

    Centralises the duplicated S3Config construction previously spread
    across data-load and data-write paths.
    """
    from tributo.data.base import S3Config

    if not raw:
        return None
    return S3Config(
        access_key_id=raw.get("access_key_id"),
        secret_access_key=raw.get("secret_access_key"),
        endpoint=raw.get("endpoint"),
        region=raw.get("region"),
    )


@PublicAPI(stability="beta")
class InferenceConfig(BaseModel):
    """Batch inference task configuration.

    Direct construction raises ``pydantic.ValidationError`` on field validation failure;
    when accessed via the ``run_inference_from_json`` entry point, it is converted to
    ``JobConfigurationError``.

    Attributes:
        source: Canonical source configuration. Preferred for new callers.
        input_uri: Legacy URI of input data (S3:// or local path).
        s3_config: S3 auth configuration.
        clickhouse_config: ClickHouse connection config (required when type=clickhouse).
        output_uri: Output URI for inference results.
        model_uri: Model path (local or s3://) — legacy compat entry.
        bundle_uri: Published bundle URI — stable model entry point.
        model_role: Artifact role to serve from the bundle (default inference).
        unsafe_model: Permit bundles without typed signatures or unsafe flavors.
        storage_profile: Storage profile name for S3 bundles.
        feature_columns: List of feature column names needed for inference (column pruning).
        predictor_config: Passthrough dict for predictor-specific config.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: CanonicalSourceInput | None = None
    input_uri: str | None = Field(default=None, min_length=1)
    output_uri: str = Field(min_length=1)
    model_uri: str | None = Field(default=None, min_length=1)
    bundle_uri: str | None = Field(default=None, min_length=1)
    model_role: str = "inference"
    unsafe_model: bool = False
    storage_profile: str | None = None

    @model_validator(mode="after")
    def _check_model_entry(self) -> "InferenceConfig":
        if (self.model_uri is None) == (self.bundle_uri is None):
            raise ValueError(
                "exactly one of 'model_uri' (legacy) or 'bundle_uri' must be provided"
            )
        return self

    @model_validator(mode="after")
    def _check_source_entry(self) -> "InferenceConfig":
        if self.source is None and self.input_uri is None:
            raise ValueError("exactly one of 'source' or 'input_uri' is required")
        if self.source is not None and self.input_uri is not None:
            raise ValueError("source and input_uri cannot be used together")
        if self.source is not None:
            legacy_fields = {
                "data_type",
                "feature_columns",
                "s3_config",
                "ch_host",
                "ch_port",
                "ch_database",
                "ch_user",
                "ch_password",
                "ch_sql",
            }
            conflicts = sorted(legacy_fields & self.model_fields_set)
            if "feature_columns" in conflicts and not self.feature_columns:
                conflicts.remove("feature_columns")
            if conflicts:
                raise ValueError(
                    "canonical source cannot be combined with legacy fields: "
                    + ", ".join(conflicts)
                )
        return self

    # Data source type: s3 | csv | clickhouse
    data_type: str = "csv"

    feature_columns: list[str] = Field(default_factory=list)
    predictor_config: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = 4096
    concurrency: int = 4
    num_cpus_per_actor: float = 1.0
    num_gpus_per_actor: float = 0.0
    output_compression: str = "zstd"
    min_rows_per_file: int | None = None
    s3_config: dict[str, str] = Field(default_factory=dict)

    # ClickHouse-specific fields
    ch_host: str = ""
    ch_port: int = 8123
    ch_database: str = ""
    ch_user: str = ""
    ch_password: str = ""
    ch_sql: str = ""

    @property
    def return_probs(self) -> bool:
        """Backward compatibility: read from predictor_config."""
        return self.predictor_config.get("return_probs", True)

    @property
    def prediction_column(self) -> str:
        """Backward compatibility: read from predictor_config."""
        return self.predictor_config.get("prediction_column", "prediction")


def _credential_free_uri(uri: str) -> str:
    """Remove userinfo and query parameters from a display URI."""
    parsed = urlsplit(uri)
    if parsed.hostname is None:
        return uri
    netloc = parsed.hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _source_display_uri(source: CanonicalSourceInput) -> str:
    """Return a credential-free source identifier for logs and results."""
    if isinstance(source, ProviderSourceConfig):
        return _credential_free_uri(source.uri)
    if isinstance(source, (ParquetSourceConfig, CsvSourceConfig)):
        return _credential_free_uri(source.path)
    if isinstance(source, IcebergSourceConfig):
        return f"iceberg://{source.catalog}/{source.table}"
    host = source.host or "<env>"
    port = f":{source.port}" if source.port is not None else ""
    database = source.database or ""
    return f"{source.dialect}://{host}{port}/{database}"


def _source_s3_config(source: CanonicalSourceInput) -> dict[str, str]:
    """Return non-empty S3 settings carried by a canonical source."""
    if isinstance(source, (ParquetSourceConfig, CsvSourceConfig)):
        value = source.s3
        return value.model_dump(exclude_none=True) if value is not None else {}
    if isinstance(source, IcebergSourceConfig):
        return dict(source.s3 or {})
    if isinstance(source, ProviderSourceConfig):
        value = source.options.get("s3")
        if isinstance(value, dict):
            return {key: item for key, item in value.items() if item is not None}
    return {}


def _legacy_source(config: InferenceConfig) -> CanonicalSourceInput:
    """Normalize the historical flat inference fields into a source config."""
    if config.source is not None:
        return config.source
    assert config.input_uri is not None

    if config.data_type == "clickhouse":
        raw: dict[str, Any] = {
            "type": "clickhouse",
            "ch_host": config.ch_host or None,
            "ch_port": config.ch_port,
            "ch_database": config.ch_database or None,
            "ch_user": config.ch_user or None,
            "ch_password": config.ch_password or None,
            "ch_sql": config.ch_sql,
        }
    elif config.data_type == "s3":
        raw = {
            "type": "s3",
            "uri": config.input_uri,
            "format": "parquet",
            "s3": config.s3_config or None,
        }
    else:
        # The historical inference path read both local and S3 input_uri
        # values as Parquet unless the ClickHouse branch was selected.
        raw = {
            "type": "parquet",
            "path": config.input_uri,
            "s3": config.s3_config or None,
        }
    normalized = LegacyConfigNormalizer.normalize(raw)
    if isinstance(normalized, RawSourceConfig):
        raise JobConfigurationError(
            f"Unknown legacy inference source type: {normalized.type!r}"
        )
    result: CanonicalSourceInput = normalized
    if config.feature_columns:
        result = apply_source_projection(result, config.feature_columns)
    return result


def _legacy_json_source(data_config: dict[str, Any]) -> CanonicalSourceInput:
    """Convert the historical JSON ``data`` object to a canonical source."""
    data_type = data_config.get("type")
    input_uri = data_config.get("uri") or data_config.get("input")
    feature_columns = data_config.get("feature_columns") or []
    s3 = data_config.get("s3") or None

    if data_type is None:
        raw: dict[str, Any] = {
            "type": "parquet",
            "path": input_uri or "",
            "s3": s3,
        }
    elif data_type == "clickhouse":
        clickhouse = data_config.get("clickhouse") or {}
        if not isinstance(clickhouse, dict):
            raise ValueError("data.clickhouse must be a mapping")
        raw = {
            "type": "clickhouse",
            "ch_host": data_config.get("ch_host", clickhouse.get("host")),
            "ch_port": data_config.get("ch_port", clickhouse.get("port")),
            "ch_database": data_config.get("ch_database", clickhouse.get("database")),
            "ch_user": data_config.get("ch_user", clickhouse.get("user")),
            "ch_password": data_config.get("ch_password", clickhouse.get("password")),
            "ch_sql": data_config.get("ch_sql", clickhouse.get("sql", "")),
            "ch_sql_params": data_config.get("ch_sql_params", clickhouse.get("params")),
        }
    elif data_type == "s3":
        raw = {
            "type": "s3",
            "uri": input_uri or "",
            "format": data_config.get("format", "parquet"),
            "s3": s3,
        }
    elif data_type in {
        "parquet",
        "csv",
        "iceberg",
        "doris",
        "mysql",
        "postgresql",
    }:
        if data_type != "parquet":
            logger.warning(
                "Legacy inference data.type=%r preserves historical Parquet "
                "semantics; use canonical source for an explicit %s source",
                data_type,
                data_type,
            )
        raw = {
            "type": "parquet",
            "path": input_uri or "",
            "s3": s3,
        }
    else:
        raw = dict(data_config)
        raw["path"] = raw.get("path") or input_uri or ""
        raw.pop("uri", None)
        raw.pop("input", None)
        raw.pop("feature_columns", None)
        raw.pop("clickhouse", None)

    normalized = LegacyConfigNormalizer.normalize(raw)
    if isinstance(normalized, RawSourceConfig):
        raise JobConfigurationError(
            f"Unknown legacy inference source type: {normalized.type!r}"
        )
    result: CanonicalSourceInput = normalized
    if feature_columns:
        result = apply_source_projection(result, feature_columns)
    return result


@PublicAPI(stability="beta")
def run_batch_inference(
    config: InferenceConfig,
    predictor_cls: type[BasePredictor] | None = None,
) -> dict[str, Any]:
    """Execute the distributed batch inference pipeline.

    Data flow:
    1. DataSourceProvider opens the canonical source with native projection;
    2. map_batches + ActorPoolStrategy for distributed inference;
    3. write_parquet streams results back to S3.

    The Driver never holds actual data, it only orchestrates tasks.

    Args:
        config: Inference task configuration.
        predictor_cls: Predictor class, defaults to XGBoostONNXPredictor.

    Returns:
        {"input_path": ..., "output_path": ..., "status": "completed"}
    """
    import ray.data

    from tributo.inference.batch_predictor import XGBoostONNXPredictor
    from tributo.training.data_loader import load_ray_dataset_from_source

    if predictor_cls is None:
        predictor_cls = XGBoostONNXPredictor

    source = _legacy_source(config)
    source_columns = source_projection(source)

    # Build predictor_config: merge predictor-specific config + S3 config +
    # feature_columns.  Source credentials are also used for the output sink
    # when the legacy output configuration is absent.
    predictor_config = {**config.predictor_config}
    s3_config = config.s3_config or _source_s3_config(source)
    if s3_config:
        predictor_config["s3_config"] = s3_config
    predictor_config["feature_names"] = config.feature_columns or source_columns

    # If feature_columns is not set, auto-read from the model entry
    # (bundle signature or legacy ONNX metadata).
    feature_columns = config.feature_columns or source_columns or []
    if not feature_columns:
        if config.bundle_uri is not None:
            feature_columns = predictor_cls.get_feature_names(
                None,
                predictor_config,
                bundle_uri=config.bundle_uri,
                role=config.model_role,
                unsafe=config.unsafe_model,
                storage_profile=config.storage_profile,
            )
        else:
            assert config.model_uri is not None  # mutually exclusive with bundle_uri
            feature_columns = predictor_cls.get_feature_names(
                config.model_uri, predictor_config
            )
        if not feature_columns:
            raise JobConfigurationError(
                "feature_columns is empty and model has no feature_names metadata. "
                "Either specify feature_columns in config JSON or publish ONNX "
                "feature_names metadata."
            )
        predictor_config["feature_names"] = feature_columns
        logger.info(
            "Auto-loaded feature_columns from model metadata: %s", feature_columns
        )

    if feature_columns:
        source = apply_source_projection(source, feature_columns)
    ds = load_ray_dataset_from_source(source.model_dump(mode="python"))

    # ── 2. Distributed inference (ActorPoolStrategy keeps models resident) ──
    model_ref = config.bundle_uri if config.bundle_uri is not None else config.model_uri
    logger.info(
        "Starting inference: predictor=%s model=%s concurrency=%d batch_size=%d "
        "cpus_per_actor=%.1f gpus_per_actor=%.1f",
        predictor_cls.__name__,
        model_ref,
        config.concurrency,
        config.batch_size,
        config.num_cpus_per_actor,
        config.num_gpus_per_actor,
    )

    if config.bundle_uri is not None:
        constructor_args: tuple[Any, ...] = (
            None,
            predictor_config,
            config.bundle_uri,
            config.model_role,
            config.unsafe_model,
            config.storage_profile,
        )
    else:
        constructor_args = (config.model_uri, predictor_config)

    ds = ds.map_batches(
        predictor_cls,
        fn_constructor_args=constructor_args,
        batch_size=config.batch_size,
        compute=ray.data.ActorPoolStrategy(
            min_size=config.concurrency,
            max_size=config.concurrency,
        ),
        num_cpus=config.num_cpus_per_actor,
        num_gpus=config.num_gpus_per_actor,
    )

    # ── 3. Streaming write-back ──
    logger.info("Writing predictions to %s", config.output_uri)
    write_kwargs: dict[str, Any] = {"compression": config.output_compression}
    if config.min_rows_per_file:
        write_kwargs["min_rows_per_file"] = config.min_rows_per_file

    if config.output_uri.startswith("s3://"):
        import pyarrow.fs as pafs

        from tributo.data._s3 import to_pyarrow_s3_kwargs

        s3 = _build_s3_config(s3_config)
        write_kwargs["filesystem"] = pafs.S3FileSystem(**to_pyarrow_s3_kwargs(s3))
        output_path = config.output_uri.removeprefix("s3://")
    else:
        output_path = config.output_uri

    ds.write_parquet(output_path, **write_kwargs)

    logger.info(
        "Batch inference complete: %s → %s",
        _source_display_uri(source),
        config.output_uri,
    )
    return {
        "input_path": _source_display_uri(source),
        "output_path": config.output_uri,
        "status": "completed",
    }


@PublicAPI(stability="beta")
def run_inference_from_json(config_path: str) -> dict[str, Any]:
    """Run batch inference from a JSON config file.

    JSON structure::

        data:
          type: s3 | csv
          uri: s3://bucket/input/*.parquet     # or local path
          format: parquet                       # explicit csv|parquet
          feature_columns: [feat_0, feat_1, ...]
          s3:                                   # required when type=s3
            region: cn-north-1
            access_key_id: ...
            secret_access_key: ...
            endpoint: http://minio:9000

        model:
          uri: s3://bucket/models/model.onnx   # or local path (legacy)
          bundle_uri: /tmp/models/bundle       # published bundle (stable entry)
          role: inference                      # artifact role (default inference)
          return_probs: true                    # passed through to predictor_config

        output:
          uri: s3://bucket/output/
          prediction_column: prediction         # passed through to predictor_config
          compression: zstd

        ray:
          concurrency: 4                        # Actor pool size
          batch_size: 4096
          num_cpus_per_actor: 1.0
          num_gpus_per_actor: 0.0
    """
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise JobConfigurationError("config root must be a mapping")

    data_cfg = raw.get("data", {})
    model_cfg = raw.get("model", {})
    output_cfg = raw.get("output", {})
    ray_cfg = raw.get("ray", {})
    if not isinstance(data_cfg, dict):
        raise JobConfigurationError("config.data must be a mapping")
    if not isinstance(model_cfg, dict):
        raise JobConfigurationError("config.model must be a mapping")
    if not isinstance(output_cfg, dict):
        raise JobConfigurationError("config.output must be a mapping")
    if not isinstance(ray_cfg, dict):
        raise JobConfigurationError("config.ray must be a mapping")

    model_uri = model_cfg.get("uri") or model_cfg.get("path", "")
    bundle_uri = model_cfg.get("bundle_uri")
    output_uri = output_cfg.get("uri") or output_cfg.get("path", "")

    if data_cfg.get("input") and not data_cfg.get("uri"):
        logger.warning("data.input is deprecated, use data.uri instead")
    if model_cfg.get("path") and not model_cfg.get("uri"):
        logger.warning("model.path is deprecated, use model.uri instead")
    if output_cfg.get("path") and not output_cfg.get("uri"):
        logger.warning("output.path is deprecated, use output.uri instead")

    # Build predictor_config: extract from JSON and pass through
    predictor_config: dict[str, Any] = {}
    if "return_probs" in model_cfg:
        predictor_config["return_probs"] = model_cfg["return_probs"]
    if "prediction_column" in output_cfg:
        predictor_config["prediction_column"] = output_cfg["prediction_column"]

    top_level_source = raw.get("source")
    nested_source = data_cfg.get("source")
    if top_level_source is not None and nested_source is not None:
        raise JobConfigurationError(
            "provide canonical source in either config.source or data.source, not both"
        )

    source_payload = top_level_source if top_level_source is not None else nested_source
    source: CanonicalSourceInput
    if source_payload is not None:
        extra_data_keys = set(data_cfg) - {"source"}
        if extra_data_keys:
            raise JobConfigurationError(
                "canonical source cannot be combined with legacy data fields: "
                + ", ".join(sorted(extra_data_keys))
            )
        from pydantic import TypeAdapter

        try:
            source = TypeAdapter(CanonicalSourceInput).validate_python(source_payload)
        except ValidationError as e:
            raise JobConfigurationError(f"Invalid inference config: {e}") from e
    else:
        try:
            source = _legacy_json_source(data_cfg)
        except (ValidationError, ValueError) as e:
            raise JobConfigurationError(f"Invalid inference config: {e}") from e

    try:
        config = InferenceConfig(
            source=source,
            output_uri=output_uri,
            model_uri=model_uri or None,
            bundle_uri=bundle_uri,
            model_role=model_cfg.get("role", "inference"),
            # Passed through to Pydantic for strict parsing — bool("false")
            # is True, so the raw JSON value must reach the model field.
            unsafe_model=model_cfg.get("unsafe", False),
            storage_profile=model_cfg.get("storage_profile"),
            predictor_config=predictor_config,
            batch_size=ray_cfg.get("batch_size", 4096),
            concurrency=ray_cfg.get("concurrency", 4),
            num_cpus_per_actor=ray_cfg.get("num_cpus_per_actor", 1.0),
            num_gpus_per_actor=ray_cfg.get("num_gpus_per_actor", 0.0),
            output_compression=output_cfg.get("compression", "zstd"),
            min_rows_per_file=output_cfg.get("min_rows_per_file"),
        )
    except (ValidationError, ValueError) as e:
        raise JobConfigurationError(f"Invalid inference config: {e}") from e

    return run_batch_inference(config)
