"""Distributed batch inference pipeline: S3 Parquet → ONNX inference → S3 Parquet.

End-to-end Ray Data streaming; the Driver only orchestrates, data does not pass through the Head node.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tributo.inference.base import BasePredictor

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tributo.exceptions import JobConfigurationError
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class InferenceConfig(BaseModel):
    """Batch inference task configuration.

    Direct construction raises ``pydantic.ValidationError`` on field validation failure;
    when accessed via the ``run_inference_from_json`` entry point, it is converted to
    ``JobConfigurationError``.

    Attributes:
        input_uri: URI of input data (S3:// or local path, or ClickHouse SQL).
        s3_config: S3 auth configuration.
        clickhouse_config: ClickHouse connection config (required when type=clickhouse).
        output_uri: Output URI for inference results.
        model_uri: Model path (local or s3://).
        feature_columns: List of feature column names needed for inference (column pruning).
        predictor_config: Passthrough dict for predictor-specific config.
    """

    model_config = ConfigDict(frozen=True)

    input_uri: str = Field(min_length=1)
    output_uri: str = Field(min_length=1)
    model_uri: str = Field(min_length=1)

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


@PublicAPI(stability="beta")
def run_batch_inference(
    config: InferenceConfig,
    predictor_cls: type[BasePredictor] | None = None,
) -> dict[str, Any]:
    """Execute the distributed batch inference pipeline.

    Data flow:
    1. ray.data.read_parquet lazy-loads from S3 (only reads feature_columns);
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

    from tributo.data import get_connector
    from tributo.inference.batch_predictor import XGBoostONNXPredictor
    from tributo.training.data_loader import load_ray_dataset_from_config

    if predictor_cls is None:
        predictor_cls = XGBoostONNXPredictor

    # Build predictor_config: merge predictor-specific config + s3_config + feature_columns
    predictor_config = {**config.predictor_config}
    if config.s3_config:
        predictor_config["s3_config"] = config.s3_config
    predictor_config["feature_names"] = config.feature_columns or None

    # If feature_columns is not set, auto-read from model metadata
    feature_columns = config.feature_columns
    if not feature_columns:
        feature_columns = predictor_cls.get_feature_names(
            config.model_uri, predictor_config
        )
        if not feature_columns:
            raise JobConfigurationError(
                "feature_columns is empty and model has no feature_names metadata. "
                "Either specify feature_columns in config JSON or re-export the model."
            )
        predictor_config["feature_names"] = feature_columns
        logger.info(
            "Auto-loaded feature_columns from model metadata: %s", feature_columns
        )

    # ── 1. Load data based on data_type ──
    if config.data_type == "clickhouse":
        ch_cfg = {
            "type": "clickhouse",
            "ch_host": config.ch_host,
            "ch_port": config.ch_port,
            "ch_database": config.ch_database,
            "ch_user": config.ch_user,
            "ch_password": config.ch_password,
            "ch_sql": config.ch_sql,
        }
        ds = load_ray_dataset_from_config(ch_cfg)
        # Read all from ClickHouse, then select columns by feature_columns
        if feature_columns:
            ds = ds.select_columns(feature_columns)
    elif config.input_uri.startswith("s3://"):
        from tributo.data.base import S3Config

        s3 = (
            S3Config(
                access_key_id=config.s3_config.get("access_key_id"),
                secret_access_key=config.s3_config.get("secret_access_key"),
                endpoint=config.s3_config.get("endpoint"),
                region=config.s3_config.get("region"),
            )
            if config.s3_config
            else None
        )
        ds = get_connector("parquet").read(
            path=config.input_uri,
            columns=feature_columns or None,
            s3=s3,
        )
    else:
        # Local path
        import pyarrow.fs as pafs

        ds = ray.data.read_parquet(config.input_uri, columns=feature_columns or None)

    # ── 2. Distributed inference (ActorPoolStrategy keeps models resident) ──
    logger.info(
        "Starting inference: predictor=%s model=%s concurrency=%d batch_size=%d "
        "cpus_per_actor=%.1f gpus_per_actor=%.1f",
        predictor_cls.__name__,
        config.model_uri,
        config.concurrency,
        config.batch_size,
        config.num_cpus_per_actor,
        config.num_gpus_per_actor,
    )

    ds = ds.map_batches(
        predictor_cls,
        fn_constructor_args=(config.model_uri, predictor_config),
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

        s3 = (
            S3Config(
                access_key_id=config.s3_config.get("access_key_id"),
                secret_access_key=config.s3_config.get("secret_access_key"),
                endpoint=config.s3_config.get("endpoint"),
                region=config.s3_config.get("region"),
            )
            if config.s3_config
            else None
        )
        write_kwargs["filesystem"] = pafs.S3FileSystem(**to_pyarrow_s3_kwargs(s3))
        output_path = config.output_uri.removeprefix("s3://")
    else:
        output_path = config.output_uri

    ds.write_parquet(output_path, **write_kwargs)

    logger.info(
        "Batch inference complete: %s → %s",
        config.input_uri,
        config.output_uri,
    )
    return {
        "input_path": config.input_uri,
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
          uri: s3://bucket/models/model.onnx   # or local path
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
    s3_cfg = data_cfg.get("s3", {})

    input_uri = data_cfg.get("uri") or data_cfg.get("input", "")
    model_uri = model_cfg.get("uri") or model_cfg.get("path", "")
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

    # Data source type and ClickHouse configuration
    data_type = data_cfg.get("type", "csv")
    ch_cfg = data_cfg.get("clickhouse", {})

    # input_uri is not required when type=clickhouse
    if data_type == "clickhouse":
        input_uri = input_uri or "clickhouse://"

    # Resolve ClickHouse connection params: explicit config > env var > default.
    # Uses ``is None`` rather than ``or`` so that falsy values (empty string,
    # port 0) are preserved when explicitly set in config.
    _ch_host = ch_cfg.get("host")
    if _ch_host is None:
        _ch_host = os.getenv("TRIBUTO_CLICKHOUSE_HOST", "localhost")

    _ch_port_cfg = ch_cfg.get("port")
    if _ch_port_cfg is not None:
        _ch_port = int(_ch_port_cfg)
    else:
        _port_env = os.getenv("TRIBUTO_CLICKHOUSE_PORT")
        _ch_port = int(_port_env) if _port_env else 8123

    _ch_database = ch_cfg.get("database")
    if _ch_database is None:
        _ch_database = os.getenv("TRIBUTO_CLICKHOUSE_DB", "")

    _ch_user = ch_cfg.get("user")
    if _ch_user is None:
        _ch_user = os.getenv("TRIBUTO_CLICKHOUSE_USER", "default")

    _ch_password = ch_cfg.get("password")
    if _ch_password is None:
        _ch_password = os.getenv("TRIBUTO_CLICKHOUSE_PASSWORD", "")

    try:
        config = InferenceConfig(
            input_uri=input_uri,
            output_uri=output_uri,
            model_uri=model_uri,
            data_type=data_type,
            feature_columns=data_cfg.get("feature_columns", []),
            predictor_config=predictor_config,
            batch_size=ray_cfg.get("batch_size", 4096),
            concurrency=ray_cfg.get("concurrency", 4),
            num_cpus_per_actor=ray_cfg.get("num_cpus_per_actor", 1.0),
            num_gpus_per_actor=ray_cfg.get("num_gpus_per_actor", 0.0),
            output_compression=output_cfg.get("compression", "zstd"),
            min_rows_per_file=output_cfg.get("min_rows_per_file"),
            s3_config=s3_cfg,
            ch_host=_ch_host,
            ch_port=_ch_port,
            ch_database=_ch_database,
            ch_user=_ch_user,
            ch_password=_ch_password,
            ch_sql=ch_cfg.get("sql", ""),
        )
    except ValidationError as e:
        raise JobConfigurationError(f"Invalid inference config: {e}") from e

    return run_batch_inference(config)
