"""Ray Dataset distributed data loading, supporting local CSV/Parquet, S3 Parquet, ClickHouse."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from tributo._common import find_project_root

if TYPE_CHECKING:
    import pandas as pd
    import ray.data


def load_ray_dataset_from_config(
    data_config: dict,
    project_root_path: Path | None = None,
) -> "ray.data.Dataset":
    """Load a Ray Dataset from a YAML ``data`` section (lazy, data does not land in Driver memory).

    Supports csv / parquet (local), s3 (via DataConnector), clickhouse.

    Args:
        data_config: The ``data`` dictionary from YAML.
        project_root_path: Project root directory for resolving relative paths.

    Returns:
        Ray Dataset, lazily evaluated.
    """
    import ray.data

    data_type = data_config.get("type", "csv")

    if data_type == "csv":
        root = project_root_path or find_project_root()
        file_path = data_config.get("path", "")
        fmt = data_config.get("format", "")
        full_path = Path(file_path)
        if not full_path.is_absolute():
            full_path = root / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"Data file not found: {full_path}")
        if fmt == "csv":
            return ray.data.read_csv(str(full_path))
        return ray.data.read_parquet(str(full_path))

    if data_type == "s3":
        from tributo.data import S3Config, get_connector

        uri = data_config.get("uri", "")
        if not uri:
            raise ValueError("missing s3 uri in data config")
        fmt = data_config.get("format", "parquet")
        if fmt not in {"parquet", "csv"}:
            raise ValueError(f"unsupported s3 format: {fmt!r}")
        s3_cfg = data_config.get("s3", {})
        s3_config = S3Config(**s3_cfg) if s3_cfg else None
        columns = data_config.get("columns")
        connector = get_connector(fmt)
        return connector.read(path=uri, s3=s3_config, columns=columns)

    if data_type == "clickhouse":
        return _load_clickhouse_dataset(data_config)

    raise ValueError(f"unsupported data type: {data_type}")


def _load_clickhouse_dataset(data_config: dict) -> "ray.data.Dataset":
    """Execute a ClickHouse SQL query and return the result as a Ray Dataset.

    Uses ``clickhouse_connect`` to execute the parameterized SQL on the
    ClickHouse server and converts the resulting PyArrow table into a Ray
    Dataset.  For large result sets, consider pre-exporting to Parquet/S3
    instead.

    Args:
        data_config: Data config dict containing:
            ch_host, ch_port, ch_database, ch_user, ch_password,
            ch_sql, ch_sql_params.

    Returns:
        Ray Dataset created from the query result.
    """
    import ray.data

    sql = data_config.get("ch_sql", "")
    if not sql:
        raise ValueError("missing ch_sql in clickhouse data config")

    # Resolve connection params: explicit config > env var > built-in default.
    # Uses ``is None`` rather than ``or`` so that falsy values (empty string,
    # port 0) are preserved when explicitly set in config.
    host = data_config.get("ch_host")
    if host is None:
        host = os.getenv("TRIBUTO_CLICKHOUSE_HOST", "localhost")

    port_cfg = data_config.get("ch_port")
    if port_cfg is not None:
        port = int(port_cfg)
    else:
        port_env = os.getenv("TRIBUTO_CLICKHOUSE_PORT")
        port = int(port_env) if port_env else 8123

    database = data_config.get("ch_database")
    if database is None:
        database = os.getenv("TRIBUTO_CLICKHOUSE_DB", "")

    user = data_config.get("ch_user")
    if user is None:
        user = os.getenv("TRIBUTO_CLICKHOUSE_USER", "default")

    password = data_config.get("ch_password")
    if password is None:
        password = os.getenv("TRIBUTO_CLICKHOUSE_PASSWORD", "")
    params = data_config.get("ch_sql_params", {}) or {}

    import clickhouse_connect  # type: ignore[import-untyped]

    client = clickhouse_connect.get_client(
        host=host,
        port=port,
        username=user,
        password=password,
        database=database,
    )
    result = client.query_arrow(sql, parameters=params)
    table = result  # clickhouse_connect returns pyarrow.Table directly

    if table is None or table.num_rows == 0:
        raise ValueError("ClickHouse query returned empty result")

    # Convert to Ray Dataset — the Arrow table lives in Driver memory,
    # then Ray distributes blocks to workers.  For large result sets,
    # set a reasonable block size so workers get manageable chunks.
    num_blocks = max(1, table.num_rows // 10000)
    return ray.data.from_arrow(table, override_num_blocks=num_blocks)


def load_dataframe_from_config(
    data_config: dict,
    project_root_path: Path | None = None,
) -> "pd.DataFrame":
    """Load a DataFrame from a YAML ``data`` section (fully loaded into Driver memory).

    ⚠️ Only suitable for small datasets. Use ``load_ray_dataset_from_config`` for large data.

    Args:
        data_config: The ``data`` dictionary from YAML.
        project_root_path: Project root directory for resolving relative paths.

    Returns:
        pandas DataFrame.
    """
    ds = load_ray_dataset_from_config(data_config, project_root_path)
    return ds.to_pandas()  # type: ignore[no-any-return]
