"""Real PostgreSQL conformance for Ray Data SQL readers."""

from __future__ import annotations

import os
import uuid

import pytest

# The CI unit-test matrix installs only the ``dev`` extra; skip collection
# when the optional engine/connector dependencies are absent. The dedicated
# integration jobs install them via their extras.
try:
    import psycopg
    from psycopg import sql
except ModuleNotFoundError:
    pytest.skip("requires the data and postgresql extras", allow_module_level=True)

from tests.data.ingestion_conformance import assert_ray_conformance
from tributo.data import (
    IngestionRequest,
    IngestionRuntimeContext,
    RayDataHandle,
    SqlPartitioning,
    SqlSourceConfig,
    open_ingestion,
)
from tributo.exceptions import JobConfigurationError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ingestion_conformance,
    pytest.mark.usefixtures("ray_local_runtime"),
    pytest.mark.filterwarnings(
        "ignore:Tip.*future versions of Ray.*:FutureWarning",
        "ignore::pytest.PytestUnraisableExceptionWarning",
    ),
]


def test_postgresql_ray_conformance() -> None:
    host = os.environ.get("TRIBUTO_POSTGRESQL_HOST", "127.0.0.1")
    port = int(os.environ.get("TRIBUTO_POSTGRESQL_PORT", "5432"))
    database = os.environ.get("TRIBUTO_POSTGRESQL_DB", "tributo")
    user = os.environ.get("TRIBUTO_POSTGRESQL_USER", "tributo")
    password = os.environ.get("TRIBUTO_POSTGRESQL_PASSWORD", "tributo-test")
    table = f"ingestion_{uuid.uuid4().hex}"
    connection_options = {
        "host": host,
        "port": port,
        "dbname": database,
        "user": user,
        "password": password,
    }
    with psycopg.connect(**connection_options) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "CREATE TABLE {} (id BIGINT PRIMARY KEY, category TEXT NOT NULL)"
                ).format(sql.Identifier(table))
            )
            cursor.executemany(
                sql.SQL("INSERT INTO {} (id, category) VALUES (%s, %s)").format(
                    sql.Identifier(table)
                ),
                [(1, "drop"), (2, "keep"), (3, "keep")],
            )
        connection.commit()

    source = SqlSourceConfig(
        dialect="postgresql",
        host=host,
        port=port,
        database=database,
        database_schema="public",
        user=user,
        password=password,
        table=table,
        columns=["id", "category"],
    )
    context = IngestionRuntimeContext()
    ray_result = open_ingestion(IngestionRequest(source=source, engine="ray"), context)
    try:
        assert isinstance(ray_result.handle, RayDataHandle)
        assert_ray_conformance(
            ray_result,
            ray_result.handle.dataset.take_all(),
            expected_rows=[
                {"id": 1, "category": "drop"},
                {"id": 2, "category": "keep"},
                {"id": 3, "category": "keep"},
            ],
            require_worker_validation=False,
        )
        parallel_source = source.model_copy(
            update={"partitioning": SqlPartitioning(column="id", num_partitions=2)}
        )
        with pytest.raises(
            JobConfigurationError, match="unsupported_postgresql_parallel_read"
        ):
            open_ingestion(
                IngestionRequest(source=parallel_source, engine="ray"), context
            )
    finally:
        ray_result.close()
        with psycopg.connect(**connection_options) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table))
                )
            connection.commit()
