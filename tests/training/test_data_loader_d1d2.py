"""D1+D2 loader contract: three-entry semantics, rollback switch, parity."""

from __future__ import annotations

import importlib
from pathlib import Path

import pyarrow as pa
import pytest

from tributo.exceptions import JobConfigurationError
from tributo.training import data_loader as dl_module
from tributo.training.data_loader import (
    load_ray_dataset_from_config,
    load_ray_dataset_from_source,
)

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Tip.*future versions of Ray.*:FutureWarning",
        "ignore::pytest.PytestUnraisableExceptionWarning",
    ),
    pytest.mark.usefixtures("ray_local_runtime"),
]


@pytest.fixture
def parquet_file(tmp_path: Path) -> str:
    table = pa.table({"id": [1, 2], "kind": ["p", "p"]})
    path = tmp_path / "data.parquet"
    pa.parquet.write_table(table, path)
    return str(path)


@pytest.fixture
def csv_file(tmp_path: Path) -> str:
    path = tmp_path / "data.csv"
    path.write_text("id,kind\n3,c\n4,c\n")
    return str(path)


def _reload_with_backend(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    monkeypatch.setenv("TRIBUTO_DATA_BACKEND", backend)
    importlib.reload(dl_module)


class TestThreeEntrySemantics:
    """type=csv in source is real CSV; in config it keeps the Parquet default."""

    def test_source_type_csv_is_real_csv(self, csv_file: str) -> None:
        ds = load_ray_dataset_from_source({"type": "csv", "path": csv_file})
        df = ds.to_pandas()
        assert list(df["kind"]) == ["c", "c"]

    def test_config_type_csv_defaults_to_parquet(self, parquet_file: str) -> None:
        with pytest.warns(FutureWarning):
            ds = load_ray_dataset_from_config({"type": "csv", "path": parquet_file})
        df = ds.to_pandas()
        assert list(df["kind"]) == ["p", "p"]

    def test_provider_shape_csv_is_real_csv(self, csv_file: str) -> None:
        ds = load_ray_dataset_from_source({"provider": "tributo.csv", "uri": csv_file})
        df = ds.to_pandas()
        assert list(df["kind"]) == ["c", "c"]

    def test_provider_shape_parquet(self, parquet_file: str) -> None:
        ds = load_ray_dataset_from_source(
            {"provider": "tributo.parquet", "uri": parquet_file}
        )
        df = ds.to_pandas()
        assert list(df["id"]) == [1, 2]

    def test_canonical_sql_route_to_clickhouse(self) -> None:
        # Resolution only — no connection is made by normalize/open.

        from pydantic import TypeAdapter

        from tributo.data.provider_registry import resolve_provider
        from tributo.data.source_config import CanonicalSourceInput

        cfg = TypeAdapter(CanonicalSourceInput).validate_python(
            {"type": "sql", "dialect": "clickhouse", "sql": "SELECT 1"}
        )
        assert resolve_provider(cfg).provider_id == "tributo.clickhouse"


class TestRollbackSwitch:
    """TRIBUTO_DATA_BACKEND=legacy bypasses the ProviderRegistry."""

    def test_legacy_backend_reads_file(
        self, monkeypatch: pytest.MonkeyPatch, parquet_file: str
    ) -> None:
        _reload_with_backend(monkeypatch, "legacy")
        try:
            ds = load_ray_dataset_from_source({"type": "parquet", "path": parquet_file})
            df = ds.to_pandas()
            assert list(df["id"]) == [1, 2]
        finally:
            _reload_with_backend(monkeypatch, "provider")

    def test_legacy_backend_rejects_provider_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reload_with_backend(monkeypatch, "legacy")
        try:
            from tributo.exceptions import JobConfigurationError

            with pytest.raises(
                JobConfigurationError, match="TRIBUTO_DATA_BACKEND=provider"
            ):
                load_ray_dataset_from_source(
                    {"provider": "tributo.parquet", "uri": "x"}
                )
        finally:
            _reload_with_backend(monkeypatch, "provider")

    def test_legacy_config_entrypoint_rejects_provider_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reload_with_backend(monkeypatch, "provider")
        with pytest.raises(JobConfigurationError, match="load_ray_dataset_from_source"):
            with pytest.warns(FutureWarning):
                load_ray_dataset_from_config(
                    {"provider": "tributo.parquet", "uri": "data.parquet"}
                )

    def test_legacy_backend_keeps_csv_default(
        self, monkeypatch: pytest.MonkeyPatch, parquet_file: str
    ) -> None:
        _reload_with_backend(monkeypatch, "legacy")
        try:
            with pytest.warns(FutureWarning):
                ds = load_ray_dataset_from_config({"type": "csv", "path": parquet_file})
            df = ds.to_pandas()
            assert list(df["kind"]) == ["p", "p"]
        finally:
            _reload_with_backend(monkeypatch, "provider")

    def test_default_backend_is_provider(self) -> None:
        assert dl_module.DATA_BACKEND == "provider"


class TestParity:
    """canonical vs legacy produce equivalent data for the same input."""

    def test_parquet_parity(self, parquet_file: str) -> None:
        ds_new = load_ray_dataset_from_source({"type": "parquet", "path": parquet_file})
        df_new = ds_new.to_pandas()
        assert list(df_new["id"]) == [1, 2]

    def test_relative_path_resolution(self, tmp_path: Path) -> None:
        # Relative path resolves against the given project root like the old loader.
        table = pa.table({"a": [1]})
        pa.parquet.write_table(table, tmp_path / "rel.parquet")
        ds = load_ray_dataset_from_source(
            {"type": "parquet", "path": "rel.parquet"},
            project_root_path=tmp_path,
        )
        assert ds.count() == 1

    def test_config_relative_path_resolves_project_root(self, tmp_path: Path) -> None:
        # Provider mode must honour project_root_path like the legacy loader.
        table = pa.table({"a": [5]})
        pa.parquet.write_table(table, tmp_path / "rel.parquet")
        with pytest.warns(FutureWarning):
            ds = load_ray_dataset_from_config(
                {"type": "parquet", "path": "rel.parquet"},
                project_root_path=tmp_path,
            )
        assert ds.count() == 1

    def test_unknown_type_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            load_ray_dataset_from_source({"type": "kafka", "bootstrap": "x"})
