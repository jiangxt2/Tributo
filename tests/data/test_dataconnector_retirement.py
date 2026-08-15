"""Deletion and neutral-contract boundaries for the former DataConnector API."""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

import pytest

import tributo.data as data
from tributo.data import S3Config, WriteMode
from tributo.data.base import S3Config as ShimS3Config
from tributo.data.base import WriteMode as ShimWriteMode
from tributo.data.contracts.storage import S3Config as CanonicalS3Config


def test_shared_contracts_keep_only_narrow_base_reexports() -> None:
    assert S3Config is CanonicalS3Config is ShimS3Config
    assert WriteMode is ShimWriteMode
    assert "DataConnector" not in data.__all__
    assert "get_connector" not in data.__all__
    assert "register_connector" not in data.__all__
    assert "list_connectors" not in data.__all__
    assert not hasattr(data, "DataConnector")
    assert not hasattr(data, "get_connector")
    assert not hasattr(data, "register_connector")
    assert not hasattr(data, "list_connectors")


def test_removed_symbols_fail_as_normal_imports() -> None:
    with pytest.raises(ImportError):
        exec("from tributo.data import DataConnector", {})
    with pytest.raises(ImportError):
        exec("from tributo.data.base import DataConnector", {})


def test_package_metadata_keeps_non_connector_entry_point_groups() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    groups = pyproject["project"]["entry-points"]

    assert "tributo.connectors" not in groups
    assert "tributo.write_bindings" in groups
    assert "tributo.trainers" in groups
    assert "tributo.algorithms" in groups
    assert "tributo.source_providers" in groups


@pytest.mark.parametrize(
    "module_name",
    [
        "tributo.data.registry",
        "tributo.data._compat_read",
        "tributo.data.parquet",
        "tributo.data.csv",
        "tributo.data.iceberg",
        "tributo.data.lance",
        "tributo.data.writing.compatibility",
    ],
)
def test_removed_connector_modules_fail_as_normal_imports(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_data_import_does_not_initialize_removed_connector_modules() -> None:
    removed = {
        "tributo.data.registry",
        "tributo.data._compat_read",
        "tributo.data.parquet",
        "tributo.data.csv",
        "tributo.data.iceberg",
        "tributo.data.lance",
        "tributo.data.writing.compatibility",
    }
    assert removed.isdisjoint(sys.modules)
