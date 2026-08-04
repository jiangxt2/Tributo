"""DataConnector 注册表测试。"""

from __future__ import annotations

import sys

import pytest

from tributo._common.dependencies import (
    LANCE,
    PYICEBERG,
    DependencyState,
    probe_dependency,
)
from tributo.data.base import DataConnector
from tributo.data.registry import get_connector, list_connectors, register_connector
from tributo.exceptions import JobConfigurationError


class TestRegistry:
    """注册表核心功能测试。"""

    def test_list_connectors_has_defaults(self):
        """内置连接器应已注册。"""
        connectors = list_connectors()
        assert "parquet" in connectors
        if probe_dependency(LANCE).state is not DependencyState.MISSING:
            assert "lance" in connectors
        if probe_dependency(PYICEBERG).state is not DependencyState.MISSING:
            assert "iceberg" in connectors

    def test_get_connector_returns_instance(self):
        connector = get_connector("parquet")
        assert isinstance(connector, DataConnector)

    def test_get_connector_unknown_raises(self):
        with pytest.raises(JobConfigurationError, match="Unknown connector"):
            get_connector("nonexistent")

    def test_register_duplicate_raises(self):
        with pytest.raises(JobConfigurationError, match="already registered"):
            register_connector("parquet", type(get_connector("parquet")))

    def test_register_custom_connector(self):
        class MyConnector(DataConnector):
            def read(self, **kwargs):
                raise NotImplementedError

            def write(self, dataset, **kwargs):
                raise NotImplementedError

        register_connector("_test_custom", MyConnector)
        assert "_test_custom" in list_connectors()
        assert isinstance(get_connector("_test_custom"), MyConnector)


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
