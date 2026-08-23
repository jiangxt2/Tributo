"""Inference result-sink adapters."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tributo.integrations.sinks.data_write import DataWriteResultSink
    from tributo.integrations.sinks.lance import LanceResultSink
    from tributo.integrations.sinks.parquet import ParquetResultSink


def __getattr__(name: str) -> Any:
    """Load only the requested sink and its runtime dependencies."""
    if name == "LanceResultSink":
        from tributo.integrations.sinks.lance import LanceResultSink

        return LanceResultSink
    if name == "DataWriteResultSink":
        from tributo.integrations.sinks.data_write import DataWriteResultSink

        return DataWriteResultSink
    if name == "ParquetResultSink":
        from tributo.integrations.sinks.parquet import ParquetResultSink

        return ParquetResultSink
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DataWriteResultSink", "LanceResultSink", "ParquetResultSink"]
