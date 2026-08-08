"""Typed options owned by Tributo's first-party exporter plugins."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from tributo.util.annotations import PublicAPI


class _NoOptions(BaseModel):
    """Forbid accidental per-format selectors in a concrete exporter."""

    model_config = ConfigDict(extra="forbid")


@PublicAPI(stability="beta")
class XGBoostUBJOptions(_NoOptions):
    """Options for ``XGBoostUBJExporter``."""


@PublicAPI(stability="beta")
class XGBoostJSONOptions(_NoOptions):
    """Options for ``XGBoostJSONExporter``."""


@PublicAPI(stability="beta")
class XGBoostNativeOptions(BaseModel):
    """Deprecated compatibility options for ``XGBoostNativeExporter``."""

    model_config = ConfigDict(extra="forbid")

    fmt: Literal["ubj", "json"] = "ubj"


@PublicAPI(stability="beta")
class XGBoostONNXOptions(BaseModel):
    """Options for ``XGBoostONNXExporter``."""

    model_config = ConfigDict(extra="forbid")

    opset: Literal[12] = 12


@PublicAPI(stability="beta")
class TorchONNXOptions(BaseModel):
    """Options for ``TorchONNXExporter``."""

    model_config = ConfigDict(extra="forbid")

    opset: Literal[18] = 18
    dynamo: bool = True
    external_data: bool = False


@PublicAPI(stability="beta")
class SafetensorsOptions(BaseModel):
    """Options for ``TorchSafetensorsExporter``."""

    model_config = ConfigDict(extra="forbid")

    max_shard_size: str = "5GB"


@PublicAPI(stability="beta")
class HFONNXOptions(BaseModel):
    """Options for ``HuggingFaceONNXExporter``."""

    model_config = ConfigDict(extra="forbid")

    opset: int | None = None
    task: str | None = None


@PublicAPI(stability="beta")
class ONNXQuantizerOptions(BaseModel):
    """Options for ``ONNXQuantizer``."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["dynamic-int8"] = "dynamic-int8"


__all__ = [
    "HFONNXOptions",
    "ONNXQuantizerOptions",
    "SafetensorsOptions",
    "TorchONNXOptions",
    "XGBoostJSONOptions",
    "XGBoostNativeOptions",
    "XGBoostONNXOptions",
    "XGBoostUBJOptions",
]
