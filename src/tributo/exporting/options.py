"""Per-exporter typed Pydantic options models.

Every exporter declares its own options schema via ``options_model``.
The Planner validates raw ``ExportTarget.options`` against the selected
candidate's ``options_model`` with ``extra="forbid"``.

All models inherit ``BaseModel`` with ``ConfigDict(extra="forbid")``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from tributo.util.annotations import PublicAPI

# ── XGBoost native exporter ──────────────────────────────────────────────────


@PublicAPI(stability="beta")
class XGBoostNativeOptions(BaseModel):
    """Options for ``XGBoostNativeExporter`` (UBJ / JSON)."""

    model_config = ConfigDict(extra="forbid")

    fmt: Literal["ubj", "json"] = "ubj"


# ── XGBoost ONNX exporter ───────────────────────────────────────────────────


@PublicAPI(stability="beta")
class XGBoostONNXOptions(BaseModel):
    """Options for ``XGBoostONNXExporter``.

    First-phase opset is locked to 12 (onnxmltools constraint).
    """

    model_config = ConfigDict(extra="forbid")

    opset: Literal[12] = 12


# ── PyTorch ONNX exporter ───────────────────────────────────────────────────


@PublicAPI(stability="beta")
class TorchONNXOptions(BaseModel):
    """Options for ``TorchONNXExporter``."""

    model_config = ConfigDict(extra="forbid")

    # Bundle mode defaults to the TorchDynamo exporter path (PyTorch >= 2.5
    # recommended path per the export plan); legacy single-path export keeps
    # its own ``dynamo=False`` default and is unaffected by this option.
    # Opset is locked to 18 — the plan only promises CI-verified values.
    opset: Literal[18] = 18
    dynamo: bool = True
    external_data: bool = False


# ── PyTorch Safetensors exporter ─────────────────────────────────────────────


@PublicAPI(stability="beta")
class SafetensorsOptions(BaseModel):
    """Options for ``TorchSafetensorsExporter``."""

    model_config = ConfigDict(extra="forbid")

    max_shard_size: str = "5GB"


# ── HF ONNX exporter ────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class HFONNXOptions(BaseModel):
    """Options for ``HFONNXExporter``.

    ``opset`` and ``task`` default to ``None`` — the exporter resolves them
    via the Optimum config for the given model type / task.
    """

    model_config = ConfigDict(extra="forbid")

    opset: int | None = None
    task: str | None = None


# ── ONNX quantizer ──────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class ONNXQuantizerOptions(BaseModel):
    """Options for ``ONNXQuantizer`` (artifact → artifact)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["dynamic-int8"] = "dynamic-int8"


# ── Registry of known options models (for discovery / docs) ──────────────────

_BUILTIN_OPTIONS: dict[str, type[BaseModel] | None] = {
    "xgboost-native-v1": XGBoostNativeOptions,
    "xgboost-onnx-v1": XGBoostONNXOptions,
    "torch-onnx-v1": TorchONNXOptions,
    "torch-safetensors-v1": SafetensorsOptions,
    "hf-onnx-v1": HFONNXOptions,
    "onnx-quantizer-v1": ONNXQuantizerOptions,
    # Options models defined alongside their exporters in integrations/.
    "torch-export-v1": None,
}
