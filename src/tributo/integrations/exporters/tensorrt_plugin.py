"""TensorRT export plugin interface — external plugin contract.

TensorRT requires a compiled ``.so`` / ``.dylib`` plugin library containing
custom ops that ONNX-to-TensorRT cannot directly convert.  This module
defines the plugin interface that external TensorRT exporters must follow.

This module does NOT implement TensorRT — it declares the contract.
A separate ``tributo-tensorrt`` package provides the concrete implementation.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from tributo.util.annotations import PublicAPI


# ── Options ──────────────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class TensorRTOptions(BaseModel):
    """Options for TensorRT export.

    Used by both first-party (TorchTRT) and third-party TensorRT exporters.
    """

    model_config = ConfigDict(extra="forbid")

    precision: str = Field(
        default="fp16",
        pattern=r"^(fp32|fp16|int8)$",
    )
    max_workspace_size_gb: float = Field(default=4.0, ge=0.5, le=256.0)
    max_batch_size: int = Field(default=1, ge=1)
    min_shapes: dict[str, list[int]] = Field(default_factory=dict)
    opt_shapes: dict[str, list[int]] = Field(default_factory=dict)
    max_shapes: dict[str, list[int]] = Field(default_factory=dict)
    calibrator_type: str | None = Field(
        default=None,
        pattern=r"^(entropy|minmax|percentile)$",
    )
    plugin_library_paths: list[str] = Field(
        default_factory=list,
        description="Paths to compiled .so/.dylib TensorRT plugin libraries.",
    )
    disable_tf32: bool = False
    strict_type_constraints: bool = False
    timing_cache_path: str | None = None


# ── Plugin library descriptor ─────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class TensorRTPluginDescriptor(BaseModel):
    """Describes a TensorRT plugin library for discovery and documentation.

    Plugin library authors register their library via the
    ``tributo.tensorrt_plugins`` entry point group, pointing to a module
    or class that provides this descriptor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plugin_id: str = Field(..., min_length=1)
    plugin_name: str = ""
    plugin_version: str = "0.1.0"
    library_path: str = Field(
        ..., description="Path to compiled .so/.dylib."
    )
    ops_supported: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Names of ONNX ops this plugin replaces."
    )
    platform: str = Field(
        default="linux_x86_64",
        pattern=r"^(linux_x86_64|linux_aarch64|macos_arm64|windows_x86_64)$",
    )
    cuda_version: str | None = None
    tensorrt_min_version: str | None = None


# ── Plugin discovery protocol ────────────────────────────────────────────────────


@runtime_checkable
@PublicAPI(stability="beta")
class TensorRTPluginProvider(Protocol):
    """Contract for TensorRT plugin library providers.

    Implementations registered via ``tributo.tensorrt_plugins`` must conform
    to this protocol.  The ``discover`` classmethod returns the plugin
    descriptor(s) that this package provides.
    """

    api_version: ClassVar[int]
    provider_id: ClassVar[str]

    @classmethod
    def discover(cls) -> list[TensorRTPluginDescriptor]:
        """Return all plugin descriptors provided by this package."""
        ...


# ── TensorRT export contract (for external implementation) ───────────────────────


@runtime_checkable
@PublicAPI(stability="beta")
class TensorRTExporterContract(Protocol):
    """What a full TensorRT exporter must satisfy.

    TensorRT export is deliberately kept as an external package
    (``tributo-tensorrt``) because of the large dependency surface
    (TensorRT SDK, CUDA, cuDNN) and platform-specific compilation.

    This protocol defines the stable contract so the core framework
    can accept TensorRT exporters without importing TensorRT.
    """

    api_version: ClassVar[int]
    exporter_id: ClassVar[str]
    priority: ClassVar[int]
    output_format: ClassVar[str]
    options_model: ClassVar[type[BaseModel]]
    validator_bindings: ClassVar[tuple[Any, ...]]
    mutates_source: ClassVar[bool]
    upstream_requirements: ClassVar[tuple[Any, ...]]

    @classmethod
    def supports(cls, request: Any) -> Any: ...

    def export(self, context: Any, source: Any, upstream: Any, target: Any) -> Any: ...
