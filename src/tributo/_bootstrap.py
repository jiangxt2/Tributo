"""Internal composition roots for first-party integrations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tributo._common.storage_profiles import StorageProfileResolver

if TYPE_CHECKING:
    from tributo.exporting.gc import BundleGarbageCollectorBackend
    from tributo.exporting.protocols import (
        ExportSourceProvider,
        ExportValidator,
        ModelExporter,
    )
    from tributo.exporting.repository import BundleAliasStore, BundleRepository
    from tributo.exporting.runtime import BundleModelFlavor


def first_party_export_plugins() -> tuple[
    tuple[type[ModelExporter], ...], tuple[type[ExportValidator], ...]
]:
    """Return built-in exporters and validators without entry-point metadata."""
    from tributo.integrations.exporters.hf_onnx import HuggingFaceONNXExporter
    from tributo.integrations.exporters.onnx_quantizer import ONNXQuantizer
    from tributo.integrations.exporters.torch_export import TorchExportExporter
    from tributo.integrations.exporters.torch_onnx import TorchONNXExporter
    from tributo.integrations.exporters.torch_safetensors import (
        TorchSafetensorsExporter,
    )
    from tributo.integrations.exporters.xgboost_native import (
        XGBoostJSONExporter,
        XGBoostUBJExporter,
    )
    from tributo.integrations.exporters.xgboost_onnx import XGBoostONNXExporter
    from tributo.integrations.validators.onnx_runtime import ONNXRuntimeValidator

    exporters: tuple[type[ModelExporter], ...] = (
        XGBoostONNXExporter,
        XGBoostUBJExporter,
        XGBoostJSONExporter,
        TorchSafetensorsExporter,
        TorchONNXExporter,
        TorchExportExporter,
        HuggingFaceONNXExporter,
        ONNXQuantizer,
    )
    validators: tuple[type[ExportValidator], ...] = (ONNXRuntimeValidator,)
    return exporters, validators


def first_party_source_providers() -> tuple[type[ExportSourceProvider], ...]:
    """Return built-in checkpoint providers without entry-point metadata."""
    from tributo.integrations.sources.ray_dnn import RayDnnSourceProvider
    from tributo.integrations.sources.ray_pu import RayPUSourceProvider
    from tributo.integrations.sources.ray_xgboost import RayXGBoostSourceProvider

    return (
        RayXGBoostSourceProvider,
        RayDnnSourceProvider,
        RayPUSourceProvider,
    )


def first_party_model_flavors() -> tuple[type[BundleModelFlavor], ...]:
    """Return built-in executable flavors without entry-point metadata."""
    from tributo.integrations.flavors.onnx_runtime import ONNXRuntimeFlavor
    from tributo.integrations.flavors.xgboost_native import XGBoostNativeFlavor

    return (ONNXRuntimeFlavor, XGBoostNativeFlavor)


def first_party_bundle_storage_adapters(
    storage_resolver: StorageProfileResolver | None,
) -> tuple[tuple[BundleRepository, ...], tuple[BundleAliasStore, ...]]:
    """Construct built-in storage adapters without distribution metadata.

    Ray ``runtime_env.py_modules`` and source-only deployments replace Python
    modules but not the installed package's entry-point metadata.  The
    internal composition root therefore owns first-party construction;
    entry points remain the extension mechanism for third-party adapters.
    """
    from tributo.integrations.storage.bundle_repository import (
        LocalBundleAliasStore,
        LocalBundleRepository,
        S3BundleAliasStore,
        S3BundleRepository,
    )

    repositories: tuple[BundleRepository, ...] = (
        LocalBundleRepository(storage_resolver),
        S3BundleRepository(storage_resolver),
    )
    alias_stores: tuple[BundleAliasStore, ...] = (
        LocalBundleAliasStore(storage_resolver),
        S3BundleAliasStore(storage_resolver),
    )
    return repositories, alias_stores


def first_party_bundle_garbage_collector(
    storage_resolver: StorageProfileResolver | None,
) -> BundleGarbageCollectorBackend:
    """Construct the first-party bundle-maintenance adapter."""
    from tributo.integrations.storage.gc import S3BundleGarbageCollector

    return S3BundleGarbageCollector(storage_resolver)
