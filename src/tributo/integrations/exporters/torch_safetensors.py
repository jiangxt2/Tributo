"""PyTorch Safetensors exporter — ``ModelExporter`` protocol implementation.

Exports a ``torch.nn.Module`` to HuggingFace safetensors format with
optional sharding.  Uses ``safetensors.torch.save_file``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar, Mapping

from pydantic import BaseModel

from tributo._common.dependencies import (
    SAFETENSORS,
    TORCH,
    DependencyState,
    probe_dependency,
)
from tributo.exporting.models import (
    ArtifactDraft,
    DraftFile,
    ExportContext,
    ExportSource,
    PlannedTarget,
    ProducerInfo,
    ResolvedArtifact,
    SupportRequest,
    SupportResult,
    ValidatorBinding,
)
from tributo.exporting.options import SafetensorsOptions
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class TorchSafetensorsExporter:
    """Export a PyTorch nn.Module to safetensors format.

    Produces one or more shard files (``model-00001-of-NNNNN.safetensors``)
    plus an ``model.safetensors.index.json`` when sharding is used.
    """

    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "torch-safetensors-v1"
    priority: ClassVar[int] = 90
    output_format: ClassVar[str] = "safetensors"
    source_kinds: ClassVar[tuple[str, ...]] = ("dnn_result", "torch_module")
    options_model: ClassVar[type[BaseModel]] = SafetensorsOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
        ValidatorBinding(validator_id="structure-v1", required=True),
    )
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        """Check whether the source is a PyTorch module."""
        if request.source_kind not in ("dnn_result", "torch_module"):
            return SupportResult(
                supported=False,
                code="UNSUPPORTED_SOURCE_KIND",
                reason=f"Expected source_kind='dnn_result' or 'torch_module', got {request.source_kind!r}",
            )
        missing: list[str] = []
        if probe_dependency(SAFETENSORS).state is not DependencyState.AVAILABLE:
            missing.append("safetensors")
        if probe_dependency(TORCH).state is not DependencyState.AVAILABLE:
            missing.append("torch")
        if missing:
            return SupportResult(
                supported=False,
                code="MISSING_DEPENDENCY",
                reason="safetensors>=0.4.3/torch>=2.5.0 required",
                missing_dependencies=tuple(missing),
            )
        return SupportResult(supported=True, code="OK")

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        """Save the model state_dict to safetensors format."""
        import torch
        from safetensors.torch import save_file

        model = source.model_object
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model).__name__}")

        max_shard_str = target.typed_options.get("max_shard_size", "5GB")
        max_shard_bytes = _parse_size_str(max_shard_str)

        state_dict = model.state_dict()
        total_params = sum(p.numel() for p in state_dict.values())
        # Estimate tensor bytes: default float32 → 4 bytes/param.
        total_bytes = sum(p.numel() * p.element_size() for p in state_dict.values())

        files: list[DraftFile] = []

        if max_shard_bytes == 0 or total_bytes <= max_shard_bytes:
            # Single-file export.
            output_path = context.artifact_dir / "model.safetensors"
            save_file(state_dict, str(output_path))
            files.append(DraftFile(relative_path="model.safetensors", role="model"))
        else:
            # Sharded export.
            shards = _shard_state_dict(state_dict, max_shard_bytes)
            index_data: dict[str, Any] = {
                "metadata": {"total_size": total_bytes},
                "weight_map": {},
            }
            for i, shard in enumerate(shards, 1):
                shard_name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
                shard_path = context.artifact_dir / shard_name
                save_file(shard, str(shard_path))
                files.append(DraftFile(relative_path=shard_name, role="model"))
                for key in shard:
                    index_data["weight_map"][key] = shard_name

            # Write index.
            index_path = context.artifact_dir / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index_data, indent=2))
            files.append(
                DraftFile(relative_path="model.safetensors.index.json", role="config")
            )

        # Save model config for reconstruction.
        if source.model_config_data:
            config_path = context.artifact_dir / "config.json"
            config_path.write_text(json.dumps(source.model_config_data, indent=2))
            files.append(DraftFile(relative_path="config.json", role="config"))

        logger.info(
            "Safetensors exported: %d params, %d files",
            total_params,
            len(files),
        )

        return ArtifactDraft(
            name=target.target.name,
            format="safetensors",
            flavor_id="safetensors-v1",
            variant=None,
            files=tuple(files),
            entrypoint=(
                "model.safetensors.index.json"
                if max_shard_bytes > 0 and total_bytes > max_shard_bytes
                else "model.safetensors"
            ),
            producer=ProducerInfo(
                exporter_id=self.exporter_id,
                framework_versions={
                    "torch": torch.__version__,
                    "safetensors": _get_safetensors_version(),
                },
                effective_options={
                    k: v
                    for k, v in target.typed_options.items()
                    if k != "max_shard_size"
                },
            ),
            derived_from=(),
        )


# ── Helpers ──────────────────────────────────────────────────────────────────────


def _parse_size_str(s: str) -> int:
    """Parse a human-readable size string to bytes (e.g. "5GB" → 5*1024^3)."""
    s = s.strip().upper()
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            num = float(s[: -len(suffix)])
            return int(num * mult)
    return int(s)


def _shard_state_dict(
    state_dict: dict[str, Any],
    max_bytes: int,
) -> list[dict[str, Any]]:
    """Shard a state_dict into chunks, each ≤ *max_bytes*."""
    shards: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    current_bytes = 0

    for key, tensor in state_dict.items():
        size = tensor.numel() * tensor.element_size()
        if current_bytes + size > max_bytes and current:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[key] = tensor
        current_bytes += size

    if current:
        shards.append(current)
    return shards


def _get_safetensors_version() -> str:
    try:
        import safetensors

        return getattr(safetensors, "__version__", "unknown")
    except ImportError:
        return "unknown"
