"""MLflow Registry importer that freezes versions before Bundle execution."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, TypeVar, cast
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from tributo.exceptions import JobConfigurationError, UnsupportedArtifactFormat
from tributo.exporting.manifest import (
    ManifestSignature,
    ManifestSourceInfo,
    SignatureField,
)
from tributo.exporting.models import BundleRef
from tributo.inference._credential_safety import safe_exception_summary
from tributo.inference.contracts import (
    ArtifactModelReference,
    RegistryModelReference,
)
from tributo.integrations.model_importers._bundle import (
    publish_model_artifact,
    republish_verified_bundle,
)
from tributo.util.annotations import PublicAPI

_T = TypeVar("_T")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
logger = logging.getLogger(__name__)


@PublicAPI(stability="alpha")
class MLflowImportOptions(BaseModel):
    """Strict MLflow connection and acquisition limits."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    tracking_uri: str | None = None
    registry_uri: str | None = None
    max_files: int = Field(default=256, ge=1, le=4096)
    max_depth: int = Field(default=16, ge=1, le=64)
    max_total_bytes: int = Field(
        default=5 * 1024 * 1024 * 1024,
        ge=1,
        le=50 * 1024 * 1024 * 1024,
    )


@PublicAPI(stability="alpha")
class MLflowModelImporter:
    """Resolve an MLflow Alias/version once and normalize its model files."""

    api_version: ClassVar[int] = 1
    provider_id: ClassVar[str] = "mlflow.v2"
    options_model: ClassVar[type[BaseModel]] = MLflowImportOptions
    reference_kinds: ClassVar[tuple[str, ...]] = ("registry",)
    uri_schemes: ClassVar[tuple[str, ...]] = ()
    credential_profile_types: ClassVar[tuple[str, ...]] = (
        "mlflow-environment",
        "bundle-storage",
    )
    capabilities: ClassVar[tuple[str, ...]] = (
        "immutable-version-resolution",
        "artifact-acquisition",
        "bundle-import",
    )

    def import_model(
        self, reference: ArtifactModelReference | RegistryModelReference
    ) -> BundleRef:
        """Resolve to a numeric version, download once, and publish a Bundle."""
        if not isinstance(reference, RegistryModelReference):
            raise ValueError("mlflow.v2 requires a registry reference")
        if reference.storage_profile is not None:
            raise JobConfigurationError(
                "mlflow.v2 does not resolve RegistryModelReference.storage_profile; "
                "configure MLflow authentication through its environment chain"
            )
        options = MLflowImportOptions.model_validate(reference.options)
        _validate_connection_uri(options.tracking_uri, "tracking_uri")
        _validate_connection_uri(options.registry_uri, "registry_uri")
        try:
            from mlflow import MlflowClient
            from mlflow.models import Model
        except ImportError as exc:
            raise RuntimeError(
                "MLflow model import requires the 'registry' extra"
            ) from exc

        client = MlflowClient(
            tracking_uri=options.tracking_uri,
            registry_uri=options.registry_uri,
        )
        version = reference.version
        if version is not None:
            resolved = _sdk_call(
                "numeric version resolution",
                lambda: client.get_model_version(reference.model_name, version),
            )
        else:
            alias = reference.alias
            assert alias is not None
            aliased = _sdk_call(
                "Alias resolution",
                lambda: client.get_model_version_by_alias(reference.model_name, alias),
            )
            resolved = _sdk_call(
                "resolved version pinning",
                lambda: client.get_model_version(
                    reference.model_name, str(aliased.version)
                ),
            )
        resolved_version = str(resolved.version)
        source_uri = str(resolved.source)
        run_id = str(resolved.run_id or "")
        source_run_id, artifact_path = _parse_runs_source(source_uri, run_id)
        provenance = _source_fingerprint(
            model_name=reference.model_name,
            version=resolved_version,
            run_id=source_run_id,
            source_uri=source_uri,
        )

        with tempfile.TemporaryDirectory(prefix="tributo-mlflow-acquire-") as raw:
            staging = Path(raw)
            expected_files = _preflight_run_artifact(
                client,
                run_id=source_run_id,
                artifact_path=artifact_path,
                max_files=options.max_files,
                max_depth=options.max_depth,
                max_total_bytes=options.max_total_bytes,
            )
            downloaded = Path(
                _sdk_call(
                    "artifact download",
                    lambda: client.download_artifacts(
                        source_run_id,
                        artifact_path,
                        dst_path=str(staging),
                    ),
                )
            )
            downloaded = _validate_staged_root(staging, downloaded)
            if not downloaded.is_dir():
                raise JobConfigurationError(
                    "MLflow model version source must resolve to a model directory"
                )
            actual_files = _validate_download_tree(
                downloaded,
                max_files=options.max_files,
                max_total_bytes=options.max_total_bytes,
            )
            if actual_files != expected_files:
                raise JobConfigurationError(
                    "MLflow artifact tree changed between bounded preflight "
                    "and download"
                )
            manifest_path = downloaded / "manifest.json"
            if manifest_path.is_file():
                from tributo.exporting.bundle_reader import BundleReader

                original = BundleReader().read_manifest(str(downloaded))
                source_info = ManifestSourceInfo(
                    source_kind="mlflow-registry-bundle",
                    source_fingerprint=provenance,
                    framework=original.source_info.framework,
                    framework_version=original.source_info.framework_version,
                    architecture_id=original.source_info.architecture_id,
                    task_type=original.source_info.task_type,
                )
                return republish_verified_bundle(
                    source_bundle_uri=str(downloaded),
                    destination_uri=reference.import_bundle_uri,
                    destination_storage_profile=reference.import_storage_profile,
                    source_info=source_info,
                )

            mlmodel_path = downloaded / "MLmodel"
            if not mlmodel_path.is_file():
                raise UnsupportedArtifactFormat(
                    "MLflow model must contain either a Tributo manifest.json "
                    "or an MLmodel file with a supported flavor"
                )
            metadata = _sdk_call(
                "model metadata parsing",
                lambda: Model.load(str(mlmodel_path)),
            )
            flavor = metadata.flavors.get("onnx")
            if not isinstance(flavor, dict):
                raise UnsupportedArtifactFormat(
                    "MLflow model flavor set is unsupported; only an existing "
                    "Tributo Bundle or the ONNX flavor is accepted"
                )
            data = flavor.get("data")
            if not isinstance(data, str) or not data:
                raise JobConfigurationError("MLflow ONNX flavor has no data path")
            entrypoint = _safe_relative_path(downloaded, data)
            if not entrypoint.is_file():
                raise JobConfigurationError("MLflow ONNX data file is missing")
            input_signature, output_signature = _mlflow_signatures(metadata.signature)
            files = {
                str(path.relative_to(downloaded)): path
                for path in downloaded.rglob("*")
                if path.is_file()
            }
            return publish_model_artifact(
                files=files,
                entrypoint=str(entrypoint.relative_to(downloaded)),
                destination_uri=reference.import_bundle_uri,
                destination_storage_profile=reference.import_storage_profile,
                artifact_name="mlflow-onnx-model",
                format_id="onnx",
                flavor_id="onnx-runtime-v1",
                variant="onnx",
                producer_id=self.provider_id,
                source_info=ManifestSourceInfo(
                    source_kind="mlflow-registry-model",
                    source_fingerprint=provenance,
                    framework="onnx",
                    framework_version=str(flavor.get("onnx_version") or "") or None,
                ),
                input_signature=input_signature,
                output_signature=output_signature,
            )


def _sdk_call(operation: str, call: Callable[[], _T]) -> _T:
    """Run one MLflow SDK call without propagating secret-bearing details."""
    try:
        return call()
    except Exception as exc:
        logger.warning(
            "MLflow %s failed (%s): %s",
            operation,
            type(exc).__name__,
            safe_exception_summary(exc),
        )
        raise JobConfigurationError(
            f"MLflow {operation} failed ({type(exc).__name__})"
        ) from None


def _parse_runs_source(source_uri: str, resolved_run_id: str) -> tuple[str, str]:
    """Accept only a bounded public ``runs:/`` model-version source."""
    parsed = urlsplit(source_uri)
    if parsed.username is not None or parsed.password is not None:
        raise JobConfigurationError("MLflow model source must not contain credentials")
    if parsed.query or parsed.fragment:
        raise JobConfigurationError(
            "MLflow model source must not contain query or fragment"
        )
    if parsed.scheme.lower() != "runs" or parsed.netloc:
        raise JobConfigurationError(
            "MLflow model version source must use the runs:/ scheme"
        )
    parts = parsed.path.removeprefix("/").split("/")
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        raise JobConfigurationError("MLflow runs:/ source path is invalid")
    run_id, artifact_parts = parts[0], parts[1:]
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise JobConfigurationError("MLflow runs:/ source has an invalid run id")
    if not resolved_run_id or resolved_run_id != run_id:
        raise JobConfigurationError(
            "MLflow model-version run id does not match its runs:/ source"
        )
    artifact_path = "/".join(artifact_parts)
    _validate_posix_artifact_path(artifact_path, max_depth=64)
    return run_id, artifact_path


def _source_fingerprint(
    *, model_name: str, version: str, run_id: str, source_uri: str
) -> str:
    payload = {
        "provider_id": MLflowModelImporter.provider_id,
        "model_name_sha256": hashlib.sha256(model_name.encode("utf-8")).hexdigest(),
        "version": version,
        "run_id": run_id,
        "source_uri_sha256": hashlib.sha256(source_uri.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _preflight_run_artifact(
    client: Any,
    *,
    run_id: str,
    artifact_path: str,
    max_files: int,
    max_depth: int,
    max_total_bytes: int,
) -> dict[str, int]:
    """List and bound the complete MLflow artifact tree before download."""
    root = PurePosixPath(artifact_path)
    pending = [artifact_path]
    visited_directories: set[str] = set()
    files: dict[str, int] = {}
    entry_count = 0
    total_bytes = 0

    while pending:
        current = pending.pop()
        if current in visited_directories:
            continue
        visited_directories.add(current)

        entries = _list_artifacts(client, run_id=run_id, path=current)
        for entry in entries:
            entry_count += 1
            if entry_count > max_files:
                raise JobConfigurationError(
                    f"MLflow artifact tree exceeds entry limit {max_files}"
                )
            path = str(entry.path)
            _validate_posix_artifact_path(path, max_depth=64)
            candidate = PurePosixPath(path)
            try:
                relative = candidate.relative_to(root)
            except ValueError:
                raise JobConfigurationError(
                    "MLflow artifact listing escaped the requested model root"
                ) from None
            if not relative.parts:
                raise JobConfigurationError(
                    "MLflow artifact listing repeated the requested model root"
                )
            if len(relative.parts) > max_depth:
                raise JobConfigurationError(
                    f"MLflow artifact path exceeds depth limit {max_depth}"
                )
            key = relative.as_posix()
            if bool(entry.is_dir):
                pending.append(path)
                continue
            size = entry.file_size
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise JobConfigurationError(
                    "MLflow artifact listing omitted a valid file size"
                )
            if key in files:
                raise JobConfigurationError(
                    "MLflow artifact listing contains duplicate paths"
                )
            total_bytes += size
            if total_bytes > max_total_bytes:
                raise JobConfigurationError(
                    f"MLflow model size exceeds limit {max_total_bytes}"
                )
            files[key] = size

    if not files:
        raise JobConfigurationError("MLflow model artifact directory is empty")
    return files


def _list_artifacts(client: Any, *, run_id: str, path: str) -> list[Any]:
    def _list() -> list[Any]:
        return cast(list[Any], client.list_artifacts(run_id, path))

    return _sdk_call("artifact listing", _list)


def _validate_posix_artifact_path(value: str, *, max_depth: int) -> None:
    if not value or len(value.encode("utf-8")) > 1024:
        raise JobConfigurationError("MLflow artifact path length is invalid")
    if value.startswith("/") or "\\" in value:
        raise JobConfigurationError("MLflow artifact path must be relative POSIX")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise JobConfigurationError("MLflow artifact path contains unsafe segments")
    if len(raw_parts) > max_depth:
        raise JobConfigurationError(
            f"MLflow artifact path exceeds depth limit {max_depth}"
        )


def _validate_staged_root(staging: Path, downloaded: Path) -> Path:
    staging_absolute = Path(os.path.abspath(staging))
    downloaded_absolute = Path(os.path.abspath(downloaded))
    if not downloaded_absolute.is_relative_to(staging_absolute):
        raise JobConfigurationError("MLflow download escaped its staging directory")
    current = downloaded_absolute
    while current != staging_absolute:
        if current.is_symlink():
            raise JobConfigurationError("MLflow download path must not use symlinks")
        current = current.parent
    if not downloaded_absolute.exists():
        raise JobConfigurationError("MLflow download did not produce an artifact")
    return downloaded_absolute


def _validate_download_tree(
    root: Path, *, max_files: int, max_total_bytes: int
) -> dict[str, int]:
    if root.is_symlink():
        raise JobConfigurationError("MLflow download root must not be a symlink")
    paths = [root] if root.is_file() else list(root.rglob("*"))
    regular_files: dict[str, int] = {}
    total_bytes = 0
    entry_count = 0
    for path in paths:
        entry_count += 1
        if entry_count > max_files:
            raise JobConfigurationError(
                f"MLflow artifact tree exceeds entry limit {max_files}"
            )
        if path.is_symlink():
            raise JobConfigurationError(
                "MLflow model directory must not contain symlinks"
            )
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            _validate_posix_artifact_path(relative, max_depth=64)
            size = path.stat().st_size
            regular_files[relative] = size
            total_bytes += size
        elif not path.is_dir():
            raise JobConfigurationError(
                "MLflow model directory contains a non-regular filesystem entry"
            )
    if total_bytes > max_total_bytes:
        raise JobConfigurationError(
            f"MLflow model size {total_bytes} exceeds limit {max_total_bytes}"
        )
    return regular_files


def _validate_connection_uri(uri: str | None, field: str) -> None:
    if uri is None:
        return
    parsed = urlsplit(uri)
    if parsed.username is not None or parsed.password is not None:
        raise JobConfigurationError(
            f"MLflow {field} must not contain plaintext credentials"
        )
    sensitive_query_keys = {
        "accesskeyid",
        "accesstoken",
        "apikey",
        "apitoken",
        "authorization",
        "authtoken",
        "clientsecret",
        "oauthtoken",
        "password",
        "refreshtoken",
        "secretaccesskey",
        "sessiontoken",
        "token",
        "xamzcredential",
        "xamzsignature",
    }
    if any(
        "".join(character for character in key.lower() if character.isalnum())
        in sensitive_query_keys
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
    ):
        raise JobConfigurationError(
            f"MLflow {field} must not contain credential query parameters"
        )


def _safe_relative_path(root: Path, relative: str) -> Path:
    _validate_posix_artifact_path(relative, max_depth=64)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    if candidate.is_symlink():
        raise JobConfigurationError("MLflow flavor path must not be a symlink")
    return candidate


def _mlflow_signatures(signature: Any) -> tuple[ManifestSignature, ManifestSignature]:
    if signature is None or signature.inputs is None or signature.outputs is None:
        raise UnsupportedArtifactFormat(
            "MLflow ONNX import requires a typed input and output signature"
        )
    inputs = tuple(
        _mlflow_field(spec, index) for index, spec in enumerate(signature.inputs)
    )
    outputs = tuple(
        _mlflow_field(spec, index) for index, spec in enumerate(signature.outputs)
    )
    return (
        ManifestSignature(input_fields=inputs),
        ManifestSignature(output_fields=outputs),
    )


def _mlflow_field(spec: Any, index: int) -> SignatureField:
    name = getattr(spec, "name", None)
    shape = getattr(spec, "shape", None)
    dtype = getattr(spec, "type", None)
    if not isinstance(name, str) or not name:
        raise UnsupportedArtifactFormat(
            "MLflow ONNX signatures must use named tensor fields"
        )
    if shape is None or dtype is None:
        raise UnsupportedArtifactFormat(
            "MLflow ONNX signatures must use TensorSpec fields"
        )
    canonical_shape = tuple(
        (
            dimension
            if isinstance(dimension, int) and dimension > 0
            else f"dynamic_{index}_{axis}"
        )
        for axis, dimension in enumerate(shape)
    )
    dtype_name = str(getattr(dtype, "name", dtype))
    return SignatureField(name=name, dtype=dtype_name, shape=canonical_shape)


__all__ = ["MLflowImportOptions", "MLflowModelImporter"]
