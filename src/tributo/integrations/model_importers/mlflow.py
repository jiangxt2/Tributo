"""MLflow Registry importer that freezes versions before Bundle execution."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from tributo.exceptions import JobConfigurationError, UnsupportedArtifactFormat
from tributo.exporting.manifest import (
    ManifestSignature,
    ManifestSourceInfo,
    SignatureField,
)
from tributo.exporting.models import BundleRef
from tributo.inference.contracts import (
    ArtifactModelReference,
    RegistryModelReference,
)
from tributo.integrations.model_importers._bundle import (
    publish_model_artifact,
    republish_verified_bundle,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class MLflowImportOptions(BaseModel):
    """Strict MLflow connection and acquisition limits."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    tracking_uri: str | None = None
    registry_uri: str | None = None
    max_files: int = Field(default=256, ge=1, le=4096)
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
            import mlflow
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
        if reference.version is not None:
            resolved = client.get_model_version(reference.model_name, reference.version)
        else:
            assert reference.alias is not None
            aliased = client.get_model_version_by_alias(
                reference.model_name, reference.alias
            )
            resolved = client.get_model_version(
                reference.model_name, str(aliased.version)
            )
        resolved_version = str(resolved.version)
        source_uri = str(resolved.source)
        run_id = str(resolved.run_id or "")
        source_digest = hashlib.sha256(source_uri.encode("utf-8")).hexdigest()
        provenance = (
            f"model={reference.model_name};version={resolved_version};"
            f"run={run_id};source_sha256={source_digest}"
        )

        with tempfile.TemporaryDirectory(prefix="tributo-mlflow-acquire-") as raw:
            downloaded = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=source_uri,
                    dst_path=raw,
                    tracking_uri=options.tracking_uri,
                )
            ).resolve()
            _validate_download_tree(
                downloaded,
                max_files=options.max_files,
                max_total_bytes=options.max_total_bytes,
            )
            if not downloaded.is_dir():
                raise JobConfigurationError(
                    "MLflow model version source must resolve to a model directory"
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
            metadata = Model.load(str(mlmodel_path))
            flavor = metadata.flavors.get("onnx")
            if not isinstance(flavor, dict):
                raise UnsupportedArtifactFormat(
                    f"MLflow flavors {sorted(metadata.flavors)} are not supported; "
                    "only an existing Tributo Bundle or the ONNX flavor is accepted"
                )
            data = flavor.get("data")
            if not isinstance(data, str) or not data:
                raise JobConfigurationError("MLflow ONNX flavor has no data path")
            entrypoint = _safe_relative_path(downloaded, data)
            if not entrypoint.is_file():
                raise FileNotFoundError(f"MLflow ONNX data file is missing: {data!r}")
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


def _validate_download_tree(
    root: Path, *, max_files: int, max_total_bytes: int
) -> None:
    if root.is_symlink():
        raise JobConfigurationError("MLflow download root must not be a symlink")
    files = [root] if root.is_file() else list(root.rglob("*"))
    regular_files = []
    total_bytes = 0
    for path in files:
        if path.is_symlink():
            raise JobConfigurationError(
                "MLflow model directory must not contain symlinks"
            )
        if path.is_file():
            regular_files.append(path)
            total_bytes += path.stat().st_size
    if len(regular_files) > max_files:
        raise JobConfigurationError(
            f"MLflow model has {len(regular_files)} files; limit is {max_files}"
        )
    if total_bytes > max_total_bytes:
        raise JobConfigurationError(
            f"MLflow model size {total_bytes} exceeds limit {max_total_bytes}"
        )


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
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise JobConfigurationError(
            f"MLflow flavor path escapes model root: {relative!r}"
        )
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
