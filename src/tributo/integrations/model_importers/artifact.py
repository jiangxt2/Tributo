"""Explicit local/S3 model-artifact importer."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import ClassVar, Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tributo._common.storage import get_boto3_client, parse_s3_url
from tributo._common.storage_profiles import StorageProfileResolver
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
from tributo.integrations.model_importers._bundle import publish_model_artifact
from tributo.util.annotations import PublicAPI

_DEFAULT_MAX_BYTES = 5 * 1024 * 1024 * 1024


@PublicAPI(stability="alpha")
class ArtifactImportOptions(BaseModel):
    """Strict metadata required to turn one weight file into a Bundle."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    artifact_name: str = Field(
        default="imported-model", pattern=r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$"
    )
    variant: Literal["onnx", "ubj", "json"] | None = None
    input_fields: tuple[SignatureField, ...] = Field(min_length=1)
    output_fields: tuple[SignatureField, ...] = Field(min_length=1)
    max_bytes: int = Field(default=_DEFAULT_MAX_BYTES, ge=1, le=_DEFAULT_MAX_BYTES)

    @model_validator(mode="after")
    def _unique_signature_names(self) -> "ArtifactImportOptions":
        for side, fields in (
            ("input", self.input_fields),
            ("output", self.output_fields),
        ):
            names = [field.name for field in fields]
            if len(set(names)) != len(names):
                raise ValueError(f"{side} signature names must be unique")
        return self


@PublicAPI(stability="alpha")
class ArtifactModelImporter:
    """Acquire one explicitly typed artifact and publish a verified Bundle."""

    api_version: ClassVar[int] = 1
    provider_id: ClassVar[str] = "tributo.artifact"
    options_model: ClassVar[type[BaseModel]] = ArtifactImportOptions
    reference_kinds: ClassVar[tuple[str, ...]] = ("artifact",)
    uri_schemes: ClassVar[tuple[str, ...]] = ("file", "s3")
    credential_profile_types: ClassVar[tuple[str, ...]] = (
        "source-storage",
        "bundle-storage",
    )
    capabilities: ClassVar[tuple[str, ...]] = (
        "acquire",
        "content-digest",
        "bundle-import",
    )

    def import_model(
        self, reference: ArtifactModelReference | RegistryModelReference
    ) -> BundleRef:
        """Acquire, digest, validate, and normalize the artifact."""
        if not isinstance(reference, ArtifactModelReference):
            raise ValueError("tributo.artifact requires an artifact reference")
        options = ArtifactImportOptions.model_validate(reference.options)
        entrypoint, variant = _validate_format(reference, options)
        with tempfile.TemporaryDirectory(prefix="tributo-artifact-acquire-") as raw:
            acquired = Path(raw) / entrypoint
            _acquire(reference, acquired, max_bytes=options.max_bytes)
            acquired_size = acquired.stat().st_size
            if acquired_size > options.max_bytes:
                raise JobConfigurationError(
                    f"Acquired artifact size {acquired_size} exceeds configured "
                    f"limit {options.max_bytes}"
                )
            digest = _sha256(acquired)
            if reference.expected_sha256 is not None:
                if digest != reference.expected_sha256:
                    raise JobConfigurationError(
                        "External artifact digest mismatch: expected "
                        f"{reference.expected_sha256[:16]}..., got {digest[:16]}..."
                    )

            return publish_model_artifact(
                files={entrypoint: acquired},
                entrypoint=entrypoint,
                destination_uri=reference.import_bundle_uri,
                destination_storage_profile=reference.import_storage_profile,
                artifact_name=options.artifact_name,
                format_id=reference.format_id,
                flavor_id=reference.flavor_id,
                variant=variant,
                producer_id=self.provider_id,
                source_info=ManifestSourceInfo(
                    source_kind="external-artifact",
                    source_fingerprint=digest,
                    framework=(
                        "xgboost" if reference.format_id == "xgboost" else "onnx"
                    ),
                    architecture_id=reference.architecture_id,
                ),
                input_signature=ManifestSignature(input_fields=options.input_fields),
                output_signature=ManifestSignature(output_fields=options.output_fields),
            )


def _validate_format(
    reference: ArtifactModelReference, options: ArtifactImportOptions
) -> tuple[str, str | None]:
    if reference.format_id == "safetensors" and reference.architecture_id is None:
        raise JobConfigurationError(
            "Safetensors is weights-only; architecture_id and a trusted "
            "ModelFactory are required before it can become an inference Bundle"
        )
    pair = (reference.format_id, reference.flavor_id)
    if pair == ("onnx", "onnx-runtime-v1"):
        if options.variant not in (None, "onnx"):
            raise JobConfigurationError("ONNX artifacts require variant='onnx'")
        return "model.onnx", "onnx"
    if pair == ("xgboost", "xgboost-native-v1"):
        variant = options.variant or "ubj"
        if variant not in {"ubj", "json"}:
            raise JobConfigurationError(
                "XGBoost artifacts require variant='ubj' or variant='json'"
            )
        input_names = tuple(field.name for field in options.input_fields)
        if input_names != ("float_input",):
            raise JobConfigurationError(
                "xgboost-native-v1 requires exactly one input signature field "
                "named 'float_input'"
            )
        return f"model.{variant}", variant
    raise UnsupportedArtifactFormat(
        f"Artifact format/flavor pair {pair!r} is not supported by "
        f"{ArtifactModelImporter.provider_id!r}"
    )


def _acquire(
    reference: ArtifactModelReference, destination: Path, *, max_bytes: int
) -> None:
    parsed = urlsplit(reference.uri)
    if parsed.username is not None or parsed.password is not None:
        raise JobConfigurationError("Artifact URI must not contain credentials")
    if parsed.query or parsed.fragment:
        raise JobConfigurationError("Artifact URI must not contain query or fragment")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if parsed.scheme.lower() == "s3":
        bucket, key = parse_s3_url(reference.uri)
        profile = StorageProfileResolver().resolve(reference.storage_profile)
        client = get_boto3_client(
            endpoint=profile.endpoint,
            access_key_id=profile.access_key_id,
            secret_access_key=profile.secret_access_key,
            region=profile.region,
            use_ssl=profile.use_ssl,
            path_style=profile.path_style,
            profile_name=profile.profile_name,
        )
        size = int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
        if size > max_bytes:
            raise JobConfigurationError(
                f"Artifact size {size} exceeds configured limit {max_bytes}"
            )
        client.download_file(bucket, key, str(destination))
        return

    if parsed.scheme.lower() not in {"", "file"}:
        raise JobConfigurationError(
            f"Unsupported artifact URI scheme {parsed.scheme.lower()!r}"
        )
    if parsed.scheme.lower() == "file" and parsed.netloc not in {"", "localhost"}:
        raise JobConfigurationError("file URI must not name a remote host")
    source = Path(unquote(parsed.path) if parsed.scheme else reference.uri).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"External artifact is not a regular file: {source}")
    size = source.stat().st_size
    if size > max_bytes:
        raise JobConfigurationError(
            f"Artifact size {size} exceeds configured limit {max_bytes}"
        )
    shutil.copyfile(source, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["ArtifactImportOptions", "ArtifactModelImporter"]
