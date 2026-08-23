"""Credential-safe local and S3 object operations for data persistence bindings."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable
from urllib.parse import unquote, urlsplit

from tributo._common.storage import get_boto3_client, parse_s3_url
from tributo._common.storage_profiles import StorageProfileResolver
from tributo.util.annotations import DeveloperAPI


@DeveloperAPI
@dataclass(frozen=True)
class ObjectFile:
    """One file exposed by an object-store prefix."""

    uri: str
    relative_path: str
    size: int


@runtime_checkable
@DeveloperAPI
class ObjectStore(Protocol):
    """Minimal physical object-storage boundary for data-domain adapters."""

    store_id: ClassVar[str]

    def read_bytes(self, uri: str, *, storage_profile: str | None = None) -> bytes: ...

    def write_bytes(
        self,
        uri: str,
        data: bytes,
        *,
        storage_profile: str | None = None,
        exclusive: bool = False,
        content_type: str | None = None,
    ) -> None: ...

    def list_files(
        self,
        uri: str,
        *,
        storage_profile: str | None = None,
    ) -> tuple[ObjectFile, ...]: ...

    def delete_tree(self, uri: str, *, storage_profile: str | None = None) -> None: ...


@DeveloperAPI
class LocalS3ObjectStore:
    """Default object store backed by local paths and public S3 APIs."""

    store_id: ClassVar[str] = "local-s3-object-v1"

    def __init__(self, profile_resolver: StorageProfileResolver | None = None) -> None:
        self._profile_resolver = profile_resolver or StorageProfileResolver()

    def read_bytes(self, uri: str, *, storage_profile: str | None = None) -> bytes:
        parsed = urlsplit(uri)
        if parsed.scheme.lower() == "s3":
            bucket, key = parse_s3_url(uri)
            response = self._client(storage_profile).get_object(
                Bucket=bucket,
                Key=key,
            )
            return bytes(response["Body"].read())
        return self._local_path(uri).read_bytes()

    def write_bytes(
        self,
        uri: str,
        data: bytes,
        *,
        storage_profile: str | None = None,
        exclusive: bool = False,
        content_type: str | None = None,
    ) -> None:
        parsed = urlsplit(uri)
        if parsed.scheme.lower() == "s3":
            bucket, key = parse_s3_url(uri)
            kwargs: dict[str, Any] = {
                "Bucket": bucket,
                "Key": key,
                "Body": data,
            }
            if exclusive:
                kwargs["IfNoneMatch"] = "*"
            if content_type is not None:
                kwargs["ContentType"] = content_type
            try:
                self._client(storage_profile).put_object(**kwargs)
            except Exception as exc:
                if exclusive and _s3_error_code(exc) in {
                    "409",
                    "412",
                    "ConditionalRequestConflict",
                    "PreconditionFailed",
                }:
                    raise FileExistsError(uri) from None
                raise
            return

        path = self._local_path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        if exclusive:
            with path.open("xb") as output:
                output.write(data)
        else:
            path.write_bytes(data)

    def list_files(
        self,
        uri: str,
        *,
        storage_profile: str | None = None,
    ) -> tuple[ObjectFile, ...]:
        parsed = urlsplit(uri)
        if parsed.scheme.lower() == "s3":
            bucket, key = parse_s3_url(uri)
            prefix = key.rstrip("/") + "/" if key else ""
            paginator = self._client(storage_profile).get_paginator("list_objects_v2")
            files: list[ObjectFile] = []
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get("Contents", ()):
                    object_key = str(item["Key"])
                    if object_key.endswith("/"):
                        continue
                    relative = (
                        object_key[len(prefix) :]
                        if prefix and object_key.startswith(prefix)
                        else object_key
                    )
                    files.append(
                        ObjectFile(
                            uri=f"s3://{bucket}/{object_key}",
                            relative_path=relative,
                            size=int(item.get("Size", 0)),
                        )
                    )
            return tuple(sorted(files, key=lambda item: item.relative_path))

        path = self._local_path(uri)
        if not path.exists():
            return ()
        if path.is_file():
            return (ObjectFile(str(path), path.name, path.stat().st_size),)
        return tuple(
            ObjectFile(
                uri=str(file_path),
                relative_path=file_path.relative_to(path).as_posix(),
                size=file_path.stat().st_size,
            )
            for file_path in sorted(item for item in path.rglob("*") if item.is_file())
        )

    def delete_tree(self, uri: str, *, storage_profile: str | None = None) -> None:
        parsed = urlsplit(uri)
        if parsed.scheme.lower() == "s3":
            files = self.list_files(uri, storage_profile=storage_profile)
            if not files:
                return
            bucket, _ = parse_s3_url(uri)
            client = self._client(storage_profile)
            for start in range(0, len(files), 1000):
                client.delete_objects(
                    Bucket=bucket,
                    Delete={
                        "Objects": [
                            {"Key": parse_s3_url(file.uri)[1]}
                            for file in files[start : start + 1000]
                        ],
                        "Quiet": True,
                    },
                )
            return

        path = self._local_path(uri)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def _client(self, storage_profile: str | None) -> Any:
        profile = self._profile_resolver.resolve(storage_profile)
        return get_boto3_client(
            endpoint=profile.endpoint,
            access_key_id=profile.access_key_id,
            secret_access_key=profile.secret_access_key,
            region=profile.region,
            use_ssl=profile.use_ssl,
            path_style=profile.path_style,
            profile_name=profile.profile_name,
        )

    @staticmethod
    def _local_path(uri: str) -> Path:
        parsed = urlsplit(uri)
        if parsed.scheme and parsed.scheme.lower() not in {"file", "s3"}:
            raise ValueError(f"unsupported object-store URI scheme: {parsed.scheme!r}")
        if parsed.scheme.lower() == "file" and parsed.netloc not in {"", "localhost"}:
            raise ValueError("file URI must not name a remote host")
        return Path(unquote(parsed.path if parsed.scheme else uri))


def _s3_error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error")
    if not isinstance(error, dict):
        return None
    code = error.get("Code")
    return str(code) if code is not None else None


_DEFAULT_OBJECT_STORE: ObjectStore = LocalS3ObjectStore()


@DeveloperAPI
def default_object_store() -> ObjectStore:
    """Return the process-local local/S3 object-store adapter."""
    return _DEFAULT_OBJECT_STORE


__all__ = [
    "LocalS3ObjectStore",
    "ObjectFile",
    "ObjectStore",
    "default_object_store",
]
