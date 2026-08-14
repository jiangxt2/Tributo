"""Parquet implementation of the Inference ResultSink contract."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, ClassVar, Protocol
from urllib.parse import unquote, urlsplit

from tributo._common.storage_profiles import StorageProfile, StorageProfileResolver
from tributo.data.base import S3Config, WriteMode
from tributo.data.contracts.handles import RayDataHandle
from tributo.data.writing.builtins import default_write_gateway
from tributo.data.writing.contracts import WriteRequest
from tributo.exceptions import ResultMaterializationError, ResultWriteError
from tributo.inference._credential_safety import safe_exception_summary
from tributo.inference.contracts import (
    ParquetResultSinkRequest,
    ResultSinkReceipt,
    ResultSinkRequest,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


class _StorageProfileResolverLike(Protocol):
    def resolve(self, profile: str | None) -> StorageProfile: ...


@PublicAPI(stability="alpha")
class ParquetResultSink:
    """Write a Ray Dataset through the shared native write Gateway.

    Ray's public API does not return ``WriteResult.num_rows``.  P0 therefore
    returns ``rows_written=None`` and never launches a second ``count()`` job
    merely to populate observability metadata.
    """

    api_version: ClassVar[int] = 1
    sink_id: ClassVar[str] = "parquet-v1"

    def __init__(
        self, storage_resolver: _StorageProfileResolverLike | None = None
    ) -> None:
        self._storage_resolver = storage_resolver or StorageProfileResolver()

    def write(
        self,
        dataset: Any,
        request: ResultSinkRequest,
        *,
        run_id: str,
        plan_digest: str,
    ) -> ResultSinkReceipt:
        """Stream predictions to Parquet and return a credential-free receipt."""
        if not isinstance(request, ParquetResultSinkRequest):
            raise ResultWriteError(
                f"Parquet result sink cannot write {request.sink_id!r}"
            )
        output_path = request.uri
        output_filesystem: Any | None = None
        parsed = urlsplit(request.uri)
        scheme = parsed.scheme.lower()
        runtime_options: dict[str, Any] = {}
        if scheme == "s3":
            profile = self._storage_resolver.resolve(request.storage_profile)
            if profile.profile_name is not None:
                raise ResultWriteError(
                    "Ray result-sink profiles must resolve inside the cluster; set "
                    f"TRIBUTO_STORAGE_PROFILE_{profile.profile_name.upper()} or "
                    "use the default IAM/environment chain"
                )
            endpoint = profile.endpoint
            if (
                endpoint is not None
                and not profile.use_ssl
                and endpoint.startswith("https://")
            ):
                endpoint = "http://" + endpoint.removeprefix("https://")
            s3_config = S3Config(
                endpoint=endpoint,
                region=profile.region,
                access_key_id=profile.access_key_id,
                secret_access_key=profile.secret_access_key,
            )
            runtime_options["s3"] = s3_config
            if request.max_bytes is not None:
                import pyarrow.fs as pafs

                from tributo.data._s3 import to_pyarrow_s3_kwargs

                output_filesystem = pafs.S3FileSystem(**to_pyarrow_s3_kwargs(s3_config))
            output_path = f"{parsed.netloc}{parsed.path}"
        elif scheme == "file":
            output_path = unquote(parsed.path)

        try:
            options: dict[str, Any] = {"compression": request.compression}
            if request.min_rows_per_file is not None:
                options["min_rows_per_file"] = request.min_rows_per_file
            default_write_gateway().execute(
                WriteRequest(
                    engine="ray",
                    target_kind="parquet",
                    target=request.uri,
                    mode=WriteMode.APPEND,
                    options=options,
                    runtime_options=runtime_options,
                ),
                RayDataHandle(dataset),
            )
        except Exception as exc:
            source_error_type = getattr(exc, "source_error_type", None)
            logger.warning(
                "Parquet result materialization failed (%s): %s",
                source_error_type or type(exc).__name__,
                safe_exception_summary(exc),
            )
            raise ResultMaterializationError(
                source_error_type or type(exc).__name__
            ) from None

        if request.max_bytes is not None:
            actual_bytes = _output_bytes(output_path, output_filesystem)
            if actual_bytes > request.max_bytes:
                raise ResultWriteError(
                    "Parquet result exceeds the configured max_bytes limit"
                )

        result_id = _result_id(
            run_id=run_id,
            plan_digest=plan_digest,
            uri=request.uri,
            compression=request.compression,
        )
        return ResultSinkReceipt(
            sink_id=self.sink_id,
            result_id=result_id,
            uri=request.uri,
            rows_written=None,
            metadata={
                "format": "parquet",
                "compression": request.compression,
            },
        )


def _result_id(*, run_id: str, plan_digest: str, uri: str, compression: str) -> str:
    payload = json.dumps(
        {
            "run_id": run_id,
            "plan_digest": plan_digest,
            "sink_id": ParquetResultSink.sink_id,
            "uri": uri,
            "compression": compression,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _output_bytes(output_path: str, filesystem: Any | None) -> int:
    """Return materialized output bytes without reading result contents."""
    if filesystem is not None:
        import pyarrow.fs as pafs

        infos = filesystem.get_file_info(pafs.FileSelector(output_path, recursive=True))
        return sum(info.size for info in infos if info.is_file)
    from pathlib import Path

    path = Path(output_path)
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


__all__ = ["ParquetResultSink"]
