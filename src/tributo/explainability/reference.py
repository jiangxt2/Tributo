"""Reference/background data providers for explainability adapters."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlsplit

import numpy as np

from tributo.explainability.contracts import ExplainabilityLimits, ReferenceBinding
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ResolvedReference:
    """Materialized reference data and its immutable provenance."""

    data: np.ndarray
    digest: str
    rows: int


@PublicAPI(stability="alpha")
class ReferenceProvider(Protocol):
    """Load and identify reference data without exposing storage details."""

    provider_id: str

    def resolve(
        self,
        binding: ReferenceBinding,
        limits: ExplainabilityLimits,
    ) -> ResolvedReference: ...

    def digest(self, binding: ReferenceBinding) -> str: ...


@PublicAPI(stability="alpha")
class FileReferenceProvider:
    """Built-in local/file:// and S3 NPY/Parquet reference provider."""

    provider_id = "file-reference-v1"

    def resolve(
        self,
        binding: ReferenceBinding,
        limits: ExplainabilityLimits,
    ) -> ResolvedReference:
        payload, suffix, digest = self._read(binding)
        if suffix == ".npy":
            data = np.load(io.BytesIO(payload), allow_pickle=False)
        else:
            import pyarrow as pa
            import pyarrow.parquet as pq

            data = pq.read_table(pa.BufferReader(payload)).to_pandas().to_numpy()
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError(
                "reference data must be a non-empty two-dimensional matrix"
            )
        if binding.rows is not None and data.shape[0] != binding.rows:
            raise ValueError(
                f"reference rows {data.shape[0]} do not match declared rows "
                f"{binding.rows}"
            )
        if binding.digest is not None and digest != binding.digest:
            raise ValueError("reference digest does not match the bound artifact")
        if limits.max_background_rows is not None:
            data = data[: limits.max_background_rows]
        return ResolvedReference(data=data, digest=digest, rows=int(data.shape[0]))

    def digest(self, binding: ReferenceBinding) -> str:
        return self._read(binding)[2]

    @staticmethod
    def _read(binding: ReferenceBinding) -> tuple[bytes, str, str]:
        parsed = urlsplit(binding.uri)
        if parsed.scheme == "s3":
            import pyarrow.fs as pafs

            filesystem, path = pafs.FileSystem.from_uri(binding.uri)
            with filesystem.open_input_file(path) as stream:
                payload = stream.read()
            suffix = Path(path).suffix
        else:
            path = Path(
                unquote(parsed.path if parsed.scheme == "file" else binding.uri)
            )
            payload = path.read_bytes()
            suffix = path.suffix
        return payload, suffix, hashlib.sha256(payload).hexdigest()


__all__ = ["FileReferenceProvider", "ReferenceProvider", "ResolvedReference"]
