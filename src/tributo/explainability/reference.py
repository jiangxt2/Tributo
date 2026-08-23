"""Reference/background data providers for explainability adapters."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit

import numpy as np

from tributo.data.persistence import ObjectStore, default_object_store
from tributo.explainability.contracts import ExplainabilityLimits, ReferenceBinding
from tributo.explainability.protocols import ReferenceProvider, ResolvedReference
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class FileReferenceProvider:
    """Built-in local/file:// and S3 NPY/Parquet reference provider."""

    provider_id: ClassVar[str] = "file-reference-v1"

    def __init__(self, object_store: ObjectStore | None = None) -> None:
        self._object_store = object_store or default_object_store()

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

    def _read(self, binding: ReferenceBinding) -> tuple[bytes, str, str]:
        parsed = urlsplit(binding.uri)
        payload = self._object_store.read_bytes(binding.uri)
        suffix = Path(parsed.path).suffix
        return payload, suffix, hashlib.sha256(payload).hexdigest()


__all__ = ["FileReferenceProvider", "ReferenceProvider", "ResolvedReference"]
