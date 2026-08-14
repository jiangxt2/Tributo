"""Credential-free contracts for distributed Lance vector-index operations."""

from __future__ import annotations

import math
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tributo.data.refs import digest
from tributo.util.annotations import PublicAPI

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_ENV_NAME_RE = re.compile(r"^TRIBUTO_LANCE_NAMESPACE_[A-Z0-9_]+$")
_RESOURCE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]*$")
_MAX_QUERY_DIMENSION = 65_536
_MAX_FILTER_LENGTH = 8_192
_FRAGMENT_SAMPLE_SIZE = 16


class _ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )


@PublicAPI(stability="alpha")
class VectorIndexType(str, Enum):
    """Vector index types verified by Tributo's first compatibility profile."""

    IVF_FLAT = "IVF_FLAT"
    IVF_PQ = "IVF_PQ"


@PublicAPI(stability="alpha")
class VectorMetric(str, Enum):
    """Distance metrics verified by Tributo's first compatibility profile."""

    L2 = "l2"
    COSINE = "cosine"
    DOT = "dot"


@PublicAPI(stability="alpha")
class CoverageStatus(str, Enum):
    """Relationship between current fragments and committed index segments."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    STALE = "stale"
    INDETERMINATE = "indeterminate"


@PublicAPI(stability="alpha")
class ResultDeliveryMode(str, Enum):
    """How a Ray Job returns a distributed search result."""

    INLINE = "inline"
    MATERIALIZED = "materialized"


@PublicAPI(stability="alpha")
class LanceDatasetRef(_ContractModel):
    """Credential-free reference to an existing Lance dataset."""

    uri: str | None = None
    namespace_impl: str | None = None
    namespace_properties_env: str | None = None
    table_id: tuple[str, ...] | None = None
    version: int | str | None = None
    storage_profile: str | None = None
    block_size: int | None = Field(default=None, gt=0)

    @field_validator("uri")
    @classmethod
    def _validate_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("uri must not be empty")
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("uri must not contain user information")
        if parsed.query or parsed.fragment:
            raise ValueError("uri must not contain query parameters or fragments")
        if parsed.scheme not in {"", "file", "s3"}:
            raise ValueError("uri scheme must be local, file, or s3")
        if parsed.scheme == "s3" and (not parsed.netloc or not parsed.path.strip("/")):
            raise ValueError("s3 uri must include a bucket and dataset path")
        if parsed.scheme == "file" and (
            parsed.netloc or not parsed.path.startswith("/")
        ):
            raise ValueError("file uri must be local and contain an absolute path")
        if parsed.scheme == "" and not Path(value).is_absolute():
            raise ValueError("local Lance dataset paths must be absolute")
        return value

    @field_validator("namespace_impl", "storage_profile")
    @classmethod
    def _validate_optional_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if _IDENTIFIER_RE.fullmatch(value) is None:
            raise ValueError("value must be a non-empty identifier")
        return value

    @field_validator("namespace_properties_env")
    @classmethod
    def _validate_namespace_env(cls, value: str | None) -> str | None:
        if value is not None and _ENV_NAME_RE.fullmatch(value) is None:
            raise ValueError(
                "namespace_properties_env must start with "
                "TRIBUTO_LANCE_NAMESPACE_ and contain only A-Z, 0-9, or _"
            )
        return value

    @field_validator("table_id")
    @classmethod
    def _validate_table_id(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        if not value or any(_IDENTIFIER_RE.fullmatch(item) is None for item in value):
            raise ValueError("table_id must contain non-empty identifier components")
        return value

    @field_validator("version", mode="before")
    @classmethod
    def _validate_version(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is int:
            if value < 1:
                raise ValueError("numeric Lance version must be positive")
            return value
        if isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value):
            return value
        raise ValueError("version must be a positive integer or a tag identifier")

    @model_validator(mode="after")
    def _validate_location(self) -> "LanceDatasetRef":
        has_uri = self.uri is not None
        has_namespace = self.namespace_impl is not None or self.table_id is not None
        if has_uri == has_namespace:
            raise ValueError(
                "provide exactly one of uri or namespace_impl with table_id"
            )
        if has_namespace and (self.namespace_impl is None or self.table_id is None):
            raise ValueError("namespace_impl and table_id must be provided together")
        if self.namespace_properties_env is not None and not has_namespace:
            raise ValueError("namespace_properties_env requires namespace mode")
        return self

    @property
    def identity_digest(self) -> str:
        """Return a stable, credential-free identity for receipts and logs."""
        return digest(self.model_dump(mode="json", exclude_none=True))


@PublicAPI(stability="alpha")
class RayWorkerResources(_ContractModel):
    """Reviewed subset of Ray remote arguments exposed to users."""

    num_cpus: float = Field(default=1.0, gt=0)
    num_gpus: float = Field(default=0.0, ge=0)
    memory: int | None = Field(default=None, gt=0)
    resources: dict[str, float] = Field(default_factory=dict)

    @field_validator("resources")
    @classmethod
    def _validate_resources(cls, value: dict[str, float]) -> dict[str, float]:
        for name, quantity in value.items():
            if _RESOURCE_RE.fullmatch(name) is None:
                raise ValueError("custom Ray resource names must be identifiers")
            if not math.isfinite(quantity) or quantity <= 0:
                raise ValueError("custom Ray resource quantities must be positive")
        return dict(value)

    def to_ray_remote_args(self) -> dict[str, Any]:
        """Convert the reviewed resource contract to Ray remote arguments."""
        values: dict[str, Any] = {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
        }
        if self.memory is not None:
            values["memory"] = self.memory
        if self.resources:
            values["resources"] = dict(self.resources)
        return values


@PublicAPI(stability="alpha")
class LanceScannerOptions(_ContractModel):
    """Versioned scanner allowlist for PyLance 9.0.0."""

    batch_size: int | None = Field(default=None, gt=0)
    batch_size_bytes: int | None = Field(default=None, gt=0)
    batch_readahead: int | None = Field(default=None, ge=0)
    fragment_readahead: int | None = Field(default=None, ge=0)
    scan_in_order: bool | None = None
    prefilter: bool | None = None
    with_row_id: bool | None = None
    with_row_address: bool | None = None
    use_stats: bool | None = None
    io_buffer_size: int | None = Field(default=None, gt=0)
    late_materialization: bool | tuple[str, ...] | None = None
    strict_batch_size: bool | None = None

    def to_lance_options(self) -> dict[str, Any]:
        """Return only explicitly configured, reviewed scanner fields."""
        values = self.model_dump(mode="python", exclude_none=True)
        if isinstance(values.get("late_materialization"), tuple):
            values["late_materialization"] = list(values["late_materialization"])
        return values


@PublicAPI(stability="alpha")
class VectorIndexBuildRequest(_ContractModel):
    """Build an IVF index over vectors already stored in Lance."""

    dataset: LanceDatasetRef
    column: str = Field(min_length=1)
    index_name: str = Field(min_length=1)
    index_type: VectorIndexType
    metric: VectorMetric = VectorMetric.L2
    num_workers: int = Field(default=4, ge=1, le=1_024)
    num_segments: int | None = Field(default=None, ge=1)
    num_partitions: int | None = Field(default=None, ge=1)
    num_sub_vectors: int | None = Field(default=None, ge=1)
    sample_rate: int = Field(default=256, ge=2)
    replace: bool = False
    worker_resources: RayWorkerResources = Field(default_factory=RayWorkerResources)
    request_key: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("column", "index_name")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        if _IDENTIFIER_RE.fullmatch(value) is None:
            raise ValueError("column and index_name must be identifiers")
        return value

    @model_validator(mode="after")
    def _validate_index_parameters(self) -> "VectorIndexBuildRequest":
        if self.index_type is VectorIndexType.IVF_PQ:
            if self.num_sub_vectors is None:
                raise ValueError("IVF_PQ requires num_sub_vectors")
        elif self.num_sub_vectors is not None:
            raise ValueError("num_sub_vectors is only valid for IVF_PQ")
        return self

    @property
    def request_digest(self) -> str:
        return digest(self.model_dump(mode="json", exclude_none=True))


@PublicAPI(stability="alpha")
class SearchResultOutput(_ContractModel):
    """Bounded result-delivery contract for Ray Job and CLI searches."""

    mode: ResultDeliveryMode = ResultDeliveryMode.INLINE
    output_uri: str | None = None
    format: Literal["parquet"] = "parquet"
    inline_max_rows: int = Field(default=100, ge=1, le=1_000)
    inline_max_bytes: int = Field(
        default=1_048_576,
        ge=1,
        le=16_777_216,
        description="Maximum UTF-8 JSON payload size for inline rows",
    )

    @field_validator("output_uri")
    @classmethod
    def _validate_output_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("output_uri must not contain user information")
        if parsed.query or parsed.fragment:
            raise ValueError("output_uri must not contain query or fragment data")
        if parsed.scheme not in {"", "file", "s3"}:
            raise ValueError("output_uri scheme must be local, file, or s3")
        if parsed.scheme == "" and not Path(value).is_absolute():
            raise ValueError("local output paths must be absolute")
        if parsed.scheme == "s3" and (not parsed.netloc or not parsed.path.strip("/")):
            raise ValueError("s3 output_uri must include bucket and object key")
        if parsed.scheme == "file" and (
            parsed.netloc or not parsed.path.startswith("/")
        ):
            raise ValueError(
                "file output_uri must be local and contain an absolute path"
            )
        if not parsed.path.endswith(".parquet"):
            raise ValueError("output_uri must identify a .parquet file")
        return value

    @model_validator(mode="after")
    def _validate_delivery(self) -> "SearchResultOutput":
        if self.mode is ResultDeliveryMode.MATERIALIZED and self.output_uri is None:
            raise ValueError("materialized delivery requires output_uri")
        if self.mode is ResultDeliveryMode.INLINE and self.output_uri is not None:
            raise ValueError("inline delivery must not include output_uri")
        return self


@PublicAPI(stability="alpha")
class VectorSearchRequest(_ContractModel):
    """Run a fixed-version distributed Top-K query through Lance-Ray.

    The dimension bound applies to direct in-process calls. Ray Job submission
    additionally enforces a 64 KiB bound on the complete serialized request.
    """

    dataset: LanceDatasetRef
    column: str = Field(min_length=1)
    query_vector: tuple[float, ...] = Field(repr=False, min_length=1)
    k: int = Field(default=10, ge=1, le=10_000)
    index_name: str = Field(min_length=1)
    metric: VectorMetric = VectorMetric.L2
    columns: tuple[str, ...] | None = None
    filter: str | None = Field(default=None, max_length=_MAX_FILTER_LENGTH)
    minimum_nprobes: int | None = Field(default=None, ge=0)
    maximum_nprobes: int | None = Field(default=None, ge=0)
    refine_factor: int | None = Field(default=None, ge=1)
    oversample_factor: float = Field(default=1.0, ge=1.0, le=100.0)
    include_unindexed: bool = True
    fast_search: bool = False
    num_workers: int = Field(default=4, ge=1, le=1_024)
    worker_resources: RayWorkerResources = Field(default_factory=RayWorkerResources)
    scanner_options: LanceScannerOptions = Field(default_factory=LanceScannerOptions)
    result: SearchResultOutput = Field(default_factory=SearchResultOutput)
    request_key: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("column", "index_name")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        if _IDENTIFIER_RE.fullmatch(value) is None:
            raise ValueError("column and index_name must be identifiers")
        return value

    @field_validator("query_vector")
    @classmethod
    def _validate_query_vector(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) > _MAX_QUERY_DIMENSION:
            raise ValueError("query vector exceeds the supported dimension bound")
        if any(not math.isfinite(item) for item in value):
            raise ValueError("query vector values must be finite")
        return value

    @field_validator("columns")
    @classmethod
    def _validate_columns(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        if not value or any(_IDENTIFIER_RE.fullmatch(item) is None for item in value):
            raise ValueError("columns must contain identifiers")
        if len(set(value)) != len(value):
            raise ValueError("columns must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _validate_probe_range(self) -> "VectorSearchRequest":
        if (
            self.minimum_nprobes is not None
            and self.maximum_nprobes is not None
            and self.minimum_nprobes > self.maximum_nprobes
        ):
            raise ValueError("minimum_nprobes must be <= maximum_nprobes")
        if self.fast_search and self.include_unindexed:
            raise ValueError(
                "fast_search skips unindexed fragments; set include_unindexed=false"
            )
        return self

    @property
    def request_digest(self) -> str:
        return digest(self.model_dump(mode="json", exclude_none=True))


@PublicAPI(stability="alpha")
class VectorOptimizeRequest(_ContractModel):
    """Incrementally index fragments appended after an initial build."""

    dataset: LanceDatasetRef
    indices: tuple[str, ...] | None = None
    num_indices_to_merge: int = Field(default=1, ge=0)
    retrain: bool = False
    request_key: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("indices")
    @classmethod
    def _validate_indices(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        if not value or any(_IDENTIFIER_RE.fullmatch(item) is None for item in value):
            raise ValueError("indices must contain identifiers")
        return value

    @model_validator(mode="after")
    def _reject_historical_version(self) -> "VectorOptimizeRequest":
        if self.dataset.version is not None:
            raise ValueError("index optimization only operates on the current version")
        return self

    @property
    def request_digest(self) -> str:
        return digest(self.model_dump(mode="json", exclude_none=True))


@PublicAPI(stability="alpha")
class LanceCompactionOptions(_ContractModel):
    """Reviewed subset of PyLance 9.0.0 distributed compaction options."""

    target_rows_per_fragment: int | None = Field(default=None, gt=0)
    max_rows_per_group: int | None = Field(default=None, gt=0)
    max_bytes_per_file: int | None = Field(default=None, gt=0)
    materialize_deletions: bool | None = None
    num_threads: int | None = Field(default=None, gt=0)
    batch_size: int | None = Field(default=None, gt=0)
    io_buffer_size: int | None = Field(default=None, gt=0)
    compaction_mode: (
        Literal["reencode", "try_binary_copy", "force_binary_copy"] | None
    ) = None
    defer_index_remap: bool | None = None
    max_source_fragments: int | None = Field(default=None, gt=0)

    def to_lance_options(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude_none=True)


@PublicAPI(stability="alpha")
class VectorCompactRequest(_ContractModel):
    """Run Lance-Ray distributed file compaction on the current version."""

    dataset: LanceDatasetRef
    options: LanceCompactionOptions = Field(default_factory=LanceCompactionOptions)
    num_workers: int = Field(default=4, ge=1, le=1_024)
    worker_resources: RayWorkerResources = Field(default_factory=RayWorkerResources)
    request_key: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def _reject_historical_version(self) -> "VectorCompactRequest":
        if self.dataset.version is not None:
            raise ValueError("compaction only operates on the current version")
        return self

    @property
    def request_digest(self) -> str:
        return digest(self.model_dump(mode="json", exclude_none=True))


class FragmentSetEvidence(_ContractModel):
    """Bounded evidence for a potentially very large fragment-ID set."""

    count: int = Field(ge=0)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_ids: tuple[int, ...] = ()

    @classmethod
    def from_ids(cls, fragment_ids: set[int]) -> "FragmentSetEvidence":
        ordered = sorted(fragment_ids)
        return cls(
            count=len(ordered),
            digest=digest(ordered),
            sample_ids=tuple(ordered[:_FRAGMENT_SAMPLE_SIZE]),
        )


class IndexCoverageEvidence(_ContractModel):
    """Post-commit evidence derived from Lance fragment and index metadata."""

    status: CoverageStatus
    planning: FragmentSetEvidence
    current: FragmentSetEvidence
    indexed: FragmentSetEvidence
    unindexed: FragmentSetEvidence
    stale: FragmentSetEvidence
    segment_count: int | None = Field(default=None, ge=0)
    overlapping_fragment_count: int = Field(default=0, ge=0)


class RuntimeVersionEvidence(_ContractModel):
    """Driver and worker distribution versions for one operation."""

    ray: str
    pylance: str
    lance_ray: str
    pyarrow: str
    worker_count: int = Field(ge=0)
    worker_versions: tuple[dict[str, str], ...] = ()
    worker_validation_complete: bool = False


@PublicAPI(stability="alpha")
class VectorIndexBuildReceipt(_ContractModel):
    """Credential-free receipt for a committed distributed index build."""

    receipt_version: Literal[1] = 1
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    planning_base_version: int
    output_dataset_version: int
    index_name: str
    index_type: VectorIndexType
    metric: VectorMetric
    coverage: IndexCoverageEvidence
    num_workers: int
    worker_resources: RayWorkerResources
    runtime: RuntimeVersionEvidence
    warnings: tuple[str, ...] = ()
    elapsed_seconds: float = Field(ge=0)


@PublicAPI(stability="alpha")
class VectorSearchReceipt(_ContractModel):
    """Credential-free metadata plus bounded or materialized query results."""

    receipt_version: Literal[1] = 1
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_version: int
    index_name: str
    metric: VectorMetric
    k: int
    row_count: int
    include_unindexed: bool
    fast_search: bool
    oversample_factor: float
    num_workers: int
    worker_resources: RayWorkerResources
    delivery_mode: ResultDeliveryMode
    inline_rows: tuple[dict[str, Any], ...] = ()
    output_uri: str | None = None
    output_format: Literal["parquet"] | None = None
    runtime: RuntimeVersionEvidence
    elapsed_seconds: float = Field(ge=0)


@PublicAPI(stability="alpha")
class VectorMaintenanceReceipt(_ContractModel):
    """Receipt for optimize_indices or compact_files."""

    receipt_version: Literal[1] = 1
    operation: Literal["optimize", "compact"]
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_dataset_version: int
    output_dataset_version: int
    coverage: tuple[IndexCoverageEvidence, ...]
    metrics: dict[str, Any] | None = None
    num_workers: int
    worker_resources: RayWorkerResources
    runtime: RuntimeVersionEvidence
    warnings: tuple[str, ...] = ()
    elapsed_seconds: float = Field(ge=0)
