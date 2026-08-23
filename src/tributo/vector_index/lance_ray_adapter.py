"""Thin, fail-closed adapter around the public Lance-Ray 0.5.0 API."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import pyarrow as pa

from tributo._common.storage_profiles import StorageProfileResolver
from tributo.data._s3 import to_lance_storage_options
from tributo.data.persistence import LanceRayBinding
from tributo.vector_index.contracts import (
    CoverageStatus,
    FragmentSetEvidence,
    IndexCoverageEvidence,
    LanceDatasetRef,
    RayWorkerResources,
    RuntimeVersionEvidence,
    VectorCompactRequest,
    VectorIndexBuildRequest,
    VectorOptimizeRequest,
    VectorSearchRequest,
)
from tributo.vector_index.errors import (
    VectorIndexConfigurationError,
    VectorIndexDependencyError,
    VectorIndexExecutionError,
)

_EXPECTED_DISTRIBUTIONS = {
    "ray": "2.55.1",
    "pylance": "9.0.0",
    "lance-ray": "0.5.0",
    "pyarrow": "19.0.1",
}
_REQUIRED_LANCE_RAY_PARAMETERS = {
    "create_index": {
        "uri",
        "column",
        "index_type",
        "name",
        "replace",
        "num_workers",
        "num_segments",
        "metric",
        "num_partitions",
        "num_sub_vectors",
        "sample_rate",
        "ray_remote_args",
        "storage_options",
        "block_size",
        "namespace_impl",
        "namespace_properties",
        "table_id",
    },
    "vector_search": {
        "uri",
        "nearest",
        "index_name",
        "columns",
        "filter",
        "num_workers",
        "oversample_factor",
        "include_unindexed",
        "fast_search",
        "scanner_options",
        "ray_remote_args",
    },
    "optimize_indices": {
        "uri",
        "indices",
        "num_indices_to_merge",
        "retrain",
        "storage_options",
        "namespace_impl",
        "namespace_properties",
        "table_id",
    },
    "compact_files": {
        "uri",
        "compaction_options",
        "num_workers",
        "ray_remote_args",
        "storage_options",
        "namespace_impl",
        "namespace_properties",
        "table_id",
    },
}


@dataclass(frozen=True)
class ResolvedDatasetAccess:
    """Runtime-only dataset handle and options; never place this in a receipt."""

    dataset: Any = field(repr=False)
    uri: str
    storage_options: dict[str, str] = field(repr=False)
    namespace_impl: str | None = None
    namespace_properties: dict[str, str] | None = field(default=None, repr=False)
    table_id: list[str] | None = None


def _distribution_versions() -> dict[str, str]:
    return {
        distribution: importlib.metadata.version(distribution)
        for distribution in _EXPECTED_DISTRIBUTIONS
    }


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _fragment_ids(dataset: Any) -> set[int]:
    result: set[int] = set()
    for fragment in dataset.get_fragments():
        fragment_id = _value(fragment, "fragment_id", None)
        if fragment_id is None:
            metadata = _value(fragment, "metadata", None)
            fragment_id = _value(metadata, "id", None)
        if type(fragment_id) is not int:
            raise VectorIndexExecutionError(
                "Lance returned fragment metadata without an integer fragment ID"
            )
        result.add(fragment_id)
    return result


def _index_by_name(dataset: Any, index_name: str) -> Any | None:
    for index in dataset.describe_indices():
        if _value(index, "name") == index_name:
            return index
    return None


def validate_vector_index_target(
    index: Any,
    *,
    column: str,
    metric: str,
) -> None:
    """Fail closed unless the named index matches the requested vector search."""
    field_names = _value(index, "field_names", None)
    if not isinstance(field_names, list) or field_names != [column]:
        raise VectorIndexConfigurationError(
            "requested index does not target the requested vector column"
        )
    index_type = _value(index, "index_type", None)
    if index_type not in {"IVF_FLAT", "IVF_PQ"}:
        raise VectorIndexConfigurationError(
            "requested index is not a supported vector index"
        )
    details = _value(index, "details", None)
    index_metric = _value(details, "metric_type", None)
    if not isinstance(index_metric, str) or index_metric.lower() != metric.lower():
        raise VectorIndexConfigurationError(
            "requested metric does not match the vector index metric"
        )


def validate_search_index_coverage(dataset: Any, index: Any) -> None:
    """Reject index metadata that cannot safely address the current fragments."""
    segments = _value(index, "segments", None)
    if not isinstance(segments, (list, tuple)):
        raise VectorIndexConfigurationError(
            "requested index does not expose verifiable segment coverage"
        )
    current = _fragment_ids(dataset)
    indexed, _segment_count, overlap_count = _index_fragment_coverage(index)
    if overlap_count:
        raise VectorIndexConfigurationError(
            "requested index has overlapping fragment coverage and must be rebuilt"
        )
    if indexed - current:
        raise VectorIndexConfigurationError(
            "requested index has stale fragment coverage and must be rebuilt"
        )
    if current and not indexed:
        raise VectorIndexConfigurationError(
            "requested index coverage cannot be verified for the current dataset"
        )


def _index_fragment_coverage(index: Any) -> tuple[set[int], int, int]:
    indexed: set[int] = set()
    overlapping: set[int] = set()
    segments = list(_value(index, "segments", ()) or ())
    for segment in segments:
        raw_ids = _value(segment, "fragment_ids", ()) or ()
        if not isinstance(raw_ids, (list, tuple, set, frozenset)):
            raise VectorIndexConfigurationError(
                "index does not expose verifiable fragment coverage"
            )
        try:
            segment_ids = {int(fragment_id) for fragment_id in raw_ids}
        except (TypeError, ValueError, OverflowError):
            raise VectorIndexConfigurationError(
                "index does not expose verifiable fragment coverage"
            ) from None
        overlapping.update(indexed.intersection(segment_ids))
        indexed.update(segment_ids)
    return indexed, len(segments), len(overlapping)


def summarize_index_coverage(
    *,
    planning_fragment_ids: set[int],
    current_fragment_ids: set[int],
    index: Any | None,
) -> IndexCoverageEvidence:
    """Classify post-commit index coverage without inventing a snapshot claim."""
    if index is None:
        indexed: set[int] = set()
        segment_count: int | None = None
        overlap_count = 0
        status = CoverageStatus.INDETERMINATE
    else:
        indexed, segment_count, overlap_count = _index_fragment_coverage(index)
        disappeared = planning_fragment_ids - current_fragment_ids
        stale = indexed - current_fragment_ids
        unindexed = current_fragment_ids - indexed
        unindexed_planning = planning_fragment_ids - indexed
        appended = current_fragment_ids - planning_fragment_ids
        if disappeared or stale or overlap_count:
            status = CoverageStatus.STALE
        elif unindexed_planning:
            status = CoverageStatus.INDETERMINATE
        elif not unindexed:
            status = CoverageStatus.COMPLETE
        elif unindexed == appended:
            status = CoverageStatus.PARTIAL
        else:
            status = CoverageStatus.INDETERMINATE

    stale_ids = indexed - current_fragment_ids
    unindexed_ids = current_fragment_ids - indexed
    return IndexCoverageEvidence(
        status=status,
        planning=FragmentSetEvidence.from_ids(planning_fragment_ids),
        current=FragmentSetEvidence.from_ids(current_fragment_ids),
        indexed=FragmentSetEvidence.from_ids(indexed),
        unindexed=FragmentSetEvidence.from_ids(unindexed_ids),
        stale=FragmentSetEvidence.from_ids(stale_ids),
        segment_count=segment_count,
        overlapping_fragment_count=overlap_count,
    )


def _namespace_properties(ref: LanceDatasetRef) -> dict[str, str] | None:
    env_name = ref.namespace_properties_env
    if env_name is None:
        return None
    raw = os.environ.get(env_name)
    if raw is None:
        raise VectorIndexConfigurationError(
            f"namespace properties environment variable {env_name} is not set"
        )
    if len(raw.encode("utf-8")) > 65_536:
        raise VectorIndexConfigurationError(
            f"namespace properties environment variable {env_name} is too large"
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise VectorIndexConfigurationError(
            f"namespace properties environment variable {env_name} must be JSON"
        ) from None
    if not isinstance(parsed, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in parsed.items()
    ):
        raise VectorIndexConfigurationError(
            f"namespace properties environment variable {env_name} must contain "
            "a JSON object of string values"
        )
    return dict(parsed)


def _vector_field(schema: pa.Schema, column: str) -> pa.Field:
    path = column.split(".")
    try:
        field = schema.field(path[0])
        for component in path[1:]:
            if not pa.types.is_struct(field.type):
                raise KeyError(component)
            field = field.type.field(component)
        return field
    except (KeyError, IndexError):
        raise VectorIndexConfigurationError(
            f"vector column {column!r} does not exist in the Lance schema"
        ) from None


def validate_vector_schema(dataset: Any, column: str) -> int:
    """Return the fixed vector dimension after validating the Lance schema."""
    field = _vector_field(dataset.schema, column)
    field_type = field.type
    if pa.types.is_fixed_size_list(field_type):
        dimension = field_type.list_size
        value_type = field_type.value_type
    elif isinstance(field_type, pa.FixedShapeTensorType) and len(field_type.shape) == 1:
        dimension = field_type.shape[0]
        value_type = field_type.value_type
    else:
        raise VectorIndexConfigurationError(
            "vector column must use a fixed-size list or one-dimensional "
            "fixed-shape tensor type"
        )
    if not pa.types.is_floating(value_type):
        raise VectorIndexConfigurationError(
            "the first Tributo vector-index profile supports floating vectors only"
        )
    return int(dimension)


class LanceRayAdapter:
    """Public-API-only bridge from Tributo contracts to Lance-Ray."""

    def __init__(self) -> None:
        try:
            self._lance_ray = importlib.import_module("lance_ray")
            self._lance = importlib.import_module("lance")
            self._ray = importlib.import_module("ray")
        except ImportError:
            raise VectorIndexDependencyError(
                "distributed vector indexing requires 'tributo[vector-index]'"
            ) from None
        self._validate_driver_runtime()
        self._validate_public_api()
        self._lance_binding = LanceRayBinding(self._lance_ray, self._lance)

    def _validate_driver_runtime(self) -> None:
        versions = _distribution_versions()
        mismatches = {
            name: (expected, versions.get(name))
            for name, expected in _EXPECTED_DISTRIBUTIONS.items()
            if versions.get(name) != expected
        }
        if mismatches:
            summary = ", ".join(
                f"{name} expected {expected}, found {actual}"
                for name, (expected, actual) in sorted(mismatches.items())
            )
            raise VectorIndexDependencyError(
                f"unsupported vector-index runtime: {summary}"
            )

    def _validate_public_api(self) -> None:
        for function_name, required in _REQUIRED_LANCE_RAY_PARAMETERS.items():
            function = getattr(self._lance_ray, function_name, None)
            if function is None:
                raise VectorIndexDependencyError(
                    f"lance-ray 0.5.0 does not expose {function_name}"
                )
            parameters = set(inspect.signature(function).parameters)
            missing = sorted(required - parameters)
            if missing:
                raise VectorIndexDependencyError(
                    f"lance-ray {function_name} is missing parameters {missing}"
                )

    def require_initialized_ray(self) -> None:
        """Reject accidental non-Ray execution of a distributed operation."""
        if not self._ray.is_initialized():
            raise VectorIndexConfigurationError(
                "Ray must be initialized before a distributed vector operation"
            )

    def runtime_evidence(
        self,
        *,
        num_workers: int,
        resources: RayWorkerResources,
    ) -> RuntimeVersionEvidence:
        """Validate package versions on actual Ray worker tasks."""
        self.require_initialized_ray()
        remote_probe = self._ray.remote(_distribution_versions)
        probe = remote_probe.options(
            **resources.to_ray_remote_args(),
            scheduling_strategy="SPREAD",
        )
        worker_payloads = tuple(
            self._ray.get([probe.remote() for _ in range(num_workers)])
        )
        expected = _distribution_versions()
        for worker in worker_payloads:
            if worker != expected:
                raise VectorIndexDependencyError(
                    "Ray worker vector-index dependency versions differ from driver"
                )
        return RuntimeVersionEvidence(
            ray=expected["ray"],
            pylance=expected["pylance"],
            lance_ray=expected["lance-ray"],
            pyarrow=expected["pyarrow"],
            worker_count=len(worker_payloads),
            worker_versions=(dict(expected),) if worker_payloads else (),
            worker_validation_complete=len(worker_payloads) == num_workers,
        )

    def _storage_options(self, ref: LanceDatasetRef) -> dict[str, str]:
        if ref.storage_profile is None and (
            ref.uri is None or not ref.uri.startswith("s3://")
        ):
            return {}
        profile = StorageProfileResolver().resolve(ref.storage_profile)
        return dict(to_lance_storage_options(profile) or {})

    def open_dataset(
        self,
        ref: LanceDatasetRef,
        *,
        version: int | str | None,
    ) -> ResolvedDatasetAccess:
        """Open one explicit Lance view through URI or public namespace APIs."""
        storage_options = self._storage_options(ref)
        namespace_properties = _namespace_properties(ref)
        table_id = list(ref.table_id) if ref.table_id is not None else None
        try:
            access = self._lance_binding.open_dataset(
                uri=ref.uri,
                version=version,
                block_size=ref.block_size,
                storage_options=storage_options,
                namespace_impl=ref.namespace_impl,
                namespace_properties=namespace_properties,
                table_id=table_id,
            )
        except Exception as exc:
            raise VectorIndexExecutionError(
                f"Lance dataset resolution failed ({type(exc).__name__})"
            ) from None
        return ResolvedDatasetAccess(
            dataset=access.dataset,
            uri=access.uri,
            storage_options=access.storage_options,
            namespace_impl=access.namespace_impl,
            namespace_properties=access.namespace_properties,
            table_id=access.table_id,
        )

    def current_dataset(self, ref: LanceDatasetRef) -> ResolvedDatasetAccess:
        return self.open_dataset(ref, version=None)

    def requested_dataset(self, ref: LanceDatasetRef) -> ResolvedDatasetAccess:
        return self.open_dataset(ref, version=ref.version)

    def create_index(self, request: VectorIndexBuildRequest) -> Any:
        """Map a validated build request to lance_ray.create_index."""
        ref = request.dataset
        common: dict[str, Any] = {
            "column": request.column,
            "index_type": request.index_type.value,
            "name": request.index_name,
            "replace": request.replace,
            "num_workers": request.num_workers,
            "num_segments": request.num_segments,
            "storage_options": self._storage_options(ref) or None,
            "block_size": ref.block_size,
            "ray_remote_args": request.worker_resources.to_ray_remote_args(),
            "metric": request.metric.value,
            "num_partitions": request.num_partitions,
            "num_sub_vectors": request.num_sub_vectors,
            "sample_rate": request.sample_rate,
        }
        if ref.uri is not None:
            common["uri"] = ref.uri
        else:
            common.update(
                uri=None,
                namespace_impl=ref.namespace_impl,
                namespace_properties=_namespace_properties(ref),
                table_id=list(ref.table_id or ()),
            )
        try:
            return self._lance_binding.create_index(**common)
        except Exception as exc:
            raise VectorIndexExecutionError(
                f"Lance-Ray create_index failed ({type(exc).__name__})"
            ) from None

    def vector_search(self, request: VectorSearchRequest, dataset: Any) -> pa.Table:
        """Run vector_search against a pre-opened, fixed-version dataset."""
        nearest: dict[str, Any] = {
            "column": request.column,
            "q": list(request.query_vector),
            "k": request.k,
            "metric": request.metric.value,
        }
        for name in ("minimum_nprobes", "maximum_nprobes", "refine_factor"):
            value = getattr(request, name)
            if value is not None:
                nearest[name] = value
        try:
            result = self._lance_binding.vector_search(
                uri=dataset,
                nearest=nearest,
                index_name=request.index_name,
                columns=list(request.columns) if request.columns is not None else None,
                filter=request.filter,
                num_workers=request.num_workers,
                ray_remote_args=request.worker_resources.to_ray_remote_args(),
                oversample_factor=request.oversample_factor,
                include_unindexed=request.include_unindexed,
                fast_search=request.fast_search,
                analyze_plan=False,
                scanner_options=request.scanner_options.to_lance_options(),
            )
        except Exception as exc:
            raise VectorIndexExecutionError(
                f"Lance-Ray vector_search failed ({type(exc).__name__})"
            ) from None
        if not isinstance(result, pa.Table):
            raise VectorIndexExecutionError(
                "Lance-Ray vector_search did not return an Arrow table"
            )
        return result

    def optimize_indices(self, request: VectorOptimizeRequest) -> Any:
        """Map a validated maintenance request to lance_ray.optimize_indices."""
        ref = request.dataset
        kwargs: dict[str, Any] = {
            "uri": ref.uri,
            "indices": list(request.indices) if request.indices is not None else None,
            "num_indices_to_merge": request.num_indices_to_merge,
            "retrain": request.retrain,
            "storage_options": self._storage_options(ref) or None,
        }
        if ref.uri is None:
            kwargs.update(
                namespace_impl=ref.namespace_impl,
                namespace_properties=_namespace_properties(ref),
                table_id=list(ref.table_id or ()),
            )
        try:
            return self._lance_binding.optimize_indices(**kwargs)
        except Exception as exc:
            raise VectorIndexExecutionError(
                f"Lance-Ray optimize_indices failed ({type(exc).__name__})"
            ) from None

    def compact_files(self, request: VectorCompactRequest) -> Any:
        """Map a validated maintenance request to lance_ray.compact_files."""
        ref = request.dataset
        kwargs: dict[str, Any] = {
            "uri": ref.uri,
            "compaction_options": request.options.to_lance_options(),
            "num_workers": request.num_workers,
            "storage_options": self._storage_options(ref) or None,
            "ray_remote_args": request.worker_resources.to_ray_remote_args(),
        }
        if ref.uri is None:
            kwargs.update(
                namespace_impl=ref.namespace_impl,
                namespace_properties=_namespace_properties(ref),
                table_id=list(ref.table_id or ()),
            )
        try:
            return self._lance_binding.compact_files(**kwargs)
        except Exception as exc:
            raise VectorIndexExecutionError(
                f"Lance-Ray compact_files failed ({type(exc).__name__})"
            ) from None

    @staticmethod
    def fragment_ids(dataset: Any) -> set[int]:
        return _fragment_ids(dataset)

    @staticmethod
    def index_by_name(dataset: Any, index_name: str) -> Any | None:
        return _index_by_name(dataset, index_name)

    @staticmethod
    def index_names(dataset: Any) -> tuple[str, ...]:
        return tuple(str(_value(index, "name")) for index in dataset.describe_indices())

    @staticmethod
    def coverage_for_indices(
        *,
        planning_fragment_ids: set[int],
        current_dataset: Any,
        index_names: Iterable[str],
    ) -> tuple[IndexCoverageEvidence, ...]:
        current_ids = _fragment_ids(current_dataset)
        return tuple(
            summarize_index_coverage(
                planning_fragment_ids=planning_fragment_ids,
                current_fragment_ids=current_ids,
                index=_index_by_name(current_dataset, index_name),
            )
            for index_name in index_names
        )
