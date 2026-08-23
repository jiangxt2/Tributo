"""Thin Lance/Lance-Ray physical data binding.

Vector Index owns index semantics and coverage validation.  This module owns
the native dataset access and Lance-Ray calls so those physical APIs do not
leak into the vector business orchestration layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from tributo.util.annotations import DeveloperAPI


@DeveloperAPI
@dataclass(frozen=True)
class ResolvedLanceDataset:
    """Runtime Lance dataset access returned by a physical Binding."""

    dataset: Any = field(repr=False)
    uri: str
    storage_options: dict[str, str] = field(repr=False)
    namespace_impl: str | None = None
    namespace_properties: dict[str, str] | None = field(default=None, repr=False)
    table_id: list[str] | None = None


@DeveloperAPI
class LanceRayBinding:
    """Delegate Lance dataset and Lance-Ray operations to public APIs."""

    binding_id = "tributo.ray.lance"

    def __init__(self, lance_ray: Any, lance: Any) -> None:
        self._lance_ray = lance_ray
        self._lance = lance

    def open_dataset(
        self,
        *,
        uri: str | None,
        version: int | str | None,
        block_size: int | None,
        storage_options: dict[str, str],
        namespace_impl: str | None,
        namespace_properties: dict[str, str] | None,
        table_id: list[str] | None,
    ) -> ResolvedLanceDataset:
        """Open a URI or namespace-resolved Lance dataset."""
        if uri is not None:
            dataset = self._lance.LanceDataset(
                uri,
                version=version,
                block_size=block_size,
                storage_options=storage_options or None,
            )
            return ResolvedLanceDataset(
                dataset=dataset,
                uri=uri,
                storage_options=storage_options,
            )

        if namespace_impl is None or table_id is None:
            raise ValueError("namespace dataset reference is incomplete")
        try:
            lance_namespace = __import__("lance_namespace")
            namespace = lance_namespace.connect(
                namespace_impl,
                namespace_properties or {},
            )
            description = namespace.describe_table(
                lance_namespace.DescribeTableRequest(id=table_id)
            )
        except Exception as exc:
            raise RuntimeError("Lance namespace resolution failed") from exc

        location = getattr(description, "location", None)
        if not isinstance(location, str) or not location:
            raise RuntimeError("Lance namespace did not return a table location")
        namespace_storage = getattr(description, "storage_options", None)
        if isinstance(namespace_storage, Mapping):
            storage_options = {
                **storage_options,
                **{str(key): str(value) for key, value in namespace_storage.items()},
            }
        dataset = self._lance.LanceDataset(
            location,
            version=version,
            block_size=block_size,
            storage_options=storage_options or None,
            namespace_client=namespace,
            table_id=table_id,
        )
        return ResolvedLanceDataset(
            dataset=dataset,
            uri=location,
            storage_options=storage_options,
            namespace_impl=namespace_impl,
            namespace_properties=namespace_properties,
            table_id=table_id,
        )

    def create_index(self, **kwargs: Any) -> Any:
        """Delegate index creation to Lance-Ray."""
        return self._lance_ray.create_index(**kwargs)

    def vector_search(self, **kwargs: Any) -> Any:
        """Delegate vector search to Lance-Ray."""
        return self._lance_ray.vector_search(**kwargs)

    def optimize_indices(self, **kwargs: Any) -> Any:
        """Delegate index optimization to Lance-Ray."""
        return self._lance_ray.optimize_indices(**kwargs)

    def compact_files(self, **kwargs: Any) -> Any:
        """Delegate file compaction to Lance-Ray."""
        return self._lance_ray.compact_files(**kwargs)


__all__ = ["LanceRayBinding", "ResolvedLanceDataset"]
