"""Thread-safe registry for native write bindings."""

from __future__ import annotations

import importlib.metadata
import threading
from dataclasses import dataclass
from typing import Callable

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from tributo.data.writing.bindings import WriteBinding
from tributo.data.writing.contracts import (
    WriteCapabilityError,
    WriteDescriptor,
    WriteRequest,
)
from tributo.util.annotations import DeveloperAPI

WriteBindingFactory = Callable[[], WriteBinding]


@dataclass(frozen=True)
class RegisteredWriteBinding:
    """Descriptor and factory for one registered writing implementation."""

    descriptor: WriteDescriptor
    factory: WriteBindingFactory


@DeveloperAPI
class WriteBindingRegistry:
    """Resolve a binding by normalized engine, target, and binding ID."""

    def __init__(self) -> None:
        self._bindings: dict[tuple[str, str, str], RegisteredWriteBinding] = {}
        self._lock = threading.RLock()

    def register(
        self, descriptor: WriteDescriptor, factory: WriteBindingFactory
    ) -> None:
        """Register one binding after validating installed distributions."""
        if not callable(factory):
            raise TypeError("write binding factory must be callable")
        self._validate_distributions(descriptor)
        key = (descriptor.engine_id, descriptor.target_kind, descriptor.binding_id)
        with self._lock:
            if key in self._bindings:
                raise WriteCapabilityError(
                    f"Write binding {descriptor.binding_id!r} is already registered"
                )
            self._bindings[key] = RegisteredWriteBinding(descriptor, factory)

    def resolve(self, request: WriteRequest) -> RegisteredWriteBinding:
        """Resolve an exact request target, or fail closed."""
        binding_id = request.binding_id
        with self._lock:
            candidates = tuple(
                item
                for (engine, target_kind, candidate_id), item in self._bindings.items()
                if engine == request.engine
                and target_kind == request.target_kind
                and (binding_id is None or candidate_id == binding_id)
            )
        if len(candidates) == 1:
            candidate = candidates[0]
            if request.mode not in candidate.descriptor.capabilities.supported_modes:
                raise WriteCapabilityError(
                    f"Write binding {candidate.descriptor.binding_id!r} does not "
                    f"support mode {request.mode.value!r}"
                )
            unsupported_options = sorted(
                set(request.options)
                - set(candidate.descriptor.capabilities.supported_options)
            )
            if unsupported_options:
                names = ", ".join(unsupported_options)
                raise WriteCapabilityError(
                    f"Write binding {candidate.descriptor.binding_id!r} does not "
                    f"support write option(s): {names}"
                )
            return candidate
        if len(candidates) > 1:
            ids = ", ".join(sorted(item.descriptor.binding_id for item in candidates))
            raise WriteCapabilityError(
                f"Multiple write bindings match {request.engine}/"
                f"{request.target_kind}: {ids}; set request.binding_id"
            )
        raise WriteCapabilityError(
            f"No write binding matches {request.engine}/{request.target_kind}"
        )

    @staticmethod
    def _validate_distributions(descriptor: WriteDescriptor) -> None:
        try:
            engine_spec = SpecifierSet(descriptor.engine_version_spec)
            declared_binding_version = Version(descriptor.binding_distribution_version)
        except (InvalidSpecifier, InvalidVersion) as exc:
            raise WriteCapabilityError(
                "Write binding declares invalid version metadata"
            ) from exc
        try:
            installed_engine = _installed_distribution_version(
                "ray" if descriptor.engine_id == "tributo.ray_data" else "daft"
            )
            installed_engine_version = Version(installed_engine)
        except (importlib.metadata.PackageNotFoundError, InvalidVersion) as exc:
            raise WriteCapabilityError(
                f"Required engine for {descriptor.engine_id!r} is unavailable"
            ) from exc
        try:
            if installed_engine_version not in engine_spec:
                raise WriteCapabilityError(
                    f"Installed engine version {installed_engine!r} does not satisfy "
                    f"{descriptor.engine_version_spec!r}"
                )
        except InvalidVersion as exc:
            raise WriteCapabilityError("Installed engine version is invalid") from exc
        try:
            installed_binding = _installed_distribution_version(
                descriptor.binding_distribution
            )
            if Version(installed_binding) != declared_binding_version:
                raise WriteCapabilityError(
                    f"Installed binding version {installed_binding!r} does not "
                    f"match declared {descriptor.binding_distribution_version!r}"
                )
        except importlib.metadata.PackageNotFoundError as exc:
            raise WriteCapabilityError(
                f"Required write binding distribution "
                f"{descriptor.binding_distribution!r} is unavailable"
            ) from exc
        except InvalidVersion as exc:
            raise WriteCapabilityError("Installed binding version is invalid") from exc
        for distribution in descriptor.dependency_distributions:
            try:
                _installed_distribution_version(distribution)
            except importlib.metadata.PackageNotFoundError as exc:
                raise WriteCapabilityError(
                    f"Required write binding distribution {distribution!r} is unavailable"
                ) from exc


def _installed_distribution_version(name: str) -> str:
    """Resolve a distribution version in both installed and source checkouts."""
    if name == "tributo":
        from tributo import __version__

        return __version__
    return importlib.metadata.version(name)
