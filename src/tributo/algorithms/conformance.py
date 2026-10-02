"""Reusable public Conformance Testkit for installed algorithm Wheels."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from packaging.utils import canonicalize_name

from tributo.algorithms.api import DistributedAlgorithmDescriptor
from tributo.algorithms.core.contracts import validate_contract_binding
from tributo.plugin import validate_distributed_algorithm_descriptor
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class AlgorithmPackageConformanceReport:
    """Immutable facts proven without importing the algorithm implementation."""

    algorithm_id: str
    implementation_id: str
    distribution: str
    package_version: str
    entry_point_name: str
    contract_ids: tuple[str, ...]
    implementation_loaded: bool = False


@PublicAPI(stability="alpha")
def validate_algorithm_descriptor_conformance(
    descriptor: DistributedAlgorithmDescriptor,
    *,
    entry_point_name: str,
    distribution_name: str,
) -> AlgorithmPackageConformanceReport:
    """Validate ownership and executable contracts without loading model code."""
    validated = validate_distributed_algorithm_descriptor(
        descriptor,
        entry_point_name=entry_point_name,
        entry_point_distribution_name=distribution_name,
        load_implementation=False,
    )
    bindings = validated.registration.contract_bindings
    if bindings is None:
        contract_ids: tuple[str, ...] = ()
    else:
        selected = (
            bindings.config,
            bindings.input,
            bindings.output,
            *((bindings.coverage,) if bindings.coverage is not None else ()),
        )
        for binding in selected:
            validate_contract_binding(binding)
        contract_ids = tuple(binding.contract_id for binding in selected)
    return AlgorithmPackageConformanceReport(
        algorithm_id=validated.name,
        implementation_id=(validated.registration.implementation.implementation_id),
        distribution=validated.package_name,
        package_version=validated.package_version,
        entry_point_name=entry_point_name,
        contract_ids=contract_ids,
    )


@PublicAPI(stability="alpha")
def validate_installed_algorithm_package(
    descriptor: DistributedAlgorithmDescriptor,
    *,
    entry_point_name: str,
) -> AlgorithmPackageConformanceReport:
    """Validate one installed Wheel using its actual distribution metadata."""
    distribution = importlib.metadata.distribution(descriptor.package_name)
    matching = tuple(
        entry_point
        for entry_point in distribution.entry_points
        if entry_point.group == "tributo.algorithms"
        and entry_point.name == entry_point_name
    )
    if len(matching) != 1:
        raise ValueError(
            "installed distribution must expose exactly one matching "
            "tributo.algorithms entry point"
        )
    descriptor_module = matching[0].value.split(":", 1)[0]
    implementation_module = (
        descriptor.registration.implementation.implementation_ref.module
    )
    if descriptor_module == implementation_module:
        raise ValueError(
            "algorithm entry point must target a lightweight descriptor module, "
            "not the implementation module"
        )
    report = validate_algorithm_descriptor_conformance(
        descriptor,
        entry_point_name=entry_point_name,
        distribution_name=distribution.metadata["Name"],
    )
    return replace(
        report,
        implementation_loaded=implementation_module in sys.modules,
    )


def _load_identity_manifest(path: Path) -> Mapping[str, Mapping[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("algorithm identity manifest is unavailable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("algorithm identity manifest is malformed")
    entries = payload.get("entry_points")
    if payload.get("schema_version") != 1 or not isinstance(entries, Mapping):
        raise ValueError("algorithm identity manifest is malformed")
    normalized: dict[str, Mapping[str, str]] = {}
    for name, identity in entries.items():
        if not isinstance(name, str) or not isinstance(identity, Mapping):
            raise ValueError("algorithm identity manifest entry is malformed")
        required = {"distribution", "algorithm_id", "implementation_id"}
        if set(identity) != required or any(
            not isinstance(identity[field], str) or not identity[field]
            for field in required
        ):
            raise ValueError("algorithm identity manifest entry is incomplete")
        normalized[name] = {field: identity[field] for field in sorted(required)}
    return normalized


def _distribution_name(entry_point: importlib.metadata.EntryPoint) -> str:
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        raise ValueError("algorithm Entry Point has no owning distribution")
    name = distribution.metadata.get("Name")
    if not isinstance(name, str) or not name:
        raise ValueError("algorithm Entry Point distribution has no name")
    return canonicalize_name(name)


def _run_installed_conformance(
    *,
    distribution_prefix: str,
    expected_count: int,
    identity_manifest: Path,
    required_contracts: Sequence[str],
    forbidden_imports: Sequence[str],
) -> tuple[AlgorithmPackageConformanceReport, ...]:
    """Validate the installed algorithm Wheels without repository test imports."""
    if not distribution_prefix or expected_count < 1:
        raise ValueError("distribution prefix and positive expected count are required")
    manifest = _load_identity_manifest(identity_manifest)
    entry_points = tuple(
        entry_point
        for entry_point in importlib.metadata.entry_points(group="tributo.algorithms")
        if _distribution_name(entry_point).startswith(
            canonicalize_name(distribution_prefix)
        )
    )
    if len(entry_points) != expected_count or set(manifest) != {
        entry_point.name for entry_point in entry_points
    }:
        raise ValueError("installed algorithm Entry Point set does not match manifest")
    required = tuple(required_contracts)
    if tuple(dict.fromkeys(required)) != required or any(
        name not in {"config", "input", "output", "coverage"} for name in required
    ):
        raise ValueError("required contract kinds are invalid")
    reports: list[AlgorithmPackageConformanceReport] = []
    for entry_point in sorted(entry_points, key=lambda item: item.name):
        descriptor = entry_point.load()
        report = validate_installed_algorithm_package(
            descriptor,
            entry_point_name=entry_point.name,
        )
        bindings = descriptor.registration.contract_bindings
        if bindings is None or any(
            getattr(bindings, name) is None for name in required
        ):
            raise ValueError(
                f"algorithm {entry_point.name!r} is missing required contracts"
            )
        actual_identity = {
            "algorithm_id": report.algorithm_id,
            "distribution": canonicalize_name(report.distribution),
            "implementation_id": report.implementation_id,
        }
        if actual_identity != dict(manifest[entry_point.name]):
            raise ValueError(
                f"algorithm {entry_point.name!r} identity does not match manifest"
            )
        reports.append(report)
    imported = sorted(
        name
        for name in forbidden_imports
        if name in sys.modules
        or any(module.startswith(f"{name}.") for module in sys.modules)
    )
    if imported:
        raise ValueError(f"descriptor discovery imported forbidden modules: {imported}")
    return tuple(reports)


__all__ = [
    "AlgorithmPackageConformanceReport",
    "validate_algorithm_descriptor_conformance",
    "validate_installed_algorithm_package",
]
