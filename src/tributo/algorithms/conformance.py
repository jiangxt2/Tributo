"""Reusable public Conformance Testkit for installed algorithm Wheels."""

from __future__ import annotations

import importlib.metadata
import sys
from dataclasses import dataclass, replace

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


__all__ = [
    "AlgorithmPackageConformanceReport",
    "validate_algorithm_descriptor_conformance",
    "validate_installed_algorithm_package",
]
