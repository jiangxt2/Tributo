"""Unit tests for deterministic registration and side-effect-free planning."""

from __future__ import annotations

import sys
from dataclasses import replace

import pytest

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmOperation,
    AlgorithmRegistration,
    AlgorithmResolutionError,
    BackendInputCompatibility,
    InputBinding,
    QualifiedReference,
    ResolvedInputDescriptor,
    RuntimeTopology,
    canonical_digest,
)
from tributo.algorithms.core import (
    AlgorithmPlanner,
    AlgorithmRegistrationRegistry,
)
from tributo.algorithms.input import FakeInputResolver
from tributo.algorithms.spi import InputResolutionContext
from tributo.training.algorithm_spec import AlgorithmStatus

from .conftest import (
    function_registration,
    request_for,
    sklearn_registration,
)


def test_planner_is_deterministic_and_does_not_import_implementation() -> None:
    module = "tests.support.portable_algorithms"
    sys.modules.pop(module, None)
    modules_before = frozenset(sys.modules)
    registry = AlgorithmRegistrationRegistry()
    registry.register(sklearn_registration())
    resolver = FakeInputResolver()
    planner = AlgorithmPlanner(registry, {resolver.resolver_id: resolver})
    request = request_for(
        "external_sklearn",
        AlgorithmOperation.FIT,
        config={"C": 1.0, "max_iter": 100},
    )

    first = planner.plan(request)
    second = planner.plan(request)

    assert first == second
    assert first.plan_id == second.plan_id
    assert len(first.config_digest) == 64
    assert first.to_json() == second.to_json()
    assert module not in sys.modules
    newly_imported = set(sys.modules) - modules_before
    assert not any(
        name == "sklearn" or name.startswith("sklearn.") for name in newly_imported
    )


def test_plan_integrity_detects_tampering() -> None:
    registry = AlgorithmRegistrationRegistry()
    registry.register(function_registration())
    resolver = FakeInputResolver()
    plan = AlgorithmPlanner(registry, {resolver.resolver_id: resolver}).plan(
        request_for("external_function", AlgorithmOperation.FIT)
    )

    plan.validate_integrity()
    with pytest.raises(AlgorithmConfigurationError, match="digest"):
        replace(plan, plan_id="f" * 64).validate_integrity()


def test_plan_records_backend_and_resolver_input_compatibility_facts() -> None:
    registry = AlgorithmRegistrationRegistry()
    registry.register(sklearn_registration())
    resolver = FakeInputResolver()

    plan = AlgorithmPlanner(registry, {resolver.resolver_id: resolver}).plan(
        request_for("external_sklearn", AlgorithmOperation.FIT)
    )
    payload = plan.to_dict()

    assert payload["implementation"]["input_compatibility"] == {
        "accepted_input_views": [
            "daft_dataframe",
            "materialized_tabular",
            "ray_data",
        ],
        "accepted_ingestion_engines": [
            "tributo.daft",
            "tributo.fake_tabular",
            "tributo.ray_data",
        ],
        "required_input_capabilities": ["materializable"],
        "supported_explicit_adapters": [
            "tributo.algorithms.input.fake:prepare_input",
            "tributo.integrations.algorithm_inputs.ingestion:prepare_daft_input",
            "tributo.integrations.algorithm_inputs.ingestion:prepare_ingestion_input",
            "tributo.integrations.algorithm_inputs.ingestion:prepare_ray_data_input",
        ],
        "distribution_policy": ["framework_managed", "single_worker"],
    }
    assert payload["input_descriptor"]["engine_id"] == "tributo.fake_tabular"
    assert payload["input_descriptor"]["input_capabilities"] == [
        "materializable",
        "shardable",
    ]


@pytest.mark.parametrize(
    ("dimension", "message"),
    [
        ("view", "input view"),
        ("engine", "ingestion engine"),
        ("capability", "required capabilities"),
        ("backend_adapter", "not declared"),
        ("resolver_adapter", "incompatible"),
    ],
)
def test_planner_fails_closed_for_each_input_compatibility_dimension(
    dimension: str,
    message: str,
) -> None:
    registration = sklearn_registration()
    compatibility = registration.implementation.input_compatibility
    if dimension == "view":
        compatibility = replace(
            compatibility,
            accepted_input_views=("tensor_batches",),
        )
    elif dimension == "engine":
        compatibility = replace(
            compatibility,
            accepted_ingestion_engines=("tests.other_engine",),
        )
    elif dimension == "capability":
        compatibility = replace(
            compatibility,
            required_input_capabilities=("random_access",),
        )
    elif dimension == "backend_adapter":
        compatibility = replace(
            compatibility,
            supported_explicit_adapters=(
                QualifiedReference.parse("tests.input:prepare"),
            ),
        )
    registration = replace(
        registration,
        implementation=replace(
            registration.implementation,
            input_compatibility=compatibility,
        ),
    )

    class ModifiedResolver(FakeInputResolver):
        def describe(
            self,
            binding: InputBinding,
            context: InputResolutionContext,
        ) -> ResolvedInputDescriptor:
            descriptor = super().describe(binding, context)
            if dimension == "resolver_adapter":
                return replace(
                    descriptor,
                    compatible_worker_input_adapter_refs=("tests.input:prepare",),
                )
            return descriptor

    registry = AlgorithmRegistrationRegistry()
    registry.register(registration)
    resolver = ModifiedResolver()

    with pytest.raises(AlgorithmConfigurationError, match=message):
        AlgorithmPlanner(registry, {resolver.resolver_id: resolver}).plan(
            request_for("external_sklearn", AlgorithmOperation.FIT)
        )


def test_registration_rejects_topology_outside_backend_distribution_policy() -> None:
    registration = function_registration(data_parallel=True)
    compatibility = BackendInputCompatibility(
        accepted_input_views=("materialized_tabular",),
        accepted_ingestion_engines=("tributo.fake_tabular",),
        required_input_capabilities=("materializable",),
        supported_explicit_adapters=(registration.runtime.worker_input_adapter_ref,),
        distribution_policy=(RuntimeTopology.SINGLE_WORKER,),
    )

    with pytest.raises(AlgorithmConfigurationError, match="distribution_policy"):
        replace(
            registration,
            implementation=replace(
                registration.implementation,
                input_compatibility=compatibility,
            ),
        )


def test_new_input_engine_and_view_require_only_declared_contract_facts() -> None:
    registration = sklearn_registration()
    registration = replace(
        registration,
        implementation=replace(
            registration.implementation,
            input_compatibility=replace(
                registration.implementation.input_compatibility,
                accepted_input_views=("vendor_columnar",),
                accepted_ingestion_engines=("vendor.engine",),
            ),
        ),
    )

    class VendorResolver(FakeInputResolver):
        def describe(
            self,
            binding: InputBinding,
            context: InputResolutionContext,
        ) -> ResolvedInputDescriptor:
            return replace(
                super().describe(binding, context),
                engine_id="vendor.engine",
                view_kind="vendor_columnar",
            )

    registry = AlgorithmRegistrationRegistry()
    registry.register(registration)
    resolver = VendorResolver()

    plan = AlgorithmPlanner(registry, {resolver.resolver_id: resolver}).plan(
        request_for("external_sklearn", AlgorithmOperation.FIT)
    )

    assert plan.input_descriptor.engine_id == "vendor.engine"
    assert plan.input_descriptor.view_kind == "vendor_columnar"


@pytest.mark.parametrize(
    ("registration", "topology", "worker_count", "framework_parallelism"),
    [
        (
            sklearn_registration(framework_managed=True),
            RuntimeTopology.FRAMEWORK_MANAGED,
            1,
            2,
        ),
        (
            function_registration(data_parallel=True),
            RuntimeTopology.DATA_PARALLEL,
            2,
            1,
        ),
    ],
)
def test_plan_digest_includes_explicit_runtime_topology(
    registration: AlgorithmRegistration,
    topology: RuntimeTopology,
    worker_count: int,
    framework_parallelism: int,
) -> None:
    registry = AlgorithmRegistrationRegistry()
    registry.register(registration)
    resolver = FakeInputResolver()
    algorithm = (
        "external_sklearn"
        if topology is RuntimeTopology.FRAMEWORK_MANAGED
        else "external_function"
    )

    plan = AlgorithmPlanner(registry, {resolver.resolver_id: resolver}).plan(
        request_for(algorithm, AlgorithmOperation.FIT)
    )
    runtime = plan.to_dict()["runtime"]

    assert runtime["topology"] == topology.value
    assert runtime["worker_count"] == worker_count
    assert runtime["framework_parallelism"] == framework_parallelism
    assert plan.plan_id == canonical_digest(plan.to_dict(include_plan_id=False))


def test_unknown_configuration_fails_before_input_description() -> None:
    class ExplodingResolver(FakeInputResolver):
        def describe(
            self,
            binding: InputBinding,
            context: InputResolutionContext,
        ) -> ResolvedInputDescriptor:
            del binding, context
            raise AssertionError("input resolver must not be called")

    registry = AlgorithmRegistrationRegistry()
    registry.register(sklearn_registration())
    resolver = ExplodingResolver()
    planner = AlgorithmPlanner(registry, {resolver.resolver_id: resolver})

    with pytest.raises(
        AlgorithmConfigurationError,
        match="undeclared key",
    ):
        planner.plan(
            request_for(
                "external_sklearn",
                AlgorithmOperation.FIT,
                config={"unknown": 1},
            )
        )


def test_ambiguous_implementations_fail_independently_of_order() -> None:
    first = replace(function_registration(), is_default=False)
    original_second = function_registration(
        "tests.support.portable_algorithms:failing_training_fragment",
    )
    second = replace(
        original_second,
        implementation=replace(
            original_second.implementation,
            implementation_id="tests.second",
        ),
        is_default=False,
    )

    for registrations in ((first, second), (second, first)):
        registry = AlgorithmRegistrationRegistry()
        for registration in registrations:
            registry.register(registration)
        with pytest.raises(AlgorithmResolutionError, match="ambiguous"):
            registry.resolve(
                algorithm="external_function",
                operation=AlgorithmOperation.FIT,
                implementation_id=None,
            )


def test_registry_rejects_conflicting_algorithm_facts() -> None:
    first = function_registration()
    original_second = function_registration(
        "tests.support.portable_algorithms:failing_training_fragment"
    )
    second = replace(
        original_second,
        implementation=replace(
            original_second.implementation,
            implementation_id="tests.second",
        ),
        spec=replace(original_second.spec, version="2.0.0"),
    )
    registry = AlgorithmRegistrationRegistry()
    registry.register(first)

    with pytest.raises(AlgorithmResolutionError, match="conflicting specs"):
        registry.register(second)


def test_deprecated_registration_warns_but_remains_resolvable() -> None:
    registration = function_registration()
    deprecated = replace(
        registration,
        spec=replace(
            registration.spec,
            status=AlgorithmStatus.DEPRECATED,
            deprecated_since="1.0.0",
            replacement="external_function_v2",
        ),
    )
    registry = AlgorithmRegistrationRegistry()
    registry.register(deprecated)

    with pytest.warns(FutureWarning, match="external_function_v2"):
        selected = registry.resolve(
            algorithm="external_function",
            operation=AlgorithmOperation.FIT,
            implementation_id=None,
        )

    assert selected is deprecated
