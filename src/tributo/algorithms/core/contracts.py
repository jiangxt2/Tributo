"""Runtime resolution for lightweight algorithm contract validators."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from tributo._common.immutable import deep_thaw
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    ContractBinding,
    canonical_digest,
)
from tributo.algorithms.spi.contracts import AlgorithmContractValidator
from tributo.util.annotations import DeveloperAPI


def _load_validator(binding: ContractBinding) -> AlgorithmContractValidator:
    reference = binding.validator_ref
    try:
        value: object = importlib.import_module(reference.module)
        for segment in reference.qualname.split("."):
            value = getattr(value, segment)
        validator = value() if isinstance(value, type) else value
    except Exception as exc:
        raise AlgorithmConfigurationError(
            f"cannot load contract validator for {binding.contract_id!r}"
        ) from exc
    if not isinstance(validator, AlgorithmContractValidator):
        raise AlgorithmConfigurationError(
            f"contract validator for {binding.contract_id!r} does not implement "
            "AlgorithmContractValidator"
        )
    if validator.api_version != 1:
        raise AlgorithmConfigurationError(
            f"unsupported contract validator api_version for {binding.contract_id!r}"
        )
    if validator.schema_digest != binding.schema_digest:
        raise AlgorithmConfigurationError(
            f"contract validator schema digest mismatch for {binding.contract_id!r}"
        )
    return validator


@DeveloperAPI
def validate_contract_binding(binding: ContractBinding) -> None:
    """Load one selected lightweight Validator and verify its bound schema."""
    _load_validator(binding)


@DeveloperAPI
def validate_contract_value(
    binding: ContractBinding,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonicalize one contract value after algorithm selection."""
    try:
        validated = _load_validator(binding).validate(value)
    except AlgorithmConfigurationError:
        raise
    except Exception as exc:
        raise AlgorithmConfigurationError(
            f"contract validation failed for {binding.contract_id!r}"
        ) from exc
    if not isinstance(validated, Mapping):
        raise AlgorithmConfigurationError(
            f"contract validator for {binding.contract_id!r} must return a mapping"
        )
    normalized = dict(deep_thaw(validated))
    canonical_digest(normalized)
    return normalized


__all__ = ["validate_contract_binding", "validate_contract_value"]
