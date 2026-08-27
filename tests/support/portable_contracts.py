"""Lightweight contract validators used by portable algorithm tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ConfigValidator:
    """Validate and normalize one small algorithm configuration."""

    api_version = 1
    schema_digest = "a" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        unknown = set(value) - {"threshold"}
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        threshold = value.get("threshold", 0.5)
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise ValueError("threshold must be numeric")
        return {"threshold": float(threshold)}


class InputValidator:
    """Require a matching binding and descriptor pair."""

    api_version = 1
    schema_digest = "b" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        descriptors = value.get("descriptors")
        if not isinstance(bindings, list) or not isinstance(descriptors, Mapping):
            raise ValueError("input contract requires bindings and descriptors")
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise ValueError("input bindings must be mappings")
            role = binding.get("name")
            descriptor = descriptors.get(role)
            if not isinstance(descriptor, Mapping):
                raise ValueError(f"input descriptor is missing for role {role!r}")
            if binding.get("reference") != descriptor.get("reference"):
                raise ValueError("input binding and descriptor references differ")
        return value


class OutputValidator:
    """Accept successful portable results only."""

    api_version = 1
    schema_digest = "c" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if value.get("status") != "succeeded":
            raise ValueError("output contract requires a successful result")
        return value


class RejectingOutputValidator(OutputValidator):
    """Reject a successful result to prove coordinator enforcement."""

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        del value
        raise ValueError("output rejected by contract")


class CoverageValidator:
    """Validate bounded execution coverage evidence."""

    api_version = 1
    schema_digest = "d" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if value.get("input_complete") is not True:
            raise ValueError("coverage contract requires complete input evidence")
        return value


class WrongDigestValidator(ConfigValidator):
    """Expose a mismatched schema digest for fail-closed tests."""

    schema_digest = "f" * 64
