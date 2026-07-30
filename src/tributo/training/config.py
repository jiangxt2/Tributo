"""Configuration validation, merging, and data-source resolution.

Implements the effective-config pipeline::

    default_config + user_config
    → merge_nested
    → apply_dot_overrides (optional, for Tune trials)
    → validate_and_normalize_config (Pydantic)
    → validate_execution_config (data sourcing)
    → canonical config dict ready for training

Also provides ``TrainingDataConfig`` (two-layer data envelope) and
a legacy adapter that partitions flat ``data`` dicts into canonical
``data.source`` + training fields.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from pydantic import Field, ValidationError

from tributo._common.config import StrictConfigModel
from tributo._common.immutable import deep_thaw
from tributo.data.source_config import (
    LegacyConfigNormalizer,
    SourceConfig,
)
from tributo.exceptions import JobConfigurationError
from tributo.training.algorithm_spec import AlgorithmSpec, DataLoadingMode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training data envelope
# ---------------------------------------------------------------------------


class TrainingDataConfig(StrictConfigModel):
    """Two-layer data configuration for canonical trainers.

    ``source`` carries the storage location; training semantics
    (e.g. label column, feature list) live on algorithm-specific
    subclasses or other config sections.
    """

    source: SourceConfig | None = Field(
        default=None,
        description="Storage source. None when datasets are supplied externally.",
    )


# ---------------------------------------------------------------------------
# Pydantic-level validation
# ---------------------------------------------------------------------------


def validate_and_normalize_config(
    spec: AlgorithmSpec,
    raw_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate *raw_config* against ``spec.config_model`` and return a dict.

    Returns ``dict(raw_config)`` when ``config_model`` is ``None``
    (legacy algorithms without a declared schema).
    """
    config = deep_thaw(raw_config)
    if spec.config_model is None:
        return config
    try:
        model = spec.config_model.model_validate(config)
        return model.model_dump(mode="python")
    except ValidationError as e:
        raise JobConfigurationError(
            f"Config validation failed for algorithm '{spec.name}':\n"
            f"{json.dumps(e.errors(), indent=2, default=str)}"
        ) from e


# ---------------------------------------------------------------------------
# Execution preconditions (zero I/O)
# ---------------------------------------------------------------------------


def validate_execution_config(
    spec: AlgorithmSpec,
    config: dict[str, Any],
    *,
    datasets_supplied: bool,
) -> None:
    """Validate that *config* is executable in the current call context.

    Rules:
    - ``CANONICAL_TRAINER`` (PU): ``data.source`` is always required.
    - ``CANONICAL_DRIVER`` (XGBoost / DNN): ``data.source`` required
      unless the caller already supplies ``datasets``.
    - ``LEGACY_DRIVER``: no source check (old plugins handle their own data).
    """
    if spec.data_loading == DataLoadingMode.CANONICAL_TRAINER:
        _require_data_source(config)
    elif spec.data_loading == DataLoadingMode.CANONICAL_DRIVER:
        if not datasets_supplied:
            _require_data_source(config)


def _require_data_source(config: dict[str, Any]) -> None:
    data = config.get("data")
    if not isinstance(data, dict):
        raise JobConfigurationError(
            "Training config requires 'data' section with a 'source' field."
        )
    source = data.get("source")
    if source is None:
        raise JobConfigurationError(
            "Training config requires 'data.source'. "
            "Set 'datasets_supplied=True' if datasets are provided externally."
        )


# ---------------------------------------------------------------------------
# Effective config pipeline
# ---------------------------------------------------------------------------


def build_effective_config(
    spec: AlgorithmSpec,
    user_config: Mapping[str, Any],
    *,
    dot_overrides: Mapping[str, Any] | None = None,
    datasets_supplied: bool = False,
) -> dict[str, Any]:
    """Merge, validate and return the canonical config dict ready for training.

    This is the single entry-point that produces a config the Runner /
    CLI can pass to a trainer constructor.  Order::

        default_config + user → merge → dot_overrides → validate → execute-check
    """
    merged = merge_nested(spec.default_config, user_config)
    if dot_overrides:
        merged = apply_dot_overrides(merged, dot_overrides)
    normalized = validate_and_normalize_config(spec, merged)
    validate_execution_config(spec, normalized, datasets_supplied=datasets_supplied)
    return normalized


# ---------------------------------------------------------------------------
# Merge utilities
# ---------------------------------------------------------------------------


def merge_nested(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge two nested mappings.  Neither input is modified."""
    result = deep_thaw(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = merge_nested(result[key], value)
        else:
            result[key] = deep_thaw(value)
    return result


def apply_dot_overrides(
    config: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply dot-path overrides to *config*.  Neither input is modified.

    Example::

        apply_dot_overrides(
            {"model": {"max_depth": 3}},
            {"model.max_depth": 7},
        )  →  {"model": {"max_depth": 7}}

    Raises:
        JobConfigurationError: Invalid dot-path or non-mapping intermediate.
    """
    result = deep_thaw(config)
    for path, value in overrides.items():
        _validate_dot_path(path)
        keys = path.split(".")
        target: dict[str, Any] = result
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            elif not isinstance(target[key], dict):
                raise JobConfigurationError(
                    f"Cannot descend through non-mapping path segment "
                    f"{key!r} (path={path!r})"
                )
            target = target[key]
        target[keys[-1]] = deep_thaw(value)
    return result


def _validate_dot_path(path: str) -> None:
    if not isinstance(path, str) or not path:
        raise JobConfigurationError(
            f"Invalid dot-path: {path!r} (must be a non-empty string)"
        )
    if ".." in path or path.startswith(".") or path.endswith("."):
        raise JobConfigurationError(f"Invalid dot-path: {path!r} (empty segment)")


# ---------------------------------------------------------------------------
# Data source resolution
# ---------------------------------------------------------------------------


def resolve_data_source(
    spec: AlgorithmSpec,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return a canonical source dict for data loading.

    - ``CANONICAL_*``: reads ``config["data"]["source"]``, re-validates
      with ``TypeAdapter(SourceConfig)``.
    - ``LEGACY_DRIVER``: normalises the flat ``config["data"]`` via
      the legacy adapter, dumps back to canonical.

    Raises:
        JobConfigurationError: Missing or invalid source.
    """
    data = config.get("data")
    if not isinstance(data, dict):
        raise JobConfigurationError("config must contain a 'data' section")

    if spec.data_loading in (
        DataLoadingMode.CANONICAL_DRIVER,
        DataLoadingMode.CANONICAL_TRAINER,
    ):
        source = data.get("source")
        if source is None:
            raise JobConfigurationError(
                f"Algorithm {spec.name!r} requires 'data.source' "
                f"(data_loading={spec.data_loading.value})"
            )
        # Re-validate through TypeAdapter to catch source-level errors early.
        from pydantic import TypeAdapter

        adapter = TypeAdapter(SourceConfig)
        try:
            validated = adapter.validate_python(source)
        except ValidationError as e:
            raise JobConfigurationError(
                f"Invalid data.source for {spec.name!r}:\n"
                f"{json.dumps(e.errors(), indent=2, default=str)}"
            ) from e
        return validated.model_dump(mode="python")

    # LEGACY_DRIVER
    cfg = LegacyConfigNormalizer.normalize(data)
    return cfg.model_dump(mode="python")
