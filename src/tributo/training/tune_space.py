"""Search space parser.

Converts JSON-format search space definitions into Ray Tune-compatible param_space dictionaries.

Supported sampling types:
- uniform: Uniform distribution [lower, upper)
- loguniform: Log-uniform distribution [lower, upper)
- quniform: Quantized uniform distribution [lower, upper), step q
- qloguniform: Quantized log-uniform distribution [lower, upper), step q
- randint: Integer uniform distribution [lower, upper)
- lograndint: Integer log-uniform distribution [lower, upper)
- qrandint: Quantized integer uniform distribution [lower, upper], step q
- qlograndint: Quantized integer log-uniform distribution [lower, upper], step q
- choice: Discrete choice
- grid_search: Grid search
"""

from __future__ import annotations

from typing import Any

from ray import tune

from tributo.exceptions import JobConfigurationError
from tributo.util.annotations import PublicAPI

# Sampling type -> Ray Tune sampling function mapping
_SAMPLING_FN_MAP: dict[str, Any] = {
    "uniform": lambda spec: tune.uniform(spec["lower"], spec["upper"]),
    "loguniform": lambda spec: tune.loguniform(spec["lower"], spec["upper"]),
    "quniform": lambda spec: tune.quniform(spec["lower"], spec["upper"], spec["q"]),
    "qloguniform": lambda spec: tune.qloguniform(
        spec["lower"], spec["upper"], spec["q"]
    ),
    "randint": lambda spec: tune.randint(spec["lower"], spec["upper"]),
    "lograndint": lambda spec: tune.lograndint(spec["lower"], spec["upper"]),
    "qrandint": lambda spec: tune.qrandint(spec["lower"], spec["upper"], spec["q"]),
    "qlograndint": lambda spec: tune.qlograndint(
        spec["lower"], spec["upper"], spec["q"]
    ),
    "choice": lambda spec: tune.choice(spec["values"]),
    "grid_search": lambda spec: tune.grid_search(spec["values"]),
}


def _validate_spec(name: str, spec_type: str, spec: dict[str, Any]) -> None:
    """Validate sampling parameters against Ray Tune requirements.

    Args:
        name: Hyperparameter name.
        spec_type: Sampling type.
        spec: Raw spec dictionary.

    Raises:
        JobConfigurationError: When parameters fail validation.
    """
    if spec_type in {
        "uniform",
        "loguniform",
        "quniform",
        "qloguniform",
        "randint",
        "lograndint",
        "qrandint",
        "qlograndint",
    }:
        lower = spec.get("lower")
        upper = spec.get("upper")
        if lower is None or upper is None:
            raise JobConfigurationError(
                f"'{name}' ({spec_type}) requires 'lower' and 'upper'"
            )
        if lower >= upper:
            raise JobConfigurationError(
                f"'{name}' ({spec_type}) requires lower < upper, got {lower} >= {upper}"
            )
        if spec_type in {"quniform", "qloguniform", "qrandint", "qlograndint"}:
            q = spec.get("q")
            if q is None or q <= 0:
                raise JobConfigurationError(
                    f"'{name}' ({spec_type}) requires positive 'q', got {q}"
                )

    elif spec_type in {"choice", "grid_search"}:
        values = spec.get("values")
        if not isinstance(values, list) or len(values) == 0:
            raise JobConfigurationError(
                f"'{name}' ({spec_type}) requires non-empty list 'values'"
            )


@PublicAPI(stability="beta")
def parse_search_space(config_path: str) -> dict[str, Any]:
    """Parse a JSON search space definition into a Ray Tune param_space dictionary.

    Args:
        config_path: JSON file path.

    Returns:
        Ray Tune-compatible param_space dictionary.

    Raises:
        JobConfigurationError: Invalid JSON format or unsupported sampling type.

    Example:
        >>> # JSON content:
        >>> # {"search_space": {"learning_rate": {"type": "loguniform", "lower": 0.001, "upper": 0.1}}}
        >>> space = parse_search_space("tune_space.json")
    """
    import json
    from pathlib import Path

    path = Path(config_path)
    if path.suffix in {".yaml", ".yml"}:
        raise ValueError("YAML search space is no longer supported; please use JSON.")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict) or "search_space" not in raw:
        raise JobConfigurationError(
            "JSON must contain 'search_space' key. "
            f"Got keys: {list(raw.keys()) if isinstance(raw, dict) else type(raw)}"
        )

    search_space = raw["search_space"]
    if not isinstance(search_space, dict):
        raise JobConfigurationError(
            f"'search_space' must be a mapping, got {type(search_space)}"
        )

    param_space: dict[str, Any] = {}
    for name, spec in search_space.items():
        if not isinstance(spec, dict):
            raise JobConfigurationError(
                f"Search space spec for '{name}' must be a mapping, got {type(spec)}"
            )

        spec_type = spec.get("type")
        if spec_type not in _SAMPLING_FN_MAP:
            raise JobConfigurationError(
                f"Unsupported sampling type '{spec_type}' for '{name}'. "
                f"Supported: {list(_SAMPLING_FN_MAP.keys())}"
            )

        _validate_spec(name, spec_type, spec)
        param_space[name] = _SAMPLING_FN_MAP[spec_type](spec)

    return param_space
