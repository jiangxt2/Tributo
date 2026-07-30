"""Search-space intermediate representation with Ray and local adapters.

``parse_search_space()`` parses a JSON file into a ``SearchSpaceSpec``
— a Tributo-owned IR that is independent of any execution backend.

``to_ray_param_space()`` converts the IR to a ``ray.tune`` param_space
dict for distributed tuning.

``resolve_local_overrides()`` produces deterministic overrides for
``--local`` mode without touching Ray internals.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from tributo._common.immutable import deep_freeze, deep_thaw
from tributo.exceptions import JobConfigurationError
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel for "no default declared"
# ---------------------------------------------------------------------------


class _MissingDefault(Enum):
    TOKEN = "missing"


MISSING = _MissingDefault.TOKEN

# ---------------------------------------------------------------------------
# Search-kind taxonomy
# ---------------------------------------------------------------------------

SearchKind = Literal[
    "uniform",
    "loguniform",
    "quniform",
    "qloguniform",
    "randint",
    "lograndint",
    "qrandint",
    "qlograndint",
    "choice",
    "grid_search",
]

_DIST_KINDS: set[SearchKind] = {
    "uniform",
    "loguniform",
    "quniform",
    "qloguniform",
    "randint",
    "lograndint",
    "qrandint",
    "qlograndint",
}
_DISCRETE_KINDS: set[SearchKind] = {"choice", "grid_search"}

# ---------------------------------------------------------------------------
# Search-space IR
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class SearchParamSpec:
    """Specification of a single hyperparameter to tune."""

    path: str
    kind: SearchKind
    lower: int | float | None = None
    upper: int | float | None = None
    q: int | float | None = None
    values: tuple[Any, ...] = ()
    default: Any = MISSING

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", deep_freeze(self.values))
        if self.default is not MISSING:
            object.__setattr__(self, "default", deep_freeze(self.default))


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class SearchSpaceSpec:
    """Immutable collection of search parameters."""

    parameters: tuple[SearchParamSpec, ...]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
def parse_search_space(config_path: str) -> SearchSpaceSpec:
    """Parse a JSON search-space file into a ``SearchSpaceSpec``.

    Expected JSON structure::

        {"search_space": {"training.lr": {"type": "loguniform", ...}, ...}}

    Dot-path keys (``training.learning_rate``) target nested config fields.
    Paths starting with ``data`` or ``output`` are rejected — datasets are
    loaded once before trials, and output is managed by Runner parameters.
    """
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or "search_space" not in raw:
        raise JobConfigurationError("JSON must contain 'search_space' key.")
    search_space = raw["search_space"]
    if not isinstance(search_space, dict):
        raise JobConfigurationError(
            f"'search_space' must be a mapping, got {type(search_space).__name__}"
        )

    params: list[SearchParamSpec] = []
    paths_seen: set[str] = set()

    for path, spec in search_space.items():
        _validate_search_param_path(path)
        # Check prefix conflicts within the search space itself.
        _check_search_prefix_conflict(path, paths_seen)
        paths_seen.add(path)

        if not isinstance(spec, dict):
            raise JobConfigurationError(
                f"Search param {path!r}: spec must be a dict, got {type(spec).__name__}"
            )
        kind = spec.get("type")
        if kind not in _ALL_KINDS:
            raise JobConfigurationError(
                f"Search param {path!r}: unknown type {kind!r}. "
                f"Supported: {sorted(_ALL_KINDS)}"
            )
        param = _build_param(path, kind, spec)
        params.append(param)

    # Prefix conflict check across all parameters.
    _check_cross_param_prefixes(params)
    return SearchSpaceSpec(parameters=tuple(params))


_ALL_KINDS: set[str] = {
    "uniform",
    "loguniform",
    "quniform",
    "qloguniform",
    "randint",
    "lograndint",
    "qrandint",
    "qlograndint",
    "choice",
    "grid_search",
}


def _build_param(path: str, kind: str, spec: dict[str, Any]) -> SearchParamSpec:
    """Build a validated ``SearchParamSpec`` from raw dict fields."""
    lower = spec.get("lower")
    upper = spec.get("upper")
    q = spec.get("q")
    values = spec.get("values")
    default = spec.get("default", MISSING)

    if kind in _DIST_KINDS:
        if lower is None or upper is None:
            raise JobConfigurationError(
                f"Search param {path!r}: '{kind}' requires 'lower' and 'upper'"
            )
        if lower >= upper:
            raise JobConfigurationError(
                f"Search param {path!r}: lower ({lower}) must be < upper ({upper})"
            )
        if "q" in kind and q is None:
            raise JobConfigurationError(f"Search param {path!r}: '{kind}' requires 'q'")
        if default is not MISSING:
            if not (lower <= default <= upper):
                raise JobConfigurationError(
                    f"Search param {path!r}: default {default} not in "
                    f"[{lower}, {upper}]"
                )
            if "q" in kind and q is not None:
                # Use tolerance-safe check — float modulo can lose precision
                # (e.g. (0.3 - 0.0) % 0.1 ≈ 0.09999… instead of 0).
                offset = default - lower
                remainder = offset % q
                if remainder > 1e-9 and (q - remainder) > 1e-9:
                    raise JobConfigurationError(
                        f"Search param {path!r}: default {default} does not "
                        f"satisfy quantisation q={q} from lower={lower}"
                    )

    if kind in _DISCRETE_KINDS:
        if not values or not isinstance(values, list):
            raise JobConfigurationError(
                f"Search param {path!r}: '{kind}' requires non-empty 'values' list"
            )
        if default is not MISSING and default not in values:
            raise JobConfigurationError(
                f"Search param {path!r}: default {default} not in values={values}"
            )

    return SearchParamSpec(
        path=path,
        kind=kind,  # type: ignore[arg-type]
        lower=lower,
        upper=upper,
        q=q,
        values=tuple(values) if values else (),
        default=default,
    )


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def _validate_search_param_path(path: str) -> None:
    if not isinstance(path, str) or not path:
        raise JobConfigurationError(f"Invalid search path: {path!r}")
    if ".." in path or path.startswith(".") or path.endswith("."):
        raise JobConfigurationError(f"Search path {path!r}: empty segment")
    top = path.split(".", 1)[0]
    if top in ("data", "output"):
        raise JobConfigurationError(
            f"Search path {path!r}: tuning 'data' or 'output' is not allowed. "
            f"Datasets are loaded once; output is managed by Runner parameters."
        )


def _check_search_prefix_conflict(path: str, seen: set[str]) -> None:
    """Check *path* against already-seen paths for prefix conflicts."""
    segments = path.split(".")
    for i in range(1, len(segments) + 1):
        prefix = ".".join(segments[:i])
        if prefix in seen:
            raise JobConfigurationError(
                f"Search path {path!r} conflicts with already-defined "
                f"search prefix {prefix!r}"
            )
    # Reverse: already-seen paths that are prefixes of this one are caught
    # when the shorter path was added first (prefix in seen above handles that).
    # We also need: this path is a prefix of an already-seen path.
    for s in seen:
        if s.startswith(path + "."):
            raise JobConfigurationError(
                f"Search path {path!r} is a prefix of already-defined {s!r}"
            )


def _check_cross_param_prefixes(params: list[SearchParamSpec]) -> None:
    """Post-hoc prefix conflict check across all params."""
    paths = {p.path for p in params}
    for p in params:
        segments = p.path.split(".")
        for i in range(1, len(segments)):
            prefix = ".".join(segments[:i])
            if prefix in paths:
                raise JobConfigurationError(
                    f"Search param {p.path!r}: intermediate segment {prefix!r} "
                    f"is itself a search parameter — this would change the "
                    f"value type from mapping to scalar."
                )


# ---------------------------------------------------------------------------
# Conflict detection against effective config
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
def warn_search_space_conflicts(
    raw_user_config: Mapping[str, Any],
    space_spec: SearchSpaceSpec,
) -> None:
    """Warn when a search path also appears as a fixed value in *raw_user_config*.

    Only inspects *raw_user_config* (before defaults are applied), so
    values that only exist due to ``default_config`` or Pydantic defaults
    are not reported as conflicts.
    """
    for param in space_spec.parameters:
        value = _resolve_dot_path(raw_user_config, param.path)
        if value is not _MISSING_SENTINEL:
            # In distributed mode this is normal — Tune overrides the base.
            logger.warning(
                "Search param %r is also set in training config (%r); "
                "Tune will override the fixed value.",
                param.path,
                value,
            )


_MISSING_SENTINEL = object()


def _resolve_dot_path(mapping: Mapping[str, Any], path: str) -> Any:
    """Return the value at *path* in *mapping*, or a sentinel if missing."""
    keys = path.split(".")
    current: Any = mapping
    for key in keys:
        if isinstance(current, Mapping) and key in current:
            current = current[key]
        else:
            return _MISSING_SENTINEL
    return current


@PublicAPI(stability="beta")
def validate_search_targets(
    effective_config: dict[str, Any],
    space_spec: SearchSpaceSpec,
) -> None:
    """Validate that search paths target existing, non-scalar-compatible nodes.

    Raises:
        JobConfigurationError: A search path would replace a mapping with
            a scalar, or vice versa.
    """
    for param in space_spec.parameters:
        keys = param.path.split(".")
        target: Any = effective_config
        for key in keys[:-1]:
            if not isinstance(target, dict) or key not in target:
                raise JobConfigurationError(
                    f"Search path {param.path!r}: segment {key!r} not found "
                    f"in effective config"
                )
            target = target[key]
        leaf_key = keys[-1]
        if not isinstance(target, dict):
            raise JobConfigurationError(
                f"Search path {param.path!r}: cannot descend into "
                f"non-mapping value at parent"
            )
        if leaf_key in target and isinstance(target[leaf_key], Mapping):
            raise JobConfigurationError(
                f"Search path {param.path!r}: target is a mapping — "
                f"use a more specific path instead of replacing an entire section"
            )


# ---------------------------------------------------------------------------
# Ray adapter
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
def to_ray_param_space(spec: SearchSpaceSpec) -> dict[str, Any]:
    """Convert a ``SearchSpaceSpec`` to a ``ray.tune`` param_space dict."""

    result: dict[str, Any] = {}
    for param in spec.parameters:
        result[param.path] = _to_ray_domain(param)
    return result


# Lazy import — ray may not be installed in all environments.
_tune: Any = None


def _get_tune() -> Any:
    global _tune
    if _tune is None:
        import ray.tune as _tune

    return _tune


def _to_ray_domain(param: SearchParamSpec) -> Any:
    """Convert a single param to the corresponding Ray Tune domain object."""
    t = _get_tune()
    th = deep_thaw
    kind = param.kind
    if kind == "uniform":
        return t.uniform(th(param.lower), th(param.upper))
    if kind == "loguniform":
        return t.loguniform(th(param.lower), th(param.upper))
    if kind == "quniform":
        return t.quniform(th(param.lower), th(param.upper), th(param.q))
    if kind == "qloguniform":
        return t.qloguniform(th(param.lower), th(param.upper), th(param.q))
    if kind == "randint":
        return t.randint(th(param.lower), th(param.upper))
    if kind == "lograndint":
        return t.lograndint(th(param.lower), th(param.upper))
    if kind == "qrandint":
        return t.qrandint(th(param.lower), th(param.upper), th(param.q))
    if kind == "qlograndint":
        return t.qlograndint(th(param.lower), th(param.upper), th(param.q))
    if kind == "choice":
        return t.choice(th(param.values))
    if kind == "grid_search":
        return t.grid_search(th(param.values))
    raise JobConfigurationError(f"Unknown search kind: {kind!r}")


# ---------------------------------------------------------------------------
# Local overrides resolver
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
def resolve_local_overrides(
    space_spec: SearchSpaceSpec,
    effective_base: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve search parameters to deterministic values for ``--local`` mode.

    Priority:
    1. Explicit ``default`` in the search param.
    2. Existing value at the same path in *effective_base*.
    3. ``JobConfigurationError`` — the param cannot be resolved.

    Returns a dot-path → value dict ready for ``apply_dot_overrides``.
    """
    result: dict[str, Any] = {}
    for param in space_spec.parameters:
        if param.default is not MISSING:
            result[param.path] = deep_thaw(param.default)
        else:
            existing = _resolve_dot_path(effective_base, param.path)
            if existing is _MISSING_SENTINEL:
                raise JobConfigurationError(
                    f"Search param {param.path!r}: no default and not present "
                    f"in effective config. Add 'default' to the search space."
                )
            result[param.path] = deep_thaw(existing)
    return result
