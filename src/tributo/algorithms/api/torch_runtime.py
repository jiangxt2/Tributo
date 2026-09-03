"""Public, framework-neutral contracts for the Ray Train Torch runtime.

The module deliberately avoids importing Torch, Ray, or any algorithm package at
import time.  Runtime implementations live behind lazy integration modules and
consume these immutable values through the public API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generator, Protocol, cast, runtime_checkable
from urllib.parse import urlsplit

from tributo._common.immutable import deep_freeze
from tributo.algorithms.api.errors import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
)
from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    from tributo.algorithms.spi.execution import RuntimeExecutionEnvelope
    from tributo.algorithms.spi.torch import TorchStageContext


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise AlgorithmConfigurationError(
            "Torch runtime metadata must be canonical JSON"
        ) from exc


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite_number(value: object, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise AlgorithmConfigurationError(f"{field_name} must be finite")
    return float(value)


def _validate_scalar_numerator(value: object, field_name: str) -> None:
    """Validate a differentiable zero-dimensional tensor without importing Torch."""
    ndim = getattr(value, "ndim", None)
    if not isinstance(ndim, int) or isinstance(ndim, bool) or ndim != 0:
        raise AlgorithmConfigurationError(
            f"{field_name} must be a differentiable zero-dimensional Tensor"
        )
    detach = getattr(value, "detach", None)
    item = getattr(value, "item", None)
    if not callable(detach) or not callable(item):
        raise AlgorithmConfigurationError(
            f"{field_name} must provide callable detach() and item()"
        )
    try:
        detached = detach()
        detached_item = getattr(detached, "item", None)
        if not callable(detached_item) or not math.isfinite(float(detached_item())):
            raise AlgorithmConfigurationError(f"{field_name} must be finite")
    except (TypeError, ValueError) as exc:
        raise AlgorithmConfigurationError(f"{field_name} must be finite") from exc


def _digest_value(value: object, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AlgorithmConfigurationError(f"{field_name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AlgorithmConfigurationError(
            f"{field_name} must be a lower-case hexadecimal digest"
        ) from exc
    if value != value.lower():
        raise AlgorithmConfigurationError(f"{field_name} must be lower-case")
    return value


def _validate_bounded_evidence(
    value: object,
    *,
    path: str = "evidence",
    depth: int = 0,
) -> None:
    if depth > 4:
        raise AlgorithmConfigurationError("Torch evidence nesting is too deep")
    if isinstance(value, Mapping):
        if len(value) > 64:
            raise AlgorithmConfigurationError("Torch evidence has too many fields")
        for key, nested in value.items():
            if (
                not isinstance(key, str)
                or not key
                or len(key) > 128
                or re.fullmatch(r"[a-z][a-z0-9_.-]*", key) is None
                or key.casefold()
                in {
                    "path",
                    "uri",
                    "locator",
                    "credential",
                    "credentials",
                    "secret",
                    "secrets",
                }
                or key.casefold().endswith(("_path", "_uri", "_locator"))
            ):
                raise AlgorithmConfigurationError(
                    f"Torch evidence key {path}.{key!r} is invalid"
                )
            _validate_bounded_evidence(nested, path=f"{path}.{key}", depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            raise AlgorithmConfigurationError(
                f"Torch evidence list {path} is too large"
            )
        for index, nested in enumerate(value):
            _validate_bounded_evidence(nested, path=f"{path}[{index}]", depth=depth + 1)
        return
    if isinstance(value, str):
        if (
            len(value) > 1024
            or value.startswith(("/", "~/"))
            or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value)
        ):
            raise AlgorithmConfigurationError(f"Torch evidence value {path} is unsafe")
        return
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise AlgorithmConfigurationError(f"Torch evidence value {path} is not portable")


_RUN_TOKEN = re.compile(r"^[a-f0-9]{8,128}$")


def _run_component(value: str, field_name: str) -> str:
    """Return a canonical path-safe run component."""
    if not isinstance(value, str) or not value or _RUN_TOKEN.fullmatch(value) is None:
        raise AlgorithmConfigurationError(
            f"{field_name} must be canonical lower-case UUID/hex"
        )
    return value


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchStageRunIdentity:
    """Identity shared by one logical Torch Stage Run and its checkpoints."""

    run_id: str
    invocation_id: str
    stage_id: str
    torch_runtime_api_version: int
    algorithm: str
    implementation_id: str
    implementation_code_digest: str
    policy_digest: str
    execution_plan_digest: str
    plan_digest: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "invocation_id",
            "stage_id",
            "algorithm",
            "implementation_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise AlgorithmConfigurationError(f"{name} must be non-empty")
        if self.torch_runtime_api_version != 1:
            raise AlgorithmConfigurationError(
                "torch_runtime_api_version must be exactly 1"
            )
        _digest_value(self.implementation_code_digest, "implementation_code_digest")
        _digest_value(self.policy_digest, "policy_digest")
        _digest_value(self.execution_plan_digest, "execution_plan_digest")
        if self.plan_digest is not None:
            _digest_value(self.plan_digest, "plan_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "invocation_id": self.invocation_id,
            "stage_id": self.stage_id,
            "torch_runtime_api_version": self.torch_runtime_api_version,
            "algorithm": self.algorithm,
            "implementation_id": self.implementation_id,
            "implementation_code_digest": self.implementation_code_digest,
            "policy_digest": self.policy_digest,
            "execution_plan_digest": self.execution_plan_digest,
            "plan_digest": self.plan_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TorchStageRunIdentity":
        try:
            return cls(**dict(value))
        except (TypeError, KeyError) as exc:
            raise AlgorithmConfigurationError(
                "invalid Torch Stage Run identity"
            ) from exc

    @property
    def identity_digest(self) -> str:
        return _digest(self.to_dict())

    @property
    def run_config_name(self) -> str:
        return torch_run_config_name(self)


@PublicAPI(stability="alpha")
def torch_run_config_name(identity: TorchStageRunIdentity) -> str:
    """Build the Core-owned deterministic Ray ``RunConfig.name``."""
    if not isinstance(identity, TorchStageRunIdentity):
        raise AlgorithmConfigurationError("Torch run identity is required")
    run_id = _run_component(identity.run_id, "run_id")
    invocation_id = _run_component(identity.invocation_id, "invocation_id")
    stage_digest = hashlib.sha256(identity.stage_id.encode("utf-8")).hexdigest()[:16]
    name = (
        f"tributo-torch-v1-{run_id}-{invocation_id}-"
        f"{stage_digest}-{identity.identity_digest}"
    )
    if len(name) > 255 or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in name
    ):
        raise AlgorithmConfigurationError("Torch RunConfig.name is not filesystem-safe")
    return name


@PublicAPI(stability="alpha")
def claim_torch_run_directory(
    storage_path: str | os.PathLike[str],
    identity: TorchStageRunIdentity,
    *,
    status: str = "running",
) -> Path:
    """Atomically claim a Stage directory and its identity manifest.

    Existing directories are reusable only for the same identity and a
    retryable status.  Nothing is overwritten or silently renamed.
    """
    if status not in {"created", "running", "failed_retryable"}:
        raise AlgorithmConfigurationError("Torch run directory status is not retryable")
    root = Path(storage_path)
    raw_storage = str(storage_path)
    if "://" in raw_storage and not raw_storage.startswith("file://"):
        # Use the filesystem abstraction only for the identity manifest.  The
        # actual Ray checkpoint payload remains owned by Ray's storage backend.
        try:
            import pyarrow.fs as pafs

            filesystem, prefix = pafs.FileSystem.from_uri(raw_storage)
            run_path = f"{prefix.rstrip('/')}/{torch_run_config_name(identity)}"
            manifest_path = f"{run_path}/torch_identity_manifest.json"
            filesystem.create_dir(run_path, recursive=True)
            info = filesystem.get_file_info(manifest_path)
            payload = {
                "schema_version": 1,
                "snapshot_schema": 1,
                "identity": identity.to_dict(),
                "status": status,
                "stale": False,
            }
            if info.type is pafs.FileType.File:
                existing = json.loads(
                    filesystem.open_input_file(manifest_path).read().decode("utf-8")
                )
                if (
                    not isinstance(existing, Mapping)
                    or existing.get("identity") != identity.to_dict()
                    or existing.get("snapshot_schema", 1) != 1
                    or existing.get("stale", False)
                    or existing.get("status")
                    not in {"created", "running", "failed_retryable"}
                ):
                    raise AlgorithmExecutionError(
                        "Torch remote run directory identity collision or stale snapshot"
                    )
            elif info.type is pafs.FileType.NotFound:
                with filesystem.open_output_stream(manifest_path) as stream:
                    stream.write((_canonical_json(payload) + "\n").encode("utf-8"))
            else:
                raise AlgorithmExecutionError(
                    "Torch remote run identity manifest has an invalid type"
                )
            return Path(f"{raw_storage.rstrip('/')}/{torch_run_config_name(identity)}")
        except AlgorithmExecutionError:
            raise
        except (ImportError, OSError, TypeError, ValueError) as exc:
            raise AlgorithmExecutionError(
                "failed to claim remote Torch run identity manifest"
            ) from exc
    if not root.is_absolute():
        raise AlgorithmConfigurationError("Torch storage_path must be absolute")
    directory = root / torch_run_config_name(identity)
    manifest = directory / "torch_identity_manifest.json"
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if not manifest.is_file():
            raise AlgorithmExecutionError(
                "Torch run directory has no identity manifest"
            ) from None
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as exc:
            raise AlgorithmExecutionError(
                "Torch run identity manifest is damaged"
            ) from exc
        if (
            not isinstance(existing, Mapping)
            or existing.get("identity") != identity.to_dict()
        ):
            raise AlgorithmExecutionError(
                "Torch run directory identity collision"
            ) from None
        if existing.get("snapshot_schema", 1) != 1 or existing.get("stale", False):
            raise AlgorithmExecutionError(
                "Torch run directory contains a stale snapshot"
            ) from None
        if existing.get("status") not in {"created", "running", "failed_retryable"}:
            raise AlgorithmExecutionError(
                "Torch run directory is terminal or stale"
            ) from None
        return directory
    payload = {
        "schema_version": 1,
        "snapshot_schema": 1,
        "identity": identity.to_dict(),
        "status": status,
        "stale": False,
    }
    try:
        fd = os.open(manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(_canonical_json(payload))
            stream.write("\n")
    except OSError as exc:
        raise AlgorithmExecutionError(
            "failed to atomically write Torch identity manifest"
        ) from exc
    return directory


@PublicAPI(stability="alpha")
def validate_torch_retry_identity(
    descriptor: "TorchCheckpointDescriptor",
    identity: TorchStageRunIdentity,
    *,
    world_size: int,
) -> None:
    """Reject stale Ray retry snapshots before model state is deserialized."""
    if not isinstance(descriptor, TorchCheckpointDescriptor):
        raise AlgorithmExecutionError("retry checkpoint has no Torch descriptor")
    if descriptor.identity.to_dict() != identity.to_dict():
        raise AlgorithmExecutionError(
            "retry checkpoint identity does not match current Stage"
        )
    if descriptor.world_size != world_size:
        raise AlgorithmExecutionError(
            "retry checkpoint world size does not match current Stage"
        )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchLossContribution:
    """One ordinary differentiable loss numerator and its denominator."""

    numerator: object
    normalizer: float

    def __post_init__(self) -> None:
        _validate_scalar_numerator(self.numerator, "loss numerator")
        normalizer = _finite_number(self.normalizer, "loss normalizer")
        if normalizer < 0:
            raise AlgorithmConfigurationError("loss normalizer must be non-negative")
        object.__setattr__(self, "normalizer", normalizer)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchCompositeLossContribution:
    """Named differentiable components reduced by an algorithm-owned reducer."""

    schema_id: str
    differentiable_components: Mapping[str, object]
    normalizer_components: Mapping[str, float]
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.schema_id, str) or not self.schema_id:
            raise AlgorithmConfigurationError("composite loss schema_id is required")
        components: dict[str, object] = dict(self.differentiable_components)
        normalizers: dict[str, float] = dict(self.normalizer_components)
        if not components:
            raise AlgorithmConfigurationError(
                "composite loss requires differentiable components"
            )
        if not normalizers:
            raise AlgorithmConfigurationError(
                "composite loss requires normalizer components"
            )
        for name, value in normalizers.items():
            if not isinstance(name, str) or not name:
                raise AlgorithmConfigurationError(
                    "composite loss normalizers must be named and non-negative"
                )
            normalized_value = _finite_number(value, f"normalizer[{name}]")
            if normalized_value < 0:
                raise AlgorithmConfigurationError(
                    "composite loss normalizers must be named and non-negative"
                )
            normalizers[name] = normalized_value
        for component_name, component_value in components.items():
            if (
                not isinstance(component_name, str)
                or not component_name
                or component_value is None
            ):
                raise AlgorithmConfigurationError(
                    "differentiable component names and values are required"
                )
            _validate_scalar_numerator(component_value, f"component[{component_name}]")
        object.__setattr__(
            self, "differentiable_components", MappingProxyType(components)
        )
        object.__setattr__(self, "normalizer_components", MappingProxyType(normalizers))
        _validate_bounded_evidence(self.evidence)
        object.__setattr__(self, "evidence", deep_freeze(self.evidence))


TorchStepLoss = TorchLossContribution | TorchCompositeLossContribution


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchMetricContribution:
    """A metric numerator/denominator pair reduced by Core."""

    numerator: float
    normalizer: float

    def __post_init__(self) -> None:
        numerator = _finite_number(self.numerator, "metric numerator")
        normalizer = _finite_number(self.normalizer, "metric normalizer")
        if normalizer < 0:
            raise AlgorithmConfigurationError("metric normalizer must be non-negative")
        if normalizer == 0 and numerator != 0:
            raise AlgorithmConfigurationError(
                "zero metric normalizer requires a zero numerator"
            )
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "normalizer", normalizer)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchAccumulationWindow:
    """State describing one optimizer accumulation window."""

    index: int
    expected_micro_batches: int
    observed_micro_batches: int = 0
    normalizer_total: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.index, int)
            or isinstance(self.index, bool)
            or not isinstance(self.expected_micro_batches, int)
            or isinstance(self.expected_micro_batches, bool)
            or not isinstance(self.observed_micro_batches, int)
            or isinstance(self.observed_micro_batches, bool)
            or self.index < 0
            or self.expected_micro_batches < 1
        ):
            raise AlgorithmConfigurationError("invalid accumulation window")
        if not 0 <= self.observed_micro_batches <= self.expected_micro_batches:
            raise AlgorithmConfigurationError("invalid observed micro-batch count")
        total = _finite_number(self.normalizer_total, "accumulation normalizer total")
        if total < 0:
            raise AlgorithmConfigurationError(
                "accumulation normalizer total must be non-negative"
            )
        object.__setattr__(self, "normalizer_total", total)

    def add(self, normalizer: float) -> "TorchAccumulationWindow":
        """Return the next immutable window state after one micro-batch."""
        value = _finite_number(normalizer, "loss normalizer")
        if value < 0:
            raise AlgorithmConfigurationError("loss normalizer must be non-negative")
        if self.observed_micro_batches >= self.expected_micro_batches:
            raise AlgorithmExecutionError("accumulation window is already complete")
        return TorchAccumulationWindow(
            index=self.index,
            expected_micro_batches=self.expected_micro_batches,
            observed_micro_batches=self.observed_micro_batches + 1,
            normalizer_total=self.normalizer_total + value,
        )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchBackwardContext:
    """Public callbacks supplied by the Core Torch loop to the backward helper."""

    world_size: int
    backward: Any
    reduce_normalizer: Any
    finalize_window: Any
    reduce_window_normalizer: Any | None = None
    compose_composite: Any | None = None

    def __post_init__(self) -> None:
        if self.world_size < 1 or not callable(self.backward):
            raise AlgorithmConfigurationError("invalid Torch backward context")
        if not callable(self.reduce_normalizer) or not callable(self.finalize_window):
            raise AlgorithmConfigurationError(
                "Torch backward callbacks must be callable"
            )
        if self.reduce_window_normalizer is not None and not callable(
            self.reduce_window_normalizer
        ):
            raise AlgorithmConfigurationError(
                "Torch window normalizer callback must be callable"
            )
        if self.compose_composite is not None and not callable(self.compose_composite):
            raise AlgorithmConfigurationError(
                "Torch composite backward callback must be callable"
            )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchBackwardResult:
    """Result returned after one loss contribution is submitted."""

    local_normalizer: float
    global_normalizer: float
    window_complete: bool


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchMetricPolicy:
    """Explicit metric reducer names keyed by metric identity."""

    reducers: Mapping[str, str]

    def __post_init__(self) -> None:
        normalized = dict(self.reducers)
        if any(not isinstance(k, str) or not k for k in normalized):
            raise AlgorithmConfigurationError("metric reducer names are required")
        object.__setattr__(self, "reducers", MappingProxyType(normalized))


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchMetricReductionContext:
    """Core callbacks used by the metric reduction helper."""

    reduce: Any

    def __post_init__(self) -> None:
        if not callable(self.reduce):
            raise AlgorithmConfigurationError("metric reduction callback is required")


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchMetricReductionResult:
    """Reduced metric values and their evidence."""

    values: Mapping[str, float]
    evidence: Mapping[str, object] = field(default_factory=dict)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchGlobalLossContext:
    """Context passed to an algorithm-owned global loss reducer."""

    world_size: int
    policy_digest: str
    execution_plan_digest: str
    config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.world_size, int)
            or isinstance(self.world_size, bool)
            or self.world_size < 1
        ):
            raise AlgorithmConfigurationError(
                "Torch global loss world_size must be positive"
            )
        _digest_value(self.policy_digest, "policy_digest")
        _digest_value(self.execution_plan_digest, "execution_plan_digest")
        object.__setattr__(self, "config", deep_freeze(self.config))


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchCompositeGlobalState:
    """Names-preserving detached component state produced by Core AllReduce."""

    components: Mapping[str, float]
    normalizers: Mapping[str, float]

    def __post_init__(self) -> None:
        components = dict(self.components)
        normalizers = dict(self.normalizers)
        if not components or not normalizers:
            raise AlgorithmConfigurationError(
                "global component and normalizer state are required"
            )
        for name, value in components.items():
            if not isinstance(name, str) or not name:
                raise AlgorithmConfigurationError("global component names are required")
            components[name] = _finite_number(value, f"global component[{name}]")
        for name, value in normalizers.items():
            if not isinstance(name, str) or not name:
                raise AlgorithmConfigurationError(
                    "global normalizer names are required"
                )
            normalizers[name] = _finite_number(value, f"global normalizer[{name}]")
            if value < 0:
                raise AlgorithmConfigurationError(
                    f"global normalizer[{name}] must be non-negative"
                )
        object.__setattr__(self, "components", MappingProxyType(components))
        object.__setattr__(self, "normalizers", MappingProxyType(normalizers))


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchGlobalLossReduction:
    """Deterministic reducer output shared by all ranks."""

    status: str
    coefficients: Mapping[str, float] = field(default_factory=dict)
    branch: str | None = None
    evidence: Mapping[str, object] = field(default_factory=dict)
    failure_code: str | None = None
    metrics: Mapping[str, TorchMetricContribution] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"accepted", "rejected"}:
            raise AlgorithmConfigurationError("loss reduction status is invalid")
        for name in ("branch", "failure_code"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str)
                or not value
                or len(value) > 128
                or re.fullmatch(r"[a-z][a-z0-9_.-]*", value) is None
            ):
                raise AlgorithmConfigurationError(
                    f"loss reduction {name} must be a bounded namespaced value"
                )
        coefficients = dict(self.coefficients)
        evidence = dict(self.evidence)
        metrics = dict(self.metrics)
        if self.status == "accepted":
            if not coefficients or self.failure_code is not None:
                raise AlgorithmConfigurationError(
                    "accepted loss reduction requires coefficients"
                )
            for name, value in coefficients.items():
                coefficients[name] = _finite_number(value, f"coefficient[{name}]")
        elif coefficients or not self.failure_code:
            raise AlgorithmConfigurationError(
                "rejected loss reduction requires a failure code and no coefficients"
            )
        if any(
            not isinstance(value, TorchMetricContribution) for value in metrics.values()
        ):
            raise AlgorithmConfigurationError("loss reduction metrics are invalid")
        if len(evidence) > 64 or any(
            not isinstance(name, str)
            or not name
            or len(name) > 128
            or re.fullmatch(r"[a-z][a-z0-9_.-]*", name) is None
            or not isinstance(value, (str, int, float, bool, type(None)))
            or (isinstance(value, str) and len(value) > 1024)
            or (isinstance(value, float) and not math.isfinite(value))
            for name, value in evidence.items()
        ):
            raise AlgorithmConfigurationError(
                "loss reduction evidence must be bounded JSON scalars"
            )
        object.__setattr__(self, "coefficients", MappingProxyType(coefficients))
        object.__setattr__(self, "evidence", MappingProxyType(evidence))
        object.__setattr__(self, "metrics", MappingProxyType(metrics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "coefficients": dict(self.coefficients),
            "branch": self.branch,
            "evidence": dict(self.evidence),
            "failure_code": self.failure_code,
            "metrics": {
                name: {
                    "numerator": value.numerator,
                    "normalizer": value.normalizer,
                }
                for name, value in self.metrics.items()
            },
        }


@runtime_checkable
@PublicAPI(stability="alpha")
class TorchGlobalLossReducer(Protocol):
    """Algorithm-owned deterministic global loss reducer."""

    api_version: int
    reducer_id: str
    component_schema_id: str
    code_digest: str

    def reduce(
        self,
        config: Mapping[str, object],
        global_state: TorchCompositeGlobalState,
        context: TorchGlobalLossContext,
    ) -> TorchGlobalLossReduction: ...


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchPreflightTokenData:
    """Immutable identity data produced by Torch preflight."""

    run_id: str
    invocation_id: str
    algorithm: str
    implementation_ref: str
    implementation_code_digest: str
    policy_digest: str
    execution_plan_digest: str
    runtime_id: str
    reducer_identity: str | None = None
    plan_digest: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "invocation_id",
            "algorithm",
            "implementation_ref",
            "runtime_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise AlgorithmConfigurationError(f"{name} must be non-empty")
        _digest_value(self.implementation_code_digest, "implementation_code_digest")
        _digest_value(self.policy_digest, "policy_digest")
        _digest_value(self.execution_plan_digest, "execution_plan_digest")
        if self.reducer_identity is not None and (
            not isinstance(self.reducer_identity, str)
            or ":" not in self.reducer_identity
        ):
            raise AlgorithmConfigurationError(
                "reducer_identity must be qualified when provided"
            )
        if self.plan_digest is not None:
            _digest_value(self.plan_digest, "plan_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "invocation_id": self.invocation_id,
            "algorithm": self.algorithm,
            "implementation_ref": self.implementation_ref,
            "implementation_code_digest": self.implementation_code_digest,
            "policy_digest": self.policy_digest,
            "execution_plan_digest": self.execution_plan_digest,
            "runtime_id": self.runtime_id,
            "reducer_identity": self.reducer_identity,
            "plan_digest": self.plan_digest,
        }


@PublicAPI(stability="alpha")
@runtime_checkable
class TorchCheckpointPayloadDraft(Protocol):
    """Optional Core payload draft hook used by checkpoint report tests."""

    checkpoint_dir: str | os.PathLike[str]

    def report(
        self,
        *,
        metrics: Mapping[str, object],
        stage_context: object,
        completed_step: int,
    ) -> None: ...


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchCheckpointRef:
    """Driver-owned reference to an actual Ray/framework checkpoint."""

    checkpoint: object
    descriptor_digest: str | None = None
    source_stage_id: str | None = None
    descriptor: "TorchCheckpointDescriptor | None" = None

    def __post_init__(self) -> None:
        if self.checkpoint is None:
            raise AlgorithmConfigurationError("Torch checkpoint reference is required")
        if self.descriptor_digest is not None:
            _digest_value(self.descriptor_digest, "descriptor_digest")
        if self.descriptor is not None:
            if not isinstance(self.descriptor, TorchCheckpointDescriptor):
                raise AlgorithmConfigurationError(
                    "Torch checkpoint descriptor is invalid"
                )
            if (
                self.descriptor_digest is not None
                and self.descriptor.digest != self.descriptor_digest
            ):
                raise AlgorithmConfigurationError(
                    "Torch checkpoint descriptor digest mismatch"
                )

    def __getstate__(self) -> object:
        raise TypeError("TorchCheckpointRef is driver-local and cannot be serialized")

    def __copy__(self) -> object:
        raise TypeError("TorchCheckpointRef cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        del memo
        raise TypeError("TorchCheckpointRef cannot be copied")

    def close(self) -> None:
        """Release a driver-side checkpoint handle when the backend supports it."""
        closer = getattr(self.checkpoint, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "TorchCheckpointRef":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


@PublicAPI(stability="alpha")
class TorchPreflightLease:
    """Invocation-local one-shot ownership token for a preflight result."""

    __slots__ = ("_data", "_state", "_lock")

    def __init__(self, data: TorchPreflightTokenData) -> None:
        if not isinstance(data, TorchPreflightTokenData):
            raise AlgorithmConfigurationError("preflight lease requires TokenData")
        self._data = data
        self._state = "fresh"
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def _matches(
        self, *, run_id: str, invocation_id: str, plan_digest: str, runtime_id: str
    ) -> bool:
        return (
            self._data.run_id == run_id
            and self._data.invocation_id == invocation_id
            and (
                self._data.plan_digest is None
                and self._data.execution_plan_digest == plan_digest
                or self._data.plan_digest is not None
                and self._data.plan_digest == plan_digest
            )
            and self._data.runtime_id == runtime_id
        )

    def claim(
        self,
        *,
        run_id: str,
        invocation_id: str,
        plan_digest: str,
        runtime_id: str,
    ) -> None:
        with self._lock:
            if self._state != "fresh":
                raise AlgorithmExecutionError("Torch preflight lease is not fresh")
            if not self._matches(
                run_id=run_id,
                invocation_id=invocation_id,
                plan_digest=plan_digest,
                runtime_id=runtime_id,
            ):
                raise AlgorithmExecutionError("Torch preflight lease identity mismatch")
            self._state = "claimed"

    def consume(
        self,
        *,
        run_id: str,
        invocation_id: str,
        plan_digest: str,
        runtime_id: str,
    ) -> TorchPreflightTokenData:
        with self._lock:
            if self._state != "claimed":
                raise AlgorithmExecutionError("Torch preflight lease is not claimed")
            if not self._matches(
                run_id=run_id,
                invocation_id=invocation_id,
                plan_digest=plan_digest,
                runtime_id=runtime_id,
            ):
                raise AlgorithmExecutionError("Torch preflight token identity mismatch")
            self._state = "consumed"
            return self._data

    def close(self) -> None:
        with self._lock:
            if self._state in {"fresh", "claimed"}:
                self._state = "closed"

    @property
    def data(self) -> TorchPreflightTokenData:
        with self._lock:
            if self._state not in {"fresh", "claimed"}:
                raise AlgorithmExecutionError("Torch preflight lease is closed")
            return self._data

    def __copy__(self) -> object:
        raise TypeError("TorchPreflightLease cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        del memo
        raise TypeError("TorchPreflightLease cannot be copied")

    def __getstate__(self) -> object:
        raise TypeError("TorchPreflightLease cannot be serialized")


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchWorkerControlEnvelope:
    """Credential-free serialized control for initial Stage Checkpoint input."""

    schema_version: int
    run_id: str
    invocation_id: str
    source_stage_id: str | None
    target_stage_id: str
    purpose: str
    checkpoint_locator: "TorchCheckpointLocator"
    checkpoint_descriptor_digest: str
    policy_digest: str
    execution_plan_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise AlgorithmConfigurationError(
                "unsupported Torch worker control version"
            )
        if self.purpose not in {"stage_dependency", "cross_run_initial_recovery"}:
            raise AlgorithmConfigurationError("invalid Torch worker control purpose")
        if self.purpose == "stage_dependency" and (
            not isinstance(self.source_stage_id, str) or not self.source_stage_id
        ):
            raise AlgorithmConfigurationError(
                "stage_dependency control requires a source_stage_id"
            )
        for name in ("run_id", "invocation_id", "target_stage_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise AlgorithmConfigurationError(f"{name} must be non-empty")
        if self.source_stage_id == self.target_stage_id:
            raise AlgorithmConfigurationError(
                "Torch control source and target Stage must differ"
            )
        if not isinstance(self.checkpoint_locator, TorchCheckpointLocator):
            raise AlgorithmConfigurationError("Torch checkpoint locator is invalid")
        _digest_value(self.checkpoint_descriptor_digest, "checkpoint_descriptor_digest")
        if (
            self.checkpoint_locator.descriptor_digest
            != self.checkpoint_descriptor_digest
        ):
            raise AlgorithmConfigurationError(
                "Torch control locator and descriptor digests do not match"
            )
        _digest_value(self.policy_digest, "policy_digest")
        _digest_value(self.execution_plan_digest, "execution_plan_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "invocation_id": self.invocation_id,
            "source_stage_id": self.source_stage_id,
            "target_stage_id": self.target_stage_id,
            "purpose": self.purpose,
            "checkpoint_locator": self.checkpoint_locator.to_dict(),
            "checkpoint_descriptor_digest": self.checkpoint_descriptor_digest,
            "policy_digest": self.policy_digest,
            "execution_plan_digest": self.execution_plan_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TorchWorkerControlEnvelope":
        raw_locator = value.get("checkpoint_locator")
        if not isinstance(raw_locator, Mapping):
            raise AlgorithmConfigurationError(
                "Torch worker control locator must be a typed locator"
            )
        locator = TorchCheckpointLocator.from_dict(raw_locator)
        try:
            return cls(
                schema_version=value["schema_version"],
                run_id=value["run_id"],
                invocation_id=value["invocation_id"],
                source_stage_id=value.get("source_stage_id"),
                target_stage_id=value["target_stage_id"],
                purpose=value["purpose"],
                checkpoint_locator=locator,
                checkpoint_descriptor_digest=value["checkpoint_descriptor_digest"],
                policy_digest=value["policy_digest"],
                execution_plan_digest=value["execution_plan_digest"],
            )
        except KeyError as exc:
            raise AlgorithmConfigurationError(
                "Torch worker control is missing a field"
            ) from exc


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchRuntimeExecutionEnvelope:
    """Driver-local Torch envelope carrying a claimed preflight lease."""

    base: "RuntimeExecutionEnvelope"
    preflight_lease: TorchPreflightLease

    def __post_init__(self) -> None:
        if not hasattr(self.base, "plan") or not hasattr(self.base, "run_id"):
            raise AlgorithmConfigurationError(
                "Torch envelope requires a RuntimeExecutionEnvelope"
            )
        if not isinstance(self.preflight_lease, TorchPreflightLease):
            raise AlgorithmConfigurationError(
                "Torch envelope requires a preflight lease"
            )

    def __getstate__(self) -> object:
        raise TypeError("Torch runtime envelope cannot be serialized")

    def __copy__(self) -> object:
        raise TypeError("Torch runtime envelope cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        del memo
        raise TypeError("Torch runtime envelope cannot be copied")


def _loss_numerator_and_normalizer(loss: TorchStepLoss) -> tuple[object, float]:
    if isinstance(loss, TorchLossContribution):
        return loss.numerator, loss.normalizer
    if isinstance(loss, TorchCompositeLossContribution):
        raise AlgorithmConfigurationError(
            "composite loss requires invoke_torch_global_loss_reducer"
        )
    raise AlgorithmConfigurationError("unsupported Torch loss contribution")


@PublicAPI(stability="alpha")
def apply_torch_loss_backward(
    loss: TorchStepLoss,
    window: TorchAccumulationWindow,
    context: TorchBackwardContext,
) -> TorchBackwardResult:
    """Submit one ordinary loss contribution to the Core accumulation helper."""
    if isinstance(loss, TorchCompositeLossContribution):
        if context.compose_composite is None:
            raise AlgorithmConfigurationError(
                "composite loss requires a Core composite backward callback"
            )
        numerator = context.compose_composite(loss)
        if not isinstance(getattr(numerator, "ndim", None), int):
            raise AlgorithmExecutionError(
                "composite backward callback returned invalid numerator"
            )
        normalizer = sum(loss.normalizer_components.values())
    else:
        numerator, normalizer = _loss_numerator_and_normalizer(loss)
    context.backward(numerator)
    next_window = window.add(normalizer)
    complete = next_window.observed_micro_batches >= next_window.expected_micro_batches
    if isinstance(loss, TorchCompositeLossContribution):
        if complete:
            # The reducer coefficients already encode the global objective;
            # applying the ordinary numerator/normalizer scale would normalize
            # it a second time.  ``compose_composite`` must include the
            # Core-required world-size factor before returning its scalar.
            context.finalize_window(1.0)
        return TorchBackwardResult(
            normalizer,
            next_window.normalizer_total,
            complete,
        )
    if not complete:
        return TorchBackwardResult(normalizer, next_window.normalizer_total, False)
    reduce = context.reduce_window_normalizer or context.reduce_normalizer
    global_normalizer = _finite_number(
        reduce(next_window.normalizer_total), "global loss normalizer"
    )
    if global_normalizer <= 0:
        raise AlgorithmExecutionError("global loss normalizer must be positive")
    context.finalize_window(context.world_size / global_normalizer)
    return TorchBackwardResult(normalizer, global_normalizer, True)


@PublicAPI(stability="alpha")
def reduce_torch_metrics(
    contributions: Mapping[str, TorchMetricContribution],
    policy: TorchMetricPolicy,
    context: TorchMetricReductionContext,
) -> TorchMetricReductionResult:
    """Reduce explicit metric numerator/normalizer contributions."""
    if set(contributions) != set(policy.reducers):
        raise AlgorithmConfigurationError(
            "metric contribution names do not match policy"
        )
    values: dict[str, float] = {}
    for name, contribution in contributions.items():
        if not isinstance(contribution, TorchMetricContribution):
            raise AlgorithmConfigurationError(f"metric {name!r} is not typed")
        result = context.reduce(name, contribution, policy.reducers[name])
        values[name] = _finite_number(result, f"metric[{name}]")
    return TorchMetricReductionResult(values)


@PublicAPI(stability="alpha")
def invoke_torch_global_loss_reducer(
    contribution: TorchCompositeLossContribution,
    global_state: TorchCompositeGlobalState,
    reducer: TorchGlobalLossReducer,
    context: TorchGlobalLossContext,
) -> TorchGlobalLossReduction:
    """Invoke an algorithm-owned reducer without granting collective access."""
    if not isinstance(contribution, TorchCompositeLossContribution):
        raise AlgorithmConfigurationError("composite loss contribution is required")
    if contribution.schema_id != reducer.component_schema_id:
        raise AlgorithmConfigurationError(
            "composite loss schema does not match reducer"
        )
    if getattr(reducer, "api_version", None) != 1:
        raise AlgorithmConfigurationError(
            "Torch global loss reducer API version must be 1"
        )
    if (
        not isinstance(getattr(reducer, "reducer_id", None), str)
        or not reducer.reducer_id
    ):
        raise AlgorithmConfigurationError(
            "Torch global loss reducer identity is required"
        )
    _digest_value(getattr(reducer, "code_digest", None), "reducer code digest")
    result = reducer.reduce(context.config, global_state, context)
    if not isinstance(result, TorchGlobalLossReduction):
        raise AlgorithmExecutionError("global loss reducer returned an invalid result")
    if result.status == "accepted" and set(result.coefficients) != set(
        contribution.differentiable_components
    ):
        raise AlgorithmExecutionError(
            "Torch global loss reducer coefficients do not match components"
        )
    return result


@PublicAPI(stability="alpha")
def report_torch_checkpoint(
    metrics: Mapping[str, object],
    payload_draft: TorchCheckpointPayloadDraft,
    stage_context: TorchStageContext,
    completed_step: int,
) -> None:
    """Report a Core-validated Torch checkpoint through Ray Train."""
    if completed_step < 0:
        raise AlgorithmConfigurationError("completed_step must be non-negative")
    if not isinstance(metrics, Mapping):
        raise AlgorithmConfigurationError("Torch checkpoint metrics must be a mapping")

    def validate_metric_metadata(value: object, path: str = "metrics") -> None:
        if isinstance(value, Mapping):
            for raw_key, nested in value.items():
                if not isinstance(raw_key, str):
                    raise AlgorithmConfigurationError(
                        "Torch checkpoint metric keys must be strings"
                    )
                key = raw_key.casefold()
                if key in {
                    "path",
                    "uri",
                    "locator",
                    "checkpoint",
                    "credential",
                    "credentials",
                    "secret",
                    "secrets",
                    "password",
                    "token",
                } or key.endswith(("_path", "_uri", "_locator")):
                    raise AlgorithmConfigurationError(
                        f"Torch checkpoint metrics contain a Core-owned field: {path}.{raw_key}"
                    )
                validate_metric_metadata(nested, f"{path}.{raw_key}")
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                validate_metric_metadata(nested, f"{path}[{index}]")
        elif isinstance(value, str):
            if value.startswith(("/", "~/")) or re.match(
                r"^[A-Za-z][A-Za-z0-9+.-]*://", value
            ):
                raise AlgorithmConfigurationError(
                    f"Torch checkpoint metrics contain a path or URI value: {path}"
                )

    validate_metric_metadata(metrics)
    reporter = getattr(payload_draft, "report", None)
    checkpoint_dir = getattr(payload_draft, "checkpoint_dir", None)
    if not callable(reporter) or checkpoint_dir is None:
        raise AlgorithmExecutionError(
            "Torch checkpoints require a Core payload draft with checkpoint_dir and report()"
        )
    root = Path(checkpoint_dir)
    if root.is_symlink() or not root.is_dir():
        raise AlgorithmExecutionError("Torch checkpoint payload directory is missing")
    root = root.resolve()
    runtime = getattr(stage_context, "runtime", None)
    identity = getattr(runtime, "run_identity", None)
    if identity is None:
        raise AlgorithmExecutionError(
            "Torch checkpoint Stage context has no Run identity"
        )
    if runtime is None:
        raise AlgorithmExecutionError("Torch checkpoint Stage context has no runtime")
    binding_digest = getattr(runtime, "input_binding_digest", None)
    if not isinstance(binding_digest, str) or len(binding_digest) != 64:
        raise AlgorithmExecutionError(
            "Torch checkpoint Stage context has no complete input binding digest"
        )
    descriptor_path = root / "torch_checkpoint_descriptor.json"
    if descriptor_path.is_symlink():
        raise AlgorithmExecutionError(
            "Torch checkpoint descriptor must not be a symlink"
        )
    if descriptor_path.is_file():
        try:
            previous = TorchCheckpointDescriptor.from_dict(
                json.loads(descriptor_path.read_text(encoding="utf-8"))
            )
        except (OSError, TypeError, ValueError) as exc:
            raise AlgorithmExecutionError(
                "existing Torch checkpoint descriptor is malformed"
            ) from exc
        if previous.identity != identity:
            raise AlgorithmExecutionError(
                "Torch checkpoint identity changed within a Stage"
            )
        if completed_step <= previous.completed_step:
            raise AlgorithmExecutionError(
                "Torch checkpoint completed_step must increase monotonically"
            )
    evidence_path = root / "torch_execution_evidence.json"
    if evidence_path.is_symlink():
        raise AlgorithmExecutionError(
            "Torch checkpoint execution evidence must not be a symlink"
        )
    evidence_payload = {
        name: metrics[name]
        for name in (
            "execution_workers",
            "model_state_digest",
            "reducer_id",
            "reducer_api_version",
            "reducer_schema_id",
            "reducer_code_digest",
            "reducer_branch",
            "reducer_evidence",
        )
        if name in metrics
    }
    if evidence_payload:
        temporary_evidence = root / ".torch_execution_evidence.tmp"
        if temporary_evidence.exists() or temporary_evidence.is_symlink():
            raise AlgorithmExecutionError(
                "Torch checkpoint evidence temporary path already exists"
            )
        try:
            fd = os.open(
                temporary_evidence,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(_canonical_json(evidence_payload))
                stream.write("\n")
            os.replace(temporary_evidence, evidence_path)
        except OSError as exc:
            try:
                temporary_evidence.unlink(missing_ok=True)
            except OSError:
                pass
            raise AlgorithmExecutionError(
                "failed to atomically write Torch checkpoint evidence"
            ) from exc
    files = _scan_checkpoint_files(root)
    if not files:
        raise AlgorithmExecutionError("Torch checkpoint payload is empty")
    descriptor = TorchCheckpointDescriptor(
        schema_version=1,
        identity=identity,
        run_config_name=identity.run_config_name,
        state_layout=getattr(runtime, "state_layout", "replicated"),
        world_size=runtime.world_size,
        completed_step=completed_step,
        policy_digest=runtime.policy_digest,
        execution_plan_digest=runtime.execution_plan_digest,
        input_binding_digest=binding_digest,
        implementation_code_digest=identity.implementation_code_digest,
        payload_files=dict(files),
        adapter_identity=getattr(runtime, "adapter_identity", None),
        resume_supported=getattr(runtime, "resume_supported", True),
        same_world_size_resume=getattr(runtime, "same_world_size_resume", True),
    )
    temporary_descriptor = root / ".torch_checkpoint_descriptor.tmp"
    if temporary_descriptor.exists() or temporary_descriptor.is_symlink():
        raise AlgorithmExecutionError(
            "Torch checkpoint descriptor temporary path already exists"
        )
    try:
        fd = os.open(
            temporary_descriptor,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(_canonical_json(descriptor.to_dict()))
            stream.write("\n")
        os.replace(temporary_descriptor, descriptor_path)
    except OSError as exc:
        try:
            temporary_descriptor.unlink(missing_ok=True)
        except OSError:
            pass
        raise AlgorithmExecutionError(
            "failed to atomically write Torch checkpoint descriptor"
        ) from exc
    report_metrics = dict(metrics)
    report_metrics["checkpoint_descriptor"] = descriptor.to_dict()
    reporter(
        metrics=report_metrics,
        stage_context=stage_context,
        completed_step=completed_step,
    )


def _scan_checkpoint_files(root: Path) -> dict[str, str]:
    resolved_root = root.resolve()
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(resolved_root):
            raise AlgorithmExecutionError("Torch checkpoint payload escapes its root")
        if path.name in {
            "torch_checkpoint_descriptor.json",
            "torch_stage_commit.json",
            ".metadata.json",
        }:
            continue
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files[str(path.relative_to(root))] = digest
    return files


@PublicAPI(stability="alpha")
def describe_torch_checkpoint(
    checkpoint_ref: TorchCheckpointRef,
    checkpoint_context: object,
) -> TorchCheckpointDescriptor:
    """Read and validate the Core descriptor embedded in a Driver Checkpoint."""
    if not isinstance(checkpoint_ref, TorchCheckpointRef):
        raise AlgorithmConfigurationError("Torch checkpoint reference is required")
    descriptor = checkpoint_ref.descriptor
    with _opened_checkpoint(checkpoint_ref.checkpoint) as root:
        descriptor_path = root / "torch_checkpoint_descriptor.json"
        if not descriptor_path.is_file() or descriptor_path.is_symlink():
            raise AlgorithmExecutionError("Torch checkpoint has no Core descriptor")
        try:
            parsed = TorchCheckpointDescriptor.from_dict(
                json.loads(descriptor_path.read_text(encoding="utf-8"))
            )
        except (OSError, TypeError, ValueError) as exc:
            raise AlgorithmExecutionError(
                "Torch checkpoint descriptor is malformed"
            ) from exc
        if descriptor is not None and descriptor.digest != parsed.digest:
            raise AlgorithmExecutionError("Torch checkpoint descriptor drifted")
        descriptor = parsed
        if descriptor.run_config_name != descriptor.identity.run_config_name:
            raise AlgorithmExecutionError("Torch checkpoint RunConfig name drifted")
        commit_path = root / "torch_stage_commit.json"
        if commit_path.exists() or commit_path.is_symlink():
            if commit_path.is_symlink() or not commit_path.is_file():
                raise AlgorithmExecutionError(
                    "Torch checkpoint commit marker is invalid"
                )
            try:
                commit = json.loads(commit_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError) as exc:
                raise AlgorithmExecutionError(
                    "Torch checkpoint commit marker is malformed"
                ) from exc
            if not isinstance(commit, Mapping) or (
                commit.get("identity") != descriptor.identity.to_dict()
                or commit.get("descriptor_digest") != descriptor.digest
            ):
                raise AlgorithmExecutionError(
                    "Torch checkpoint commit marker does not match descriptor"
                )
        files = _scan_checkpoint_files(root)
        if dict(descriptor.payload_files) != files:
            raise AlgorithmExecutionError(
                "Torch checkpoint payload files or digest drifted"
            )
    stage = getattr(checkpoint_context, "stage", None)
    if (
        stage is not None
        and getattr(stage, "stage_id", descriptor.identity.stage_id)
        != descriptor.identity.stage_id
    ):
        raise AlgorithmExecutionError("Torch checkpoint descriptor Stage mismatch")
    runtime = getattr(stage, "runtime", None)
    expected_identity = getattr(runtime, "run_identity", None)
    if expected_identity is not None and descriptor.identity != expected_identity:
        raise AlgorithmExecutionError("Torch checkpoint descriptor identity mismatch")
    if runtime is not None:
        for name in ("policy_digest", "execution_plan_digest"):
            expected = getattr(runtime, name, None)
            if expected is not None and getattr(descriptor, name) != expected:
                raise AlgorithmExecutionError(
                    f"Torch checkpoint descriptor {name} drifted"
                )
        expected_binding = getattr(runtime, "input_binding_digest", None)
        if (
            expected_binding is not None
            and descriptor.input_binding_digest != expected_binding
        ):
            raise AlgorithmExecutionError(
                "Torch checkpoint descriptor input binding drifted"
            )
        expected_adapter = getattr(runtime, "adapter_identity", None)
        if descriptor.adapter_identity != expected_adapter:
            raise AlgorithmExecutionError(
                "Torch checkpoint descriptor Adapter identity drifted"
            )
        for name in ("state_layout", "resume_supported", "same_world_size_resume"):
            expected = getattr(runtime, name, None)
            if getattr(descriptor, name) != expected:
                raise AlgorithmExecutionError(
                    f"Torch checkpoint descriptor {name} drifted"
                )
    for name in ("run_id", "invocation_id"):
        expected = getattr(checkpoint_context, name, None)
        if expected is not None and getattr(descriptor.identity, name) != expected:
            raise AlgorithmExecutionError(f"Torch checkpoint descriptor {name} drifted")
    return descriptor


@contextmanager
def _opened_checkpoint(checkpoint: object) -> Generator[Path, None, None]:
    """Open a Ray Checkpoint or local checkpoint directory for validation."""
    if isinstance(checkpoint, (str, Path)):
        root = Path(checkpoint)
        if not root.is_dir():
            raise AlgorithmExecutionError("Torch checkpoint directory is missing")
        yield root
        return
    as_directory = getattr(checkpoint, "as_directory", None)
    if not callable(as_directory):
        raise AlgorithmExecutionError("Torch checkpoint cannot be opened")
    try:
        with as_directory() as directory:
            yield Path(directory)
    except Exception as exc:
        raise AlgorithmExecutionError("Torch checkpoint could not be opened") from exc


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchCheckpointLocator:
    """Credential-free persistent location for a validated Torch checkpoint."""

    uri: str
    descriptor_digest: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise AlgorithmConfigurationError("unsupported Torch locator version")
        if not isinstance(self.uri, str) or not self.uri or "\x00" in self.uri:
            raise AlgorithmConfigurationError("Torch checkpoint locator URI is invalid")
        if self.uri.startswith(("/", "file://")):
            raise AlgorithmConfigurationError(
                "Torch checkpoint locator must not persist a local path"
            )
        try:
            parsed = urlsplit(self.uri)
        except ValueError as exc:
            raise AlgorithmConfigurationError(
                "Torch checkpoint locator URI is invalid"
            ) from exc
        if parsed.username is not None or parsed.password is not None:
            raise AlgorithmConfigurationError(
                "Torch checkpoint locator must not contain URI userinfo"
            )
        if parsed.query or parsed.fragment:
            raise AlgorithmConfigurationError(
                "Torch checkpoint locator must not contain query or fragment data"
            )
        _digest_value(self.descriptor_digest, "descriptor_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "uri": self.uri,
            "descriptor_digest": self.descriptor_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TorchCheckpointLocator":
        try:
            return cls(
                uri=value["uri"],
                descriptor_digest=value["descriptor_digest"],
                schema_version=value.get("schema_version", 1),
            )
        except KeyError as exc:
            raise AlgorithmConfigurationError(
                "Torch checkpoint locator is incomplete"
            ) from exc


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchRecoveryEnvelope:
    """Credential-free cross-Run recovery state for one Torch execution plan."""

    completed_stage_ids: tuple[str, ...] = ()
    stage_checkpoints: Mapping[str, TorchCheckpointLocator] = field(
        default_factory=dict
    )
    active_stage_id: str | None = None
    active_checkpoint: TorchCheckpointLocator | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise AlgorithmConfigurationError(
                "unsupported Torch recovery envelope version"
            )
        completed = tuple(self.completed_stage_ids)
        if any(not isinstance(stage, str) or not stage for stage in completed):
            raise AlgorithmConfigurationError(
                "Torch recovery completed_stage_ids must be non-empty strings"
            )
        if len(set(completed)) != len(completed):
            raise AlgorithmConfigurationError(
                "Torch recovery completed_stage_ids must be unique"
            )
        checkpoints: dict[str, TorchCheckpointLocator] = {}
        for stage_id, locator in self.stage_checkpoints.items():
            if not isinstance(stage_id, str) or not stage_id:
                raise AlgorithmConfigurationError(
                    "Torch recovery checkpoint stage IDs must be non-empty"
                )
            if not isinstance(locator, TorchCheckpointLocator):
                raise AlgorithmConfigurationError(
                    "Torch recovery stage checkpoints must use locators"
                )
            checkpoints[stage_id] = locator
        if set(checkpoints) != set(completed):
            raise AlgorithmConfigurationError(
                "Torch recovery stage checkpoints must match completed stages"
            )
        if self.active_stage_id is None:
            if self.active_checkpoint is not None:
                raise AlgorithmConfigurationError(
                    "Torch recovery active checkpoint requires active_stage_id"
                )
        else:
            if not isinstance(self.active_stage_id, str) or not self.active_stage_id:
                raise AlgorithmConfigurationError(
                    "Torch recovery active_stage_id must be non-empty"
                )
            if self.active_stage_id in completed:
                raise AlgorithmConfigurationError(
                    "Torch recovery active Stage cannot be completed"
                )
            if not isinstance(self.active_checkpoint, TorchCheckpointLocator):
                raise AlgorithmConfigurationError(
                    "Torch recovery active Stage requires a checkpoint locator"
                )
        object.__setattr__(self, "completed_stage_ids", completed)
        object.__setattr__(
            self,
            "stage_checkpoints",
            MappingProxyType(dict(sorted(checkpoints.items()))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "completed_stage_ids": list(self.completed_stage_ids),
            "stage_checkpoints": {
                stage_id: locator.to_dict()
                for stage_id, locator in self.stage_checkpoints.items()
            },
            "active_stage_id": self.active_stage_id,
            "active_checkpoint": (
                self.active_checkpoint.to_dict()
                if self.active_checkpoint is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TorchRecoveryEnvelope":
        raw_checkpoints = value.get("stage_checkpoints", {})
        if not isinstance(raw_checkpoints, Mapping):
            raise AlgorithmConfigurationError(
                "Torch recovery stage_checkpoints must be a mapping"
            )
        raw_active = value.get("active_checkpoint")
        parsed_checkpoints: dict[str, TorchCheckpointLocator] = {}
        for stage_id, locator in raw_checkpoints.items():
            if not isinstance(stage_id, str) or not isinstance(locator, Mapping):
                raise AlgorithmConfigurationError(
                    "Torch recovery stage_checkpoints entries are malformed"
                )
            parsed_checkpoints[stage_id] = TorchCheckpointLocator.from_dict(locator)
        try:
            return cls(
                schema_version=value.get("schema_version", 1),
                completed_stage_ids=tuple(value.get("completed_stage_ids", ())),
                stage_checkpoints=parsed_checkpoints,
                active_stage_id=cast(str | None, value.get("active_stage_id")),
                active_checkpoint=(
                    TorchCheckpointLocator.from_dict(raw_active)
                    if isinstance(raw_active, Mapping)
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                "Torch recovery envelope is malformed"
            ) from exc


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchRankProgressStatistics:
    """Typed per-rank prefix statistics restored with a Torch checkpoint."""

    rows_processed: int = 0
    coverage_totals: Mapping[str, int] = field(default_factory=dict)
    loss_numerator_total: float = 0.0
    loss_normalizer_total: float = 0.0
    metric_totals: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    evaluation_totals: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    reducer_observation: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.rows_processed, int)
            or isinstance(self.rows_processed, bool)
            or self.rows_processed < 0
        ):
            raise AlgorithmConfigurationError("Torch rank rows_processed is invalid")
        coverage = dict(self.coverage_totals)
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for name, value in coverage.items()
        ):
            raise AlgorithmConfigurationError("Torch rank coverage totals are invalid")

        def normalize_totals(
            values: Mapping[str, tuple[float, float]],
        ) -> dict[str, tuple[float, float]]:
            normalized: dict[str, tuple[float, float]] = {}
            for name, pair in values.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(pair, (list, tuple))
                    or len(pair) != 2
                ):
                    raise AlgorithmConfigurationError(
                        "Torch rank metric totals are invalid"
                    )
                numerator = _finite_number(pair[0], f"rank metric[{name}] numerator")
                normalizer = _finite_number(pair[1], f"rank metric[{name}] normalizer")
                if normalizer < 0:
                    raise AlgorithmConfigurationError(
                        "Torch rank metric normalizer is invalid"
                    )
                normalized[name] = (numerator, normalizer)
            return normalized

        loss_numerator = _finite_number(
            self.loss_numerator_total, "rank loss numerator total"
        )
        loss_normalizer = _finite_number(
            self.loss_normalizer_total, "rank loss normalizer total"
        )
        if loss_normalizer < 0:
            raise AlgorithmConfigurationError("Torch rank loss normalizer is invalid")
        metric_totals = normalize_totals(self.metric_totals)
        evaluation_totals = normalize_totals(self.evaluation_totals)
        _validate_bounded_evidence(
            self.reducer_observation, path="rank.reducer_observation"
        )
        object.__setattr__(
            self, "coverage_totals", MappingProxyType(dict(sorted(coverage.items())))
        )
        object.__setattr__(self, "loss_numerator_total", loss_numerator)
        object.__setattr__(self, "loss_normalizer_total", loss_normalizer)
        object.__setattr__(
            self, "metric_totals", MappingProxyType(dict(sorted(metric_totals.items())))
        )
        object.__setattr__(
            self,
            "evaluation_totals",
            MappingProxyType(dict(sorted(evaluation_totals.items()))),
        )
        object.__setattr__(
            self, "reducer_observation", deep_freeze(self.reducer_observation)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_processed": self.rows_processed,
            "coverage_totals": dict(self.coverage_totals),
            "loss_numerator_total": self.loss_numerator_total,
            "loss_normalizer_total": self.loss_normalizer_total,
            "metric_totals": {
                name: list(pair) for name, pair in self.metric_totals.items()
            },
            "evaluation_totals": {
                name: list(pair) for name, pair in self.evaluation_totals.items()
            },
            "reducer_observation": dict(self.reducer_observation),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TorchRankProgressStatistics":
        try:
            return cls(
                rows_processed=value.get("rows_processed", 0),
                coverage_totals=value.get("coverage_totals", {}),
                loss_numerator_total=value.get("loss_numerator_total", 0.0),
                loss_normalizer_total=value.get("loss_normalizer_total", 0.0),
                metric_totals=value.get("metric_totals", {}),
                evaluation_totals=value.get("evaluation_totals", {}),
                reducer_observation=value.get("reducer_observation", {}),
            )
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                "Torch rank progress statistics are malformed"
            ) from exc


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchCheckpointProgress:
    """Deterministic cursor state required for exact Torch recovery."""

    epoch: int
    micro_batch_cursor: int
    optimizer_step: int
    scheduler_step: int
    accumulation_steps: int
    dataset_cursor_by_rank: Mapping[str, int] = field(default_factory=dict)
    shuffle_seed: int | None = None
    rows_processed: int = 0
    coverage_totals: Mapping[str, int] = field(default_factory=dict)
    loss_numerator_total: float = 0.0
    loss_normalizer_total: float = 0.0
    metric_totals: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    evaluation_totals: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    rank_statistics: Mapping[str, TorchRankProgressStatistics] = field(
        default_factory=dict
    )
    epoch_scheduler_applied: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise AlgorithmConfigurationError(
                "unsupported Torch checkpoint progress version"
            )
        for name in (
            "epoch",
            "micro_batch_cursor",
            "optimizer_step",
            "scheduler_step",
            "accumulation_steps",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or (name == "accumulation_steps" and value < 1)
            ):
                raise AlgorithmConfigurationError(
                    f"Torch checkpoint progress {name} is invalid"
                )
        cursors: dict[str, int] = {}
        for rank, cursor in self.dataset_cursor_by_rank.items():
            if (
                not isinstance(rank, str)
                or not rank
                or not isinstance(cursor, int)
                or isinstance(cursor, bool)
                or cursor < 0
            ):
                raise AlgorithmConfigurationError(
                    "Torch checkpoint dataset cursors are invalid"
                )
            cursors[rank] = cursor
        if self.shuffle_seed is not None and (
            not isinstance(self.shuffle_seed, int)
            or isinstance(self.shuffle_seed, bool)
        ):
            raise AlgorithmConfigurationError(
                "Torch checkpoint shuffle_seed is invalid"
            )
        if (
            not isinstance(self.rows_processed, int)
            or isinstance(self.rows_processed, bool)
            or self.rows_processed < 0
        ):
            raise AlgorithmConfigurationError(
                "Torch checkpoint rows_processed is invalid"
            )
        coverage = dict(self.coverage_totals)
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for name, value in coverage.items()
        ):
            raise AlgorithmConfigurationError(
                "Torch checkpoint coverage totals are invalid"
            )
        loss_numerator = _finite_number(
            self.loss_numerator_total, "checkpoint loss numerator total"
        )
        loss_normalizer = _finite_number(
            self.loss_normalizer_total, "checkpoint loss normalizer total"
        )
        if loss_normalizer < 0:
            raise AlgorithmConfigurationError(
                "Torch checkpoint loss normalizer is invalid"
            )

        def normalize_totals(
            values: Mapping[str, tuple[float, float]],
        ) -> dict[str, tuple[float, float]]:
            normalized: dict[str, tuple[float, float]] = {}
            for name, pair in values.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(pair, (list, tuple))
                    or len(pair) != 2
                ):
                    raise AlgorithmConfigurationError(
                        "Torch checkpoint metric totals are invalid"
                    )
                numerator = _finite_number(
                    pair[0], f"checkpoint metric[{name}] numerator"
                )
                normalizer = _finite_number(
                    pair[1], f"checkpoint metric[{name}] normalizer"
                )
                if normalizer < 0:
                    raise AlgorithmConfigurationError(
                        "Torch checkpoint metric normalizer is invalid"
                    )
                normalized[name] = (numerator, normalizer)
            return normalized

        metric_totals = normalize_totals(self.metric_totals)
        evaluation_totals = normalize_totals(self.evaluation_totals)
        rank_statistics = dict(self.rank_statistics)
        if len(rank_statistics) > 1024 or any(
            not isinstance(rank, str)
            or not rank
            or not isinstance(stats, TorchRankProgressStatistics)
            for rank, stats in rank_statistics.items()
        ):
            raise AlgorithmConfigurationError(
                "Torch checkpoint rank statistics are invalid"
            )
        if not isinstance(self.epoch_scheduler_applied, bool):
            raise AlgorithmConfigurationError(
                "Torch checkpoint epoch_scheduler_applied must be boolean"
            )
        object.__setattr__(
            self,
            "dataset_cursor_by_rank",
            MappingProxyType(dict(sorted(cursors.items()))),
        )
        object.__setattr__(self, "rows_processed", self.rows_processed)
        object.__setattr__(
            self, "coverage_totals", MappingProxyType(dict(sorted(coverage.items())))
        )
        object.__setattr__(self, "loss_numerator_total", loss_numerator)
        object.__setattr__(self, "loss_normalizer_total", loss_normalizer)
        object.__setattr__(
            self, "metric_totals", MappingProxyType(dict(sorted(metric_totals.items())))
        )
        object.__setattr__(
            self,
            "evaluation_totals",
            MappingProxyType(dict(sorted(evaluation_totals.items()))),
        )
        object.__setattr__(
            self,
            "rank_statistics",
            MappingProxyType(dict(sorted(rank_statistics.items()))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "epoch": self.epoch,
            "micro_batch_cursor": self.micro_batch_cursor,
            "optimizer_step": self.optimizer_step,
            "scheduler_step": self.scheduler_step,
            "accumulation_steps": self.accumulation_steps,
            "dataset_cursor_by_rank": dict(self.dataset_cursor_by_rank),
            "shuffle_seed": self.shuffle_seed,
            "rows_processed": self.rows_processed,
            "coverage_totals": dict(self.coverage_totals),
            "loss_numerator_total": self.loss_numerator_total,
            "loss_normalizer_total": self.loss_normalizer_total,
            "metric_totals": {
                name: list(pair) for name, pair in self.metric_totals.items()
            },
            "evaluation_totals": {
                name: list(pair) for name, pair in self.evaluation_totals.items()
            },
            "rank_statistics": {
                rank: statistics.to_dict()
                for rank, statistics in self.rank_statistics.items()
            },
            "epoch_scheduler_applied": self.epoch_scheduler_applied,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TorchCheckpointProgress":
        raw_cursors = value.get("dataset_cursor_by_rank", {})
        if not isinstance(raw_cursors, Mapping):
            raise AlgorithmConfigurationError(
                "Torch checkpoint dataset cursors must be a mapping"
            )
        raw_statistics = value.get("rank_statistics", {})
        if not isinstance(raw_statistics, Mapping) or any(
            not isinstance(rank, str) or not isinstance(statistics, Mapping)
            for rank, statistics in raw_statistics.items()
        ):
            raise AlgorithmConfigurationError(
                "Torch checkpoint rank statistics must be a typed mapping"
            )
        try:
            return cls(
                schema_version=value.get("schema_version", 1),
                epoch=value["epoch"],
                micro_batch_cursor=value["micro_batch_cursor"],
                optimizer_step=value["optimizer_step"],
                scheduler_step=value["scheduler_step"],
                accumulation_steps=value["accumulation_steps"],
                dataset_cursor_by_rank=dict(raw_cursors),
                shuffle_seed=value.get("shuffle_seed"),
                rows_processed=value.get("rows_processed", 0),
                coverage_totals=value.get("coverage_totals", {}),
                loss_numerator_total=value.get("loss_numerator_total", 0.0),
                loss_normalizer_total=value.get("loss_normalizer_total", 0.0),
                metric_totals=value.get("metric_totals", {}),
                evaluation_totals=value.get("evaluation_totals", {}),
                rank_statistics={
                    rank: TorchRankProgressStatistics.from_dict(statistics)
                    for rank, statistics in raw_statistics.items()
                },
                epoch_scheduler_applied=value.get("epoch_scheduler_applied", False),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                "Torch checkpoint progress is malformed"
            ) from exc


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchCheckpointDescriptor:
    """Validated manifest embedded in every portable Torch checkpoint."""

    schema_version: int
    identity: TorchStageRunIdentity
    run_config_name: str
    state_layout: str
    world_size: int
    completed_step: int
    policy_digest: str
    execution_plan_digest: str
    input_binding_digest: str
    implementation_code_digest: str
    payload_files: Mapping[str, str]
    adapter_identity: str | None = None
    resume_supported: bool = True
    same_world_size_resume: bool | None = True
    torch_runtime_api_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise AlgorithmConfigurationError("unsupported Torch checkpoint descriptor")
        if not isinstance(self.identity, TorchStageRunIdentity):
            raise AlgorithmConfigurationError("checkpoint identity is invalid")
        if (
            self.torch_runtime_api_version != 1
            or self.identity.torch_runtime_api_version != 1
        ):
            raise AlgorithmConfigurationError(
                "Torch checkpoint runtime API version must be exactly 1"
            )
        for field_name in ("run_config_name", "input_binding_digest"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise AlgorithmConfigurationError(f"{field_name} must be non-empty")
        if self.run_config_name != self.identity.run_config_name:
            raise AlgorithmConfigurationError(
                "checkpoint run_config_name does not match Stage identity"
            )
        if self.state_layout not in {"replicated", "component", "sharded"}:
            raise AlgorithmConfigurationError("checkpoint state_layout is invalid")
        for field_name in ("world_size", "completed_step"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise AlgorithmConfigurationError(f"{field_name} must be non-negative")
        if self.world_size < 1:
            raise AlgorithmConfigurationError("world_size must be positive")
        for field_name in (
            "policy_digest",
            "execution_plan_digest",
            "input_binding_digest",
            "implementation_code_digest",
        ):
            _digest_value(getattr(self, field_name), field_name)
        if self.identity.policy_digest != self.policy_digest:
            raise AlgorithmConfigurationError(
                "checkpoint Policy digest does not match identity"
            )
        if self.identity.execution_plan_digest != self.execution_plan_digest:
            raise AlgorithmConfigurationError(
                "checkpoint execution plan digest does not match identity"
            )
        if self.identity.implementation_code_digest != self.implementation_code_digest:
            raise AlgorithmConfigurationError(
                "checkpoint implementation code digest does not match identity"
            )
        files = dict(self.payload_files)
        if not files:
            raise AlgorithmConfigurationError("checkpoint payload_files are required")
        for name, digest in files.items():
            if (
                not isinstance(name, str)
                or not name
                or name.startswith("/")
                or "\\" in name
                or ".." in name.split("/")
            ):
                raise AlgorithmConfigurationError(
                    "checkpoint payload file path is unsafe"
                )
            _digest_value(digest, f"payload_files[{name}]")
        if not isinstance(self.resume_supported, bool):
            raise AlgorithmConfigurationError("resume_supported must be boolean")
        if self.same_world_size_resume is not None and not isinstance(
            self.same_world_size_resume, bool
        ):
            raise AlgorithmConfigurationError("same_world_size_resume must be boolean")
        if self.resume_supported and self.same_world_size_resume is not True:
            raise AlgorithmConfigurationError(
                "supported recovery requires same world size"
            )
        if not self.resume_supported and self.same_world_size_resume is not None:
            raise AlgorithmConfigurationError(
                "unsupported recovery must omit same_world_size_resume"
            )
        object.__setattr__(
            self, "payload_files", MappingProxyType(dict(sorted(files.items())))
        )

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "run_config_name": self.run_config_name,
            "state_layout": self.state_layout,
            "world_size": self.world_size,
            "completed_step": self.completed_step,
            "policy_digest": self.policy_digest,
            "execution_plan_digest": self.execution_plan_digest,
            "input_binding_digest": self.input_binding_digest,
            "implementation_code_digest": self.implementation_code_digest,
            "payload_files": dict(self.payload_files),
            "adapter_identity": self.adapter_identity,
            "resume_supported": self.resume_supported,
            "torch_runtime_api_version": self.torch_runtime_api_version,
        }
        if self.same_world_size_resume is not None:
            payload["same_world_size_resume"] = self.same_world_size_resume
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TorchCheckpointDescriptor":
        try:
            identity = TorchStageRunIdentity.from_dict(value["identity"])
            resume_supported = value.get("resume_supported", True)
            return cls(
                schema_version=value["schema_version"],
                identity=identity,
                run_config_name=value["run_config_name"],
                state_layout=value["state_layout"],
                world_size=value["world_size"],
                completed_step=value["completed_step"],
                policy_digest=value["policy_digest"],
                execution_plan_digest=value["execution_plan_digest"],
                input_binding_digest=value["input_binding_digest"],
                implementation_code_digest=value["implementation_code_digest"],
                payload_files=value["payload_files"],
                adapter_identity=value.get("adapter_identity"),
                resume_supported=resume_supported,
                same_world_size_resume=value.get(
                    "same_world_size_resume", True if resume_supported else None
                ),
                torch_runtime_api_version=value.get("torch_runtime_api_version", 1),
            )
        except KeyError as exc:
            raise AlgorithmConfigurationError(
                f"Torch checkpoint descriptor is missing {exc.args[0]!r}"
            ) from exc


__all__ = [
    "TorchAccumulationWindow",
    "TorchBackwardContext",
    "TorchBackwardResult",
    "TorchCheckpointPayloadDraft",
    "TorchCheckpointDescriptor",
    "TorchCheckpointLocator",
    "TorchRecoveryEnvelope",
    "TorchCheckpointProgress",
    "TorchRankProgressStatistics",
    "TorchCheckpointRef",
    "TorchCompositeGlobalState",
    "TorchCompositeLossContribution",
    "TorchGlobalLossContext",
    "TorchGlobalLossReducer",
    "TorchGlobalLossReduction",
    "TorchLossContribution",
    "TorchMetricContribution",
    "TorchMetricPolicy",
    "TorchMetricReductionContext",
    "TorchMetricReductionResult",
    "TorchPreflightLease",
    "TorchPreflightTokenData",
    "TorchStageRunIdentity",
    "TorchStepLoss",
    "TorchRuntimeExecutionEnvelope",
    "TorchWorkerControlEnvelope",
    "apply_torch_loss_backward",
    "claim_torch_run_directory",
    "describe_torch_checkpoint",
    "invoke_torch_global_loss_reducer",
    "reduce_torch_metrics",
    "report_torch_checkpoint",
    "torch_run_config_name",
    "validate_torch_retry_identity",
]
