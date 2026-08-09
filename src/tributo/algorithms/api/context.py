"""Restricted context exposed to a user-owned Ray Worker function."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from tributo._common.immutable import deep_freeze, deep_thaw
from tributo.algorithms.api.errors import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
)
from tributo.algorithms.api.models import (
    AlgorithmExecutionResult,
    ArtifactDraft,
    canonical_digest,
)
from tributo.util.annotations import PublicAPI

_MAX_USER_ARTIFACT_BYTES = 16 * 1024 * 1024


@PublicAPI(stability="alpha")
class UserExecutionContext:
    """Least-authority API available to a trusted module-qualified function."""

    def __init__(
        self,
        *,
        inputs: Mapping[str, object],
        configuration: Mapping[str, Any],
        worker_metadata: Mapping[str, Any],
        artifacts: tuple[ArtifactDraft, ...] = (),
        cancelled: bool = False,
    ) -> None:
        self._inputs = dict(inputs)
        self._configuration = cast(Mapping[str, Any], deep_freeze(configuration))
        self._worker_metadata = cast(Mapping[str, Any], deep_freeze(worker_metadata))
        self._artifacts_in = tuple(artifacts)
        self._cancelled = cancelled
        self._metrics: dict[str, Any] = {}
        self._outputs: dict[str, Any] = {}
        self._artifacts: list[ArtifactDraft] = []

    @property
    def worker_metadata(self) -> Mapping[str, Any]:
        """Return immutable rank and resource facts for the current Worker."""
        return self._worker_metadata

    @property
    def configuration(self) -> Mapping[str, Any]:
        """Return the immutable validated algorithm configuration."""
        return self._configuration

    @property
    def cancelled(self) -> bool:
        """Return the cancellation snapshot supplied by the Runtime."""
        return self._cancelled

    def get_input(self, name: str) -> object:
        """Return one named framework-neutral Worker input view."""
        if name not in self._inputs:
            raise AlgorithmExecutionError(
                f"unknown input {name!r}; available: {sorted(self._inputs)}"
            )
        return self._inputs[name]

    def get_resume_checkpoint(self) -> ArtifactDraft | None:
        """Return the unique checkpoint candidate supplied for resume."""
        checkpoints = [
            artifact for artifact in self._artifacts_in if artifact.kind == "checkpoint"
        ]
        if len(checkpoints) > 1:
            raise AlgorithmExecutionError(
                "execution received more than one resume checkpoint"
            )
        return checkpoints[0] if checkpoints else None

    def report(self, metrics: Mapping[str, Any]) -> None:
        """Report portable metrics without silently overwriting a prior value."""
        frozen = self._validated_report(metrics, kind="metrics")
        duplicates = sorted(set(self._metrics).intersection(frozen))
        if duplicates:
            raise AlgorithmExecutionError(
                f"metrics were reported more than once: {duplicates}"
            )
        self._metrics.update(deep_thaw(frozen))

    def report_outputs(self, outputs: Mapping[str, Any]) -> None:
        """Report bounded JSON-compatible outputs."""
        frozen = self._validated_report(outputs, kind="outputs")
        duplicates = sorted(set(self._outputs).intersection(frozen))
        if duplicates:
            raise AlgorithmExecutionError(
                f"outputs were reported more than once: {duplicates}"
            )
        self._outputs.update(deep_thaw(frozen))

    @staticmethod
    def _validated_report(
        values: Mapping[str, Any],
        *,
        kind: str,
    ) -> Mapping[str, Any]:
        """Normalize invalid user reports into the execution error taxonomy."""
        try:
            frozen = cast(Mapping[str, Any], deep_freeze(values))
            canonical_digest(frozen)
        except (AlgorithmConfigurationError, TypeError) as exc:
            raise AlgorithmExecutionError(
                f"user {kind} must contain only finite portable JSON values"
            ) from exc
        return frozen

    def stage_artifact(
        self,
        *,
        name: str,
        kind: str,
        format: str,
        payload: bytes,
    ) -> ArtifactDraft:
        """Stage a bounded artifact payload for platform validation."""
        if not isinstance(payload, bytes):
            raise AlgorithmExecutionError("user artifact payload must be bytes")
        if len(payload) > _MAX_USER_ARTIFACT_BYTES:
            raise AlgorithmExecutionError(
                "user artifact exceeds the 16 MiB portable execution limit"
            )
        if any(artifact.name == name for artifact in self._artifacts):
            raise AlgorithmExecutionError(f"duplicate artifact name: {name!r}")
        artifact = ArtifactDraft.from_payload(
            name=name,
            kind=kind,
            format=format,
            payload=payload,
            trusted=False,
        )
        self._artifacts.append(artifact)
        return artifact

    def report_checkpoint(
        self,
        *,
        payload: bytes,
        format: str,
        name: str = "checkpoint",
    ) -> ArtifactDraft:
        """Stage a checkpoint candidate through the artifact boundary."""
        return self.stage_artifact(
            name=name,
            kind="checkpoint",
            format=format,
            payload=payload,
        )

    def build_result(self) -> AlgorithmExecutionResult:
        """Build the platform result after the user function returns."""
        return AlgorithmExecutionResult(
            status="succeeded",
            metrics=self._metrics,
            outputs=self._outputs,
            artifacts=tuple(self._artifacts),
        )


__all__ = ["UserExecutionContext"]
