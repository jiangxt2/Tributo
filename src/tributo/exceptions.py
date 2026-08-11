"""Exception hierarchy for Tributo.

Following Ray's pattern of having a clear exception hierarchy with
serialization support for distributed execution.
"""

from __future__ import annotations

from typing import Any

from tributo.util.annotations import PublicAPI


@PublicAPI(stability="stable")
class TributoError(Exception):
    """Base class for all Tributo exception types."""

    pass


@PublicAPI(stability="stable")
class JobSubmissionError(TributoError):
    """Raised when job submission fails."""


@PublicAPI(stability="stable")
class JobExecutionError(TributoError):
    """Raised when job execution fails."""


@PublicAPI(stability="stable")
class SessionFatalError(JobExecutionError):
    """Raised when an error is fatal to the whole export session.

    Session-fatal errors (staging/path integrity violations, source
    recovery failures) cancel all remaining DAG nodes regardless of
    whether the failing node is required — the plan requires these
    errors to be classified distinctly from per-node failures.
    """


@PublicAPI(stability="stable")
class JobConfigurationError(TributoError):
    """Raised when job configuration is invalid."""


@PublicAPI(stability="stable")
class JobTimeoutError(TributoError):
    """Raised when job execution times out."""


@PublicAPI(stability="stable")
class ModelExportError(TributoError):
    """Raised when model export or validation fails."""


@PublicAPI(stability="stable")
class DataSourceError(TributoError):
    """Raised when data source I/O operations fail.

    Covers business-domain I/O errors such as S3 connection timeouts,
    Lance write conflicts, or unexpected data format issues.
    """


@PublicAPI(stability="alpha")
class EngineNotAvailableError(DataSourceError):
    """Requested ingestion engine binding is missing or incompatible."""


# ── Inference exceptions (compatible with inference-protocol §5.5) ────────


@PublicAPI(stability="stable")
class ModelLoadError(TributoError):
    """Model file could not be loaded (e.g. corrupt file, version mismatch)."""


@PublicAPI(stability="stable")
class ModelFormatUnsupportedError(TributoError):
    """Model format not supported by this inference engine."""


@PublicAPI(stability="stable")
class ModelSchemaMismatchError(TributoError):
    """Model input schema does not match the preprocessor output."""


@PublicAPI(stability="stable")
class InputColumnMissingError(TributoError):
    """Required input column is missing in the query result."""


@PublicAPI(stability="stable")
class EmptyInputError(TributoError):
    """Input query returned zero rows."""


@PublicAPI(stability="stable")
class DataQueryError(TributoError):
    """ClickHouse or other data source query failed."""


@PublicAPI(stability="stable")
class ResultWriteError(TributoError):
    """Failed to write inference results to the output data source."""


@PublicAPI(stability="alpha")
class ResultMaterializationError(TributoError):
    """A lazy inference graph failed while a result sink materialized it.

    Ray Data executes upstream reads, transforms, model calls, and distributed
    writes from the terminal sink action.  This exception deliberately avoids
    claiming that an action-time failure came from the sink itself when the
    public Ray API cannot identify the failing operator reliably.
    """

    def __init__(self, source_error_type: str) -> None:
        self.source_error_type = source_error_type
        super().__init__(
            f"Inference result materialization failed ({source_error_type})"
        )


@PublicAPI(stability="stable")
class PredictionError(TributoError):
    """Model prediction failed at runtime."""


@PublicAPI(stability="stable")
class ArtifactCorruptedError(TributoError):
    """Downloaded artifact hash does not match the manifest.

    The file may have been corrupted during download or storage.
    """


# ── Bundle export exceptions ────────────────────────────────────────────────


@PublicAPI(stability="beta")
class BundleExportError(TributoError):
    """Bundle export failed — required nodes did not succeed.

    Carries ``execution_result`` with per-node status and failure details.
    """

    def __init__(self, message: str, execution_result: Any = None) -> None:
        super().__init__(message)
        self.execution_result = execution_result


@PublicAPI(stability="beta")
class BundleCommitBusyError(TributoError):
    """A bundle commit is temporarily blocked by another active writer.

    The immutable bundle may be retried with the same stable identity.
    """

    retryable = True


@PublicAPI(stability="beta")
class AliasConflict(TributoError):
    """Alias CAS update failed — concurrent modification detected."""


@PublicAPI(stability="beta")
class UnsupportedArtifactFormat(TributoError):
    """Consumer does not support this artifact format or flavor.

    Raised at startup/construction time, not on the first request.
    """


@PublicAPI(stability="beta")
class PostPublishCallbackError(TributoError):
    """A required post-publish action failed after the bundle was published.

    The model is published — this error means a secondary action such as
    execution recording or an inline callback failed, not the export itself.
    Carries ``bundle_result``. ``receipts`` is empty when no Hook delivery can
    be claimed.
    """

    def __init__(
        self,
        message: str,
        bundle_result: Any = None,
        receipts: tuple[Any, ...] = (),
    ) -> None:
        super().__init__(message)
        self.bundle_result = bundle_result
        self.receipts = receipts


@PublicAPI(stability="beta")
class PluginLoadIssue(TributoError):
    """A plugin entry-point could not be loaded or validated.

    Non-fatal — collected into registry diagnostics.
    """

    def __init__(
        self,
        group: str,
        entry_point_name: str,
        reason: str,
        exc: BaseException | None = None,
    ) -> None:
        self.group = group
        self.entry_point_name = entry_point_name
        self.reason = reason
        self.original_exception = exc
        super().__init__(f"Plugin {group!r}/{entry_point_name!r}: {reason}")


# ── Streaming exceptions (fail-closed safety baseline) ──────────────────────


@PublicAPI(stability="beta")
class StreamSourceError(TributoError):
    """Base class for streaming source failures.

    Raised when a stream source cannot guarantee its delivery semantics —
    e.g. an offset commit fails or a poisoned message cannot be skipped
    safely.  Carries the partition coordinates when available.
    """

    def __init__(
        self,
        message: str,
        *,
        topic: str | None = None,
        partition: int | None = None,
        offset: int | None = None,
    ) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset
        super().__init__(message)


@PublicAPI(stability="beta")
class KafkaCommitError(StreamSourceError):
    """Kafka offset commit failed.

    The pending batch offsets are retained — the caller may retry
    ``commit()``; they are cleared only after a successful commit.
    """


@PublicAPI(stability="beta")
class KafkaPoisonMessageError(StreamSourceError):
    """A Kafka record could not be decoded or validated (fail-closed).

    Raised instead of skipping the record: the source stops rather than
    silently dropping data.  ``reason`` describes the rejection:
    ``"message_error"``, ``"tombstone"``, ``"decode"``, ``"non_dict"``
    on the offending record, or ``"terminated"`` when a fresh
    ``poll()`` is attempted after a poison record already stopped the
    source.
    """

    def __init__(
        self,
        message: str,
        *,
        topic: str | None = None,
        partition: int | None = None,
        offset: int | None = None,
        reason: str | None = None,
    ) -> None:
        self.reason = reason
        super().__init__(message, topic=topic, partition=partition, offset=offset)


# ── Training resource safety ─────────────────────────────────────────────────


@PublicAPI(stability="beta")
class ResourceBudgetExceededError(TributoError):
    """Single-worker materialization exceeded the configured budget.

    Raised *before* the unbounded concat — the batch that would exceed the
    budget is never appended.  Carries structured context for diagnostics:
    algorithm, split, worker/rank, observed bytes and the budget that was hit.
    """

    def __init__(
        self,
        message: str,
        *,
        algorithm: str | None = None,
        split: str | None = None,
        worker_rank: str | int | None = None,
        observed_bytes: int | None = None,
        budget_bytes: int | None = None,
        observed_rows: int | None = None,
        max_rows: int | None = None,
    ) -> None:
        super().__init__(message)
        self.algorithm = algorithm
        self.split = split
        self.worker_rank = worker_rank
        self.observed_bytes = observed_bytes
        self.budget_bytes = budget_bytes
        self.observed_rows = observed_rows
        self.max_rows = max_rows
