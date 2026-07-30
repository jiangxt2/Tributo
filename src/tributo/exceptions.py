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
class AliasConflict(TributoError):
    """Alias CAS update failed — concurrent modification detected."""


@PublicAPI(stability="beta")
class UnsupportedArtifactFormat(TributoError):
    """Consumer does not support this artifact format or flavor.

    Raised at startup/construction time, not on the first request.
    """


@PublicAPI(stability="beta")
class PostPublishCallbackError(TributoError):
    """Post-publish callback failed after the bundle was already published.

    The model is published — this error means the callback (e.g. MLflow)
    failed, not the export itself.  Carries ``bundle_result``.
    """

    def __init__(self, message: str, bundle_result: Any = None) -> None:
        super().__init__(message)
        self.bundle_result = bundle_result


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
