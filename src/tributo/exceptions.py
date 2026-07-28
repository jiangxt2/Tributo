"""Exception hierarchy for Tributo.

Following Ray's pattern of having a clear exception hierarchy with
serialization support for distributed execution.
"""

from __future__ import annotations

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
