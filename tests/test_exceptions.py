"""Tests for exception hierarchy."""

from __future__ import annotations

import pytest

from tributo.exceptions import (
    DataSourceError,
    JobConfigurationError,
    JobExecutionError,
    JobSubmissionError,
    JobTimeoutError,
    ModelExportError,
    TributoError,
)


def test_tributo_error_basic():
    """Test basic TributoError."""
    error = TributoError("Test error")
    assert str(error) == "Test error"


def test_tributo_error_with_cause():
    """Test TributoError with cause chaining."""
    cause = ValueError("Original error")
    try:
        raise TributoError("Test error") from cause
    except TributoError as error:
        assert str(error) == "Test error"
        assert error.__cause__ is cause


def test_job_submission_error():
    """Test JobSubmissionError."""
    error = JobSubmissionError("Submission failed")
    assert isinstance(error, TributoError)
    assert str(error) == "Submission failed"


def test_job_execution_error():
    """Test JobExecutionError."""
    error = JobExecutionError("Execution failed")
    assert isinstance(error, TributoError)
    assert str(error) == "Execution failed"


def test_job_configuration_error():
    """Test JobConfigurationError."""
    error = JobConfigurationError("Configuration invalid")
    assert isinstance(error, TributoError)
    assert str(error) == "Configuration invalid"


def test_job_timeout_error():
    """Test JobTimeoutError."""
    error = JobTimeoutError("Timeout")
    assert isinstance(error, TributoError)
    assert str(error) == "Timeout"


def test_model_export_error():
    """Test ModelExportError."""
    error = ModelExportError("Export failed")
    assert isinstance(error, TributoError)
    assert str(error) == "Export failed"


def test_datasource_error():
    """Test DataSourceError."""
    error = DataSourceError("S3 connection timeout")
    assert isinstance(error, TributoError)
    assert str(error) == "S3 connection timeout"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
