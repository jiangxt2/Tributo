"""Tests for deployment-neutral Ray runtime targets."""

from __future__ import annotations

import pytest

from tributo import (
    RuntimeExecutionMode,
    RuntimeLifecycle,
    RuntimeSubmissionMode,
    RuntimeTarget,
)
from tributo.exceptions import JobConfigurationError


def test_local_master_owns_local_runtime() -> None:
    target = RuntimeTarget.from_master("local")

    assert target.execution_mode is RuntimeExecutionMode.LOCAL
    assert target.submission_mode is RuntimeSubmissionMode.LOCAL
    assert target.lifecycle is RuntimeLifecycle.OWNED
    assert target.provider == "local"
    assert not target.is_managed


def test_http_master_uses_ray_jobs_on_attached_cluster() -> None:
    target = RuntimeTarget.from_master("http://ray-head:8265/")

    assert target.execution_mode is RuntimeExecutionMode.CLUSTER
    assert target.submission_mode is RuntimeSubmissionMode.JOBS
    assert target.lifecycle is RuntimeLifecycle.ATTACHED
    assert target.require_jobs_address() == "http://ray-head:8265"


def test_ray_master_is_interactive_client_mode() -> None:
    target = RuntimeTarget.from_master("ray://ray-head:10001/")

    assert target.execution_mode is RuntimeExecutionMode.CLUSTER
    assert target.submission_mode is RuntimeSubmissionMode.CLIENT
    assert target.lifecycle is RuntimeLifecycle.ATTACHED
    with pytest.raises(JobConfigurationError, match="Ray Jobs endpoint"):
        target.require_jobs_address()


def test_managed_master_requires_explicit_provider_reference() -> None:
    target = RuntimeTarget.from_master("managed://ray_cluster_launcher/cluster.yaml")

    assert target.is_managed
    assert target.provider == "ray_cluster_launcher"
    assert target.provider_config == "cluster.yaml"
    with pytest.raises(JobConfigurationError, match="unavailable"):
        target.require_jobs_address()


@pytest.mark.parametrize(
    "master",
    [
        "",
        "k8s://cluster",
        "ray://",
        "managed://",
        "managed://provider/",
        "https://user:secret@ray-head:8265",
        "ray://user:secret@ray-head:10001",
    ],
)
def test_unsupported_master_fails_closed(master: str) -> None:
    with pytest.raises(JobConfigurationError):
        RuntimeTarget.from_master(master)


def test_local_target_rejects_cluster_fields() -> None:
    with pytest.raises(JobConfigurationError, match="owned local"):
        RuntimeTarget(
            master="local",
            execution_mode=RuntimeExecutionMode.LOCAL,
            submission_mode=RuntimeSubmissionMode.JOBS,
            lifecycle=RuntimeLifecycle.OWNED,
            provider="local",
            address="http://ray-head:8265",
        )


def test_attached_target_rejects_mismatched_submission_endpoint() -> None:
    with pytest.raises(JobConfigurationError, match="jobs submission requires"):
        RuntimeTarget(
            master="http://ray-head:8265",
            execution_mode=RuntimeExecutionMode.CLUSTER,
            submission_mode=RuntimeSubmissionMode.JOBS,
            lifecycle=RuntimeLifecycle.ATTACHED,
            provider="external",
            address="ray://ray-head:10001",
        )

    with pytest.raises(JobConfigurationError, match="credentials"):
        RuntimeTarget(
            master="ray://user:secret@ray-head:10001",
            execution_mode=RuntimeExecutionMode.CLUSTER,
            submission_mode=RuntimeSubmissionMode.CLIENT,
            lifecycle=RuntimeLifecycle.ATTACHED,
            provider="external",
            address="ray://user:secret@ray-head:10001",
        )
