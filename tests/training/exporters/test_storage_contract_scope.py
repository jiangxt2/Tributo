"""Stable storage metadata and explicit experimental extension boundaries."""

from __future__ import annotations

import json

import pytest

from tributo._common.storage_profiles import StorageProfileResolver
from tributo.exceptions import (
    AliasConflict,
    BundleCommitBusyError,
    JobConfigurationError,
)
from tributo.explainability.contracts import (
    ExplainabilityConfig,
    ExplainabilityDescriptor,
)
from tributo.exporting import export, load_bundle
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.manifest import ExportManifest, ExportManifestV2
from tributo.exporting.models import (
    BundleRef,
    BundleResult,
    HookReceipt,
    PublishedBundle,
)
from tributo.exporting.publisher import Publisher
from tributo.exporting.repository import BundleAliasStore, BundleRepository
from tributo.util.annotations import get_stability


def test_core_storage_metadata_does_not_promote_extensions() -> None:
    for obj in (
        BundleReader,
        Publisher,
        BundleRepository,
        BundleAliasStore,
        BundleRef,
        BundleResult,
        PublishedBundle,
        ExportManifest,
        BundleCommitBusyError,
        AliasConflict,
    ):
        assert get_stability(obj) == "stable"
    for beta_obj in (ExportManifestV2, HookReceipt, export, load_bundle):
        assert get_stability(beta_obj) == "beta"
    for alpha_obj in (ExplainabilityConfig, ExplainabilityDescriptor):
        assert get_stability(alpha_obj) == "alpha"


def test_named_profile_preserves_credential_safety(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TRIBUTO_STORAGE_PROFILE_CONTRACT",
        json.dumps(
            {
                "endpoint": "http://minio.example:9000",
                "region": "us-east-1",
                "path_style": True,
                "access_key_id": "fixture-key",
                "secret_access_key": "fixture-secret",
            }
        ),
    )
    profile = StorageProfileResolver().resolve("contract")
    assert profile.endpoint == "http://minio.example:9000"
    assert profile.path_style
    assert profile.access_key_id == "fixture-key"
    assert profile.secret_access_key == "fixture-secret"
    assert "fixture-key" not in repr(profile)
    assert "fixture-secret" not in repr(profile)
    assert "minio.example" not in repr(profile)


def test_invalid_profile_diagnostic_does_not_echo_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TRIBUTO_STORAGE_PROFILE_CONTRACT",
        json.dumps({"unknown": "fixture-secret"}),
    )
    with pytest.raises(JobConfigurationError) as error:
        StorageProfileResolver().resolve("contract")
    assert "invalid storage profile fields" in str(error.value)
    assert "fixture-secret" not in str(error.value)
