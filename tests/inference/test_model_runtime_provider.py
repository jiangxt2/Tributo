"""Tests for the format-neutral model-runtime boundary."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tributo.exporting.models import BundleRef
from tributo.inference.contracts import ResolvedModelSelection
from tributo.integrations.model_runtimes.bundle import (
    BundleModelKernelProvider,
    BundlePredictionKernelFactory,
)


def _selection() -> ResolvedModelSelection:
    return ResolvedModelSelection(
        bundle_ref=BundleRef(
            canonical_uri="/models/bundle",
            bundle_id="bundle-1",
            manifest_sha256="a" * 64,
        ),
        role="inference",
        flavor_id="opaque-runtime-v1",
        storage_profile="model-domain",
        source_provenance="test",
    )


def test_bundle_kernel_factory_loads_only_when_worker_creates_kernel() -> None:
    runtime = object()
    factory = BundlePredictionKernelFactory(model=_selection())

    with patch(
        "tributo.integrations.model_runtimes.bundle.BundleModelLoader.open",
        return_value=runtime,
    ) as open_model:
        actual = factory.create()

    assert actual is runtime
    open_model.assert_called_once_with(
        "/models/bundle",
        role="inference",
        storage_profile="model-domain",
        unsafe=False,
        expected_manifest_sha256="a" * 64,
        use_case="batch",
    )


def test_bundle_model_provider_returns_serializable_factory() -> None:
    selection = _selection()

    factory = BundleModelKernelProvider().prediction_factory(selection)

    assert factory.model == selection
    assert factory.factory_id == "tributo.bundle-prediction-v1"


def test_bundle_model_provider_rejects_unresolved_models() -> None:
    with pytest.raises(TypeError, match="ResolvedModelSelection"):
        BundleModelKernelProvider().prediction_factory(object())
