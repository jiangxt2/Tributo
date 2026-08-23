"""Bundle-backed model-kernel provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from tributo.exporting.runtime import BundleModelLoader
from tributo.inference.contracts import ResolvedModelSelection
from tributo.inference.kernel import PredictionKernel, PredictionKernelFactory
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class BundlePredictionKernelFactory:
    """Load one pinned Bundle prediction kernel inside a worker process."""

    factory_id: ClassVar[str] = "tributo.bundle-prediction-v1"
    model: ResolvedModelSelection

    def create(self) -> PredictionKernel:
        """Open the verified Bundle artifact selected by the opaque model ref."""
        return BundleModelLoader().open(
            self.model.bundle_ref.canonical_uri,
            role=self.model.role,
            storage_profile=self.model.storage_profile,
            unsafe=self.model.unsafe,
            expected_manifest_sha256=self.model.bundle_ref.manifest_sha256,
            use_case="batch",
        )


@PublicAPI(stability="alpha")
class BundleModelKernelProvider:
    """Adapt resolved Bundle selections to format-neutral kernel factories."""

    provider_id: ClassVar[str] = "tributo.bundle-model-runtime-v1"

    def prediction_factory(self, model: object) -> PredictionKernelFactory:
        """Return a serializable factory without loading weights on the driver."""
        if not isinstance(model, ResolvedModelSelection):
            raise TypeError(
                "BundleModelKernelProvider requires ResolvedModelSelection, got "
                f"{type(model).__name__}"
            )
        return BundlePredictionKernelFactory(model=model)


__all__ = ["BundleModelKernelProvider", "BundlePredictionKernelFactory"]
