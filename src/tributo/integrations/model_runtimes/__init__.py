"""Model-runtime adapters that produce inference kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tributo.integrations.model_runtimes.bundle import (
        BundleModelKernelProvider,
        BundlePredictionKernelFactory,
    )
    from tributo.integrations.model_runtimes.explainability import (
        BundleExplainabilityModelProvider,
        BundleExplainabilityModelSession,
        BundleExplainabilityModelSessionFactory,
    )
    from tributo.integrations.model_runtimes.resolver import (
        BundleModelReferenceResolver,
    )


def __getattr__(name: str) -> Any:
    if name in {"BundleModelKernelProvider", "BundlePredictionKernelFactory"}:
        from tributo.integrations.model_runtimes import bundle

        return getattr(bundle, name)
    if name in {
        "BundleExplainabilityModelProvider",
        "BundleExplainabilityModelSession",
        "BundleExplainabilityModelSessionFactory",
    }:
        from tributo.integrations.model_runtimes import explainability

        return getattr(explainability, name)
    if name == "BundleModelReferenceResolver":
        from tributo.integrations.model_runtimes.resolver import (
            BundleModelReferenceResolver,
        )

        return BundleModelReferenceResolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BundleModelKernelProvider",
    "BundleExplainabilityModelProvider",
    "BundleExplainabilityModelSession",
    "BundleExplainabilityModelSessionFactory",
    "BundleModelReferenceResolver",
    "BundlePredictionKernelFactory",
]
