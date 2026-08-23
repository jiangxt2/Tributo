"""Compatibility wrapper for Bundle-backed batch prediction."""

from __future__ import annotations

from tributo.inference.contracts import (
    InputBindingSpec,
    OutputBindingSpec,
    ResolvedModelSelection,
)
from tributo.inference.kernel import KernelBatchPredictor, ModelKernelProvider
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class BundleBatchPredictor(KernelBatchPredictor):
    """Preserve the Bundle-shaped constructor over the generic predictor."""

    def __init__(
        self,
        model: ResolvedModelSelection,
        input_binding: InputBindingSpec,
        output_binding: OutputBindingSpec,
        *,
        kernel_provider: ModelKernelProvider | None = None,
    ) -> None:
        provider = kernel_provider or _default_model_kernel_provider()
        super().__init__(
            provider.prediction_factory(model),
            input_binding,
            output_binding,
        )


def _default_model_kernel_provider() -> ModelKernelProvider:
    """Resolve the compatibility provider through top-level composition."""
    from tributo.runtime import default_model_kernel_provider

    return default_model_kernel_provider()


__all__ = ["BundleBatchPredictor"]
