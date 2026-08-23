"""Format-neutral batch-kernel contracts and table/tensor binding."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from tributo.inference.contracts import InputBindingSpec, OutputBindingSpec
from tributo.util.annotations import PublicAPI


@runtime_checkable
@PublicAPI(stability="alpha")
class PredictionKernel(Protocol):
    """Process-local prediction capability returned by a runtime provider."""

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...

    def close(self) -> None: ...


@runtime_checkable
@PublicAPI(stability="alpha")
class PredictionKernelFactory(Protocol):
    """Serializable factory that loads one prediction kernel inside a worker."""

    factory_id: ClassVar[str]

    def create(self) -> PredictionKernel: ...


@runtime_checkable
@PublicAPI(stability="alpha")
class ModelKernelProvider(Protocol):
    """Resolve an opaque model selection into a worker-local kernel factory."""

    provider_id: ClassVar[str]

    def prediction_factory(self, model: object) -> PredictionKernelFactory: ...


@PublicAPI(stability="alpha")
class KernelBatchPredictor:
    """Bind table batches to one format-neutral resident prediction kernel."""

    def __init__(
        self,
        kernel_factory: PredictionKernelFactory,
        input_binding: InputBindingSpec,
        output_binding: OutputBindingSpec,
    ) -> None:
        self._kernel = kernel_factory.create()
        self._input_binding = input_binding
        self._output_binding = output_binding
        self._closed = False

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compile table columns into named tensors and bind named outputs."""
        retained_columns = list(self._input_binding.passthrough_columns)
        if self._output_binding.preserve_features:
            retained_columns.extend(
                column
                for binding in self._input_binding.tensors
                for column in binding.columns
            )
        retained_columns = list(dict.fromkeys(retained_columns))
        _validate_columns(
            batch,
            columns=tuple(retained_columns),
            null_policy=self._input_binding.null_policy,
            nan_policy=self._input_binding.nan_policy,
        )

        inputs = {
            binding.tensor_name: _build_input_tensor(
                batch,
                columns=binding.columns,
                dtype=binding.dtype,
                null_policy=self._input_binding.null_policy,
                nan_policy=self._input_binding.nan_policy,
            )
            for binding in self._input_binding.tensors
        }
        input_rows = _tensor_row_count(inputs, kind="input")
        raw_outputs = self._kernel.predict(inputs)

        result = {column: np.asarray(batch[column]) for column in retained_columns}
        for binding in self._output_binding.tensors:
            if binding.tensor_name not in raw_outputs:
                raise KeyError(
                    f"Model output {binding.tensor_name!r} is missing; available: "
                    f"{sorted(raw_outputs)}"
                )
            if binding.column in result:
                raise ValueError(
                    f"Output column {binding.column!r} collides with a retained "
                    "input column"
                )
            output = np.asarray(raw_outputs[binding.tensor_name])
            if output.ndim == 0:
                raise ValueError(
                    f"Model output {binding.tensor_name!r} must preserve the batch "
                    "dimension"
                )
            if output.shape[0] != input_rows:
                raise ValueError(
                    f"Model output {binding.tensor_name!r} changed batch row count "
                    f"from {input_rows} to {output.shape[0]}"
                )
            if binding.squeeze_singleton and output.ndim == 2 and output.shape[1] == 1:
                output = output[:, 0]
            if binding.dtype is not None:
                output = output.astype(binding.dtype, copy=False)
            result[binding.column] = output
        return result

    def close(self) -> None:
        """Release kernel resources; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        self._kernel.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _build_input_tensor(
    batch: dict[str, np.ndarray],
    *,
    columns: tuple[str, ...],
    dtype: str | None,
    null_policy: str,
    nan_policy: str,
) -> np.ndarray:
    _validate_columns(
        batch,
        columns=columns,
        null_policy=null_policy,
        nan_policy=nan_policy,
    )

    arrays = [np.asarray(batch[column]) for column in columns]
    if len(arrays) == 1 and arrays[0].ndim > 1:
        tensor = arrays[0]
    else:
        tensor = np.column_stack(arrays)
    if dtype is not None:
        target_dtype = np.dtype(dtype)
        if target_dtype.kind not in {"f", "c"} and _contains_non_finite(tensor):
            raise ValueError(
                "Non-finite floating-point values cannot be safely converted to "
                f"integer dtype {target_dtype.name!r}"
            )
        tensor = tensor.astype(target_dtype, copy=False)
    if nan_policy == "error" and _contains_nan(tensor):
        raise ValueError(
            "NaN values are not allowed by nan_policy='error'; use "
            "nan_policy='allow' only when the model defines missing-value semantics"
        )
    return np.asarray(tensor)


def _tensor_row_count(tensors: dict[str, np.ndarray], *, kind: str) -> int:
    row_count: int | None = None
    for name, tensor in tensors.items():
        if tensor.ndim == 0:
            raise ValueError(f"Model {kind} {name!r} must include a batch dimension")
        if row_count is None:
            row_count = int(tensor.shape[0])
        elif tensor.shape[0] != row_count:
            raise ValueError(
                f"Model {kind} tensors disagree on batch row count: "
                f"expected {row_count}, got {tensor.shape[0]} for {name!r}"
            )
    if row_count is None:  # InputBindingSpec requires at least one tensor.
        raise ValueError(f"Model {kind} tensors must not be empty")
    return row_count


def _validate_columns(
    batch: dict[str, np.ndarray],
    *,
    columns: tuple[str, ...],
    null_policy: str,
    nan_policy: str,
) -> None:
    missing = [column for column in columns if column not in batch]
    if missing:
        raise KeyError(
            f"Batch is missing input columns {missing}; available: {sorted(batch)}"
        )
    if null_policy == "error":
        null_columns = [column for column in columns if _contains_null(batch[column])]
        if null_columns:
            raise ValueError(f"NULL values are not allowed in columns {null_columns}")
    if nan_policy == "error":
        nan_columns = [
            column for column in columns if _contains_nan(np.asarray(batch[column]))
        ]
        if nan_columns:
            raise ValueError(
                f"NaN values are not allowed in columns {nan_columns} by "
                "nan_policy='error'; use "
                "nan_policy='allow' only when the model defines missing-value semantics"
            )


def _contains_null(value: np.ndarray) -> bool:
    if np.ma.isMaskedArray(value) and bool(np.any(np.ma.getmaskarray(value))):
        return True
    array = np.asarray(value)
    if array.dtype.kind != "O":
        return False
    return any(value is None for value in array.ravel())


def _contains_nan(array: np.ndarray) -> bool:
    if array.dtype.kind in {"f", "c"}:
        return bool(np.any(np.isnan(array)))
    if array.dtype.kind != "O":
        return False
    return any(
        isinstance(value, (float, np.floating)) and bool(np.isnan(value))
        for value in array.ravel()
    )


def _contains_non_finite(array: np.ndarray) -> bool:
    if array.dtype.kind in {"f", "c"}:
        return bool(np.any(~np.isfinite(array)))
    if array.dtype.kind != "O":
        return False
    return any(
        isinstance(value, (float, np.floating, complex, np.complexfloating))
        and not bool(np.isfinite(value))
        for value in array.ravel()
    )


__all__ = [
    "KernelBatchPredictor",
    "ModelKernelProvider",
    "PredictionKernel",
    "PredictionKernelFactory",
]
