"""Stateful Ray batch actor over the shared Bundle runtime."""

from __future__ import annotations

import numpy as np

from tributo.exporting.runtime import BundleModelLoader, BundleModelRuntime
from tributo.inference.contracts import (
    InputBindingSpec,
    OutputBindingSpec,
    ResolvedModelSelection,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class BundleBatchPredictor:
    """Load one pinned Bundle per Ray actor and execute named bindings."""

    def __init__(
        self,
        model: ResolvedModelSelection,
        input_binding: InputBindingSpec,
        output_binding: OutputBindingSpec,
    ) -> None:
        self._model = model
        self._input_binding = input_binding
        self._output_binding = output_binding
        self._closed = False
        self._runtime: BundleModelRuntime = BundleModelLoader().open(
            model.bundle_ref.canonical_uri,
            role=model.role,
            storage_profile=model.storage_profile,
            unsafe=model.unsafe,
            expected_manifest_sha256=model.bundle_ref.manifest_sha256,
            use_case="batch",
        )

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compile the table batch into named tensors and bind named outputs."""
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
        raw_outputs = self._runtime.predict(inputs)

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
            if binding.squeeze_singleton and output.ndim == 2 and output.shape[1] == 1:
                output = output[:, 0]
            if binding.dtype is not None:
                output = output.astype(binding.dtype, copy=False)
            result[binding.column] = output
        return result

    def close(self) -> None:
        """Release Bundle reader resources; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        self._runtime.close()

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


__all__ = ["BundleBatchPredictor"]
