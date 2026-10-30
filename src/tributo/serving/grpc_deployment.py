"""gRPC inference service Deployment.

Based on Ray Serve's gRPC support, provides Unary, Server streaming,
and Client streaming RPC modes.

The primary loading path is a ``bundle_uri`` plus an explicit ``role``,
routed through the shared ``BundleModelLoader``; a raw ``model_path``
remains as a compatibility adapter.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import grpc
import numpy as np

from tributo._common.model_input_contract import (
    canonical_onnx_dtype,
    is_onnx_invalid_argument,
    normalize_model_shape,
    validate_named_inputs,
)
from tributo.serving.observability import InferenceContext, log_inference_audit
from tributo.serving.proto import inference_pb2
from tributo.serving.schema import PredictInput

if TYPE_CHECKING:
    from ray.serve.grpc_util import RayServegRPCContext, gRPCInputStream

from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


def _validate_features(request: inference_pb2.PredictRequest) -> np.ndarray | None:
    """Validate and convert legacy flat request features.

    Args:
        request: gRPC predict request.

    Returns:
        Converted feature array; returns ``None`` if validation fails.
    """
    if not request.features:
        return None
    return np.array(request.features, dtype=np.float32).reshape(1, -1)


#: Input protocol schema version understood by this service.
_SUPPORTED_SCHEMA_VERSION = 1


def _prepare_inputs(
    request: inference_pb2.PredictRequest,
    expected_names: tuple[str, ...] | set[str] | None = None,
    expected_dtypes: tuple[str | None, ...] | None = None,
    expected_shapes: tuple[tuple[int | None, ...] | None, ...] | None = None,
) -> dict[str, np.ndarray] | np.ndarray | None:
    """Convert request inputs to named arrays (versioned) or legacy matrix.

    ``int64_data`` is the lossless integer carrier (``repeated double``
    cannot represent int64 above 2**53); when present it takes precedence
    over ``data``, and an ``int64`` input without it fails fast instead
    of silently losing precision.

    *expected_names*, *expected_dtypes*, and *expected_shapes* — the model's
    typed input signature.  Every versioned request is checked before it
    reaches ONNX Runtime; legacy flat features are restricted to a single
    model input and are checked against that same signature.
    """
    if isinstance(expected_names, (set, frozenset)) and (
        expected_dtypes is not None or expected_shapes is not None
    ):
        raise ValueError(
            "expected_names must be an ordered sequence when typed dtypes or "
            "shapes are provided; a set cannot preserve signature alignment"
        )
    expected_names_tuple = (
        tuple(sorted(expected_names))
        if isinstance(expected_names, (set, frozenset))
        else expected_names
    )
    if request.inputs:
        result: dict[str, np.ndarray] = {}
        for t in request.inputs:
            if t.schema_version != _SUPPORTED_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported input schema_version {t.schema_version!r}; "
                    f"expected {_SUPPORTED_SCHEMA_VERSION}"
                )
            if t.name in result:
                raise ValueError(f"Duplicate input name {t.name!r}")
            if t.datatype == "int64" and not t.int64_data:
                raise ValueError(
                    f"Input {t.name!r} declares datatype 'int64' but carries "
                    "no int64_data — the double 'data' field loses precision "
                    "above 2**53, use int64_data"
                )
            if t.int64_data:
                data: list[Any] = list(t.int64_data)
            else:
                data = list(t.data)
            result[t.name] = PredictInput(
                name=t.name,
                shape=list(t.shape),
                datatype=t.datatype,
                data=data,
            ).to_numpy()
        if expected_names_tuple is not None:
            validate_named_inputs(
                result,
                expected_names=expected_names_tuple,
                expected_dtypes=expected_dtypes,
                expected_shapes=expected_shapes,
            )
        return result
    legacy = _validate_features(request)
    if legacy is None or expected_names_tuple is None:
        return legacy
    if len(expected_names_tuple) != 1:
        raise ValueError(
            "Legacy features support exactly one model input; use versioned "
            "inputs for a multi-input model"
        )
    validate_named_inputs(
        {expected_names_tuple[0]: legacy},
        expected_names=expected_names_tuple,
        expected_dtypes=expected_dtypes,
        expected_shapes=expected_shapes,
    )
    return legacy


def _prepare_inputs_or_invalid(
    request: inference_pb2.PredictRequest,
    context: RayServegRPCContext,
    expected_names: tuple[str, ...] | set[str] | None = None,
    expected_dtypes: tuple[str | None, ...] | None = None,
    expected_shapes: tuple[tuple[int | None, ...] | None, ...] | None = None,
) -> dict[str, np.ndarray] | np.ndarray | None:
    """Prepare inputs, converting protocol errors to INVALID_ARGUMENT.

    Without this, a duplicate name, unknown datatype, bad shape, an
    out-of-range integer (``OverflowError``), or a type mismatch would
    bubble up as an UNKNOWN gRPC status — the caller's error is a client
    contract violation, so it must surface as INVALID_ARGUMENT.
    """
    try:
        result = _prepare_inputs(
            request,
            expected_names,
            expected_dtypes,
            expected_shapes,
        )
        if result is None:
            _set_invalid_argument(
                context,
                "request must contain versioned inputs or legacy features",
            )
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        _set_invalid_argument(context, str(exc))
        return None


def _set_invalid_argument(context: RayServegRPCContext, message: str) -> None:
    """Set gRPC INVALID_ARGUMENT status code and details."""
    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
    context.set_details(message)


def _run_or_invalid(
    run: Any,
    inputs: dict[str, np.ndarray] | np.ndarray,
    context: RayServegRPCContext,
) -> list[np.ndarray] | None:
    """Run inference and map client-shaped runtime errors to gRPC status."""
    try:
        return run(inputs)
    except (ValueError, TypeError, OverflowError) as exc:
        _set_invalid_argument(context, str(exc))
        return None
    except Exception as exc:
        if not is_onnx_invalid_argument(exc):
            raise
        _set_invalid_argument(context, str(exc))
        return None


@PublicAPI(stability="beta")
class gRPCInferenceService:
    """gRPC inference service Deployment.

    Based on Ray Serve's gRPC support, provides Unary, Server streaming,
    and Client streaming RPC modes.

    Note: Do not apply @serve.deployment decorator directly on this class.
    The decoration is handled by deploy_serve_app() to support parameter
    overrides like num_replicas.
    """

    #: Model input names, populated on load.  Class-level default so
    #: instances built without __init__ (tests, __new__) never hit a
    #: missing attribute in the RPC input-name checks.
    _input_names: tuple[str, ...] = ()
    _input_dtypes: tuple[str | None, ...] = ()
    _input_shapes: tuple[tuple[int | None, ...] | None, ...] = ()
    _bundle_id: str | None = None
    _model_version: str | None = None

    def __init__(
        self,
        model_path: str | None = None,
        *,
        bundle_uri: str | None = None,
        role: str = "inference",
        unsafe: bool = False,
        storage_profile: str | None = None,
    ):
        """Initialize gRPC inference service.

        Args:
            model_path: ONNX model file path (legacy compat adapter).
            bundle_uri: Published bundle URI (stable serving entry point).
            role: Artifact role to serve; defaults to ``inference``.
            unsafe: Permit loading bundles without typed signatures or
                flavors that are not safe.
            storage_profile: Storage profile name for S3 bundles.
        """
        if (model_path is None) == (bundle_uri is None):
            raise ValueError(
                "exactly one of 'model_path' (legacy) or 'bundle_uri' must be provided"
            )

        self._runtime: Any = None
        self._session: Any = None
        self._input_name = ""
        self._input_names: tuple[str, ...] = ()
        self._input_dtypes: tuple[str | None, ...] = ()
        self._input_shapes: tuple[tuple[int | None, ...] | None, ...] = ()
        self._output_names: tuple[str, ...] = ()
        self._bundle_id: str | None = None
        self._model_version: str | None = None

        if bundle_uri is not None:
            self._open_bundle(
                bundle_uri, role=role, unsafe=unsafe, storage_profile=storage_profile
            )
            return

        self._open_legacy(model_path)

    def _open_bundle(
        self,
        bundle_uri: str,
        *,
        role: str,
        unsafe: bool,
        storage_profile: str | None,
    ) -> None:
        """Load the model through the shared BundleModelLoader."""
        from tributo.exporting.runtime import BundleModelLoader

        loader = BundleModelLoader()
        self._runtime = loader.open(
            bundle_uri, role=role, unsafe=unsafe, storage_profile=storage_profile
        )
        self._input_name = self._runtime.model.input_names[0]
        self._input_names = tuple(self._runtime.model.input_names)
        self._input_dtypes = tuple(self._runtime.model.input_dtypes)
        self._input_shapes = tuple(self._runtime.model.input_shapes)
        self._output_names = self._runtime.model.output_names
        self._bundle_id = self._runtime.bundle_id
        self._model_version = self._runtime.model_version
        logger.info(
            "gRPC bundle loaded from %s (role=%r), input_name=%s, inputs=%s",
            bundle_uri,
            role,
            self._input_name,
            self._runtime.model.input_names,
        )

    def _open_legacy(self, model_path: str | None) -> None:
        """Legacy compat path: raw ONNX file, no bundle manifest."""
        assert model_path is not None  # guarded by the exclusivity check
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for gRPC serving. Install with: uv sync"
            ) from e

        self._session = ort.InferenceSession(model_path)
        session_inputs = self._session.get_inputs()
        self._input_name = session_inputs[0].name
        self._input_names = tuple(inp.name for inp in session_inputs)
        self._input_dtypes = tuple(
            canonical_onnx_dtype(getattr(inp, "type", None)) for inp in session_inputs
        )
        self._input_shapes = tuple(
            normalize_model_shape(getattr(inp, "shape", None)) for inp in session_inputs
        )
        self._output_names = tuple(
            out.name if out.name else f"output_{i}"
            for i, out in enumerate(self._session.get_outputs())
        )
        logger.info(
            "gRPC model loaded from %s, input_name=%s, inputs=%s",
            model_path,
            self._input_name,
            [inp.name for inp in self._session.get_inputs()],
        )

    def _run(self, inputs: dict[str, np.ndarray] | np.ndarray) -> list[np.ndarray]:
        """Run the model and return outputs as a list of arrays.

        The legacy session path accepts both named inputs (versioned
        protocol) and the flat legacy matrix (mapped to the first input).
        """
        if self._runtime is not None:
            assert isinstance(inputs, dict)
            result = self._runtime.predict(inputs)
            return [np.asarray(result[name]) for name in self._output_names]
        assert self._session is not None
        if isinstance(inputs, dict):
            feed = inputs
        else:
            feed = {self._input_name: inputs}
        return [np.asarray(o) for o in self._session.run(None, feed)]

    def _response(
        self,
        predictions: list[Any] | None = None,
        *,
        context: InferenceContext | None = None,
    ) -> inference_pb2.PredictResponse:
        """Build a response carrying E3 correlation/version metadata."""
        values: dict[str, Any] = {
            "predictions": predictions or [],
            "confidence": max(predictions) if predictions else 0.0,
        }
        if context is not None:
            values.update(
                {
                    key: value
                    for key, value in context.response_fields(
                        bundle_id=self._bundle_id,
                        model_version=self._model_version,
                    ).items()
                    if value is not None
                }
            )
        return inference_pb2.PredictResponse(**values)

    async def Predict(
        self,
        request: inference_pb2.PredictRequest,
        grpc_context: RayServegRPCContext,
    ) -> inference_pb2.PredictResponse:
        """Unary RPC inference.

        Args:
            request: Predict request.
            grpc_context: gRPC context (Ray Serve convention parameter name).

        Returns:
            Predict result.
        """
        context = InferenceContext.from_grpc(grpc_context)
        started = time.perf_counter()
        inputs = _prepare_inputs_or_invalid(
            request,
            grpc_context,
            self._input_names,
            self._input_dtypes or None,
            self._input_shapes or None,
        )
        if inputs is None:
            log_inference_audit(
                logger,
                context,
                bundle_id=self._bundle_id,
                model_version=self._model_version,
                status="invalid_argument",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return self._response(context=context)

        start = time.perf_counter()
        try:
            outputs = _run_or_invalid(self._run, inputs, grpc_context)
        except Exception:
            log_inference_audit(
                logger,
                context,
                bundle_id=self._bundle_id,
                model_version=self._model_version,
                status="error",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        if outputs is None:
            log_inference_audit(
                logger,
                context,
                bundle_id=self._bundle_id,
                model_version=self._model_version,
                status="invalid_argument",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return self._response(context=context)
        predictions = outputs[0].flatten().tolist()
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "gRPC Predict: predictions_len=%d, time=%.2fms",
            len(predictions),
            elapsed_ms,
        )

        log_inference_audit(
            logger,
            context,
            bundle_id=self._bundle_id,
            model_version=self._model_version,
            status="ok",
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        return self._response(predictions, context=context)

    async def StreamPredict(
        self,
        request: inference_pb2.PredictRequest,
        grpc_context: RayServegRPCContext,
    ):
        """Server streaming inference (batched response).

        Args:
            request: Predict request.
            grpc_context: gRPC context (Ray Serve convention parameter name).

        Yields:
            Batched predict results.
        """
        context = InferenceContext.from_grpc(grpc_context)
        started = time.perf_counter()
        inputs = _prepare_inputs_or_invalid(
            request,
            grpc_context,
            self._input_names,
            self._input_dtypes or None,
            self._input_shapes or None,
        )
        if inputs is None:
            log_inference_audit(
                logger,
                context,
                bundle_id=self._bundle_id,
                model_version=self._model_version,
                status="invalid_argument",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return

        try:
            outputs = _run_or_invalid(self._run, inputs, grpc_context)
        except Exception:
            log_inference_audit(
                logger,
                context,
                bundle_id=self._bundle_id,
                model_version=self._model_version,
                status="error",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        if outputs is None:
            log_inference_audit(
                logger,
                context,
                bundle_id=self._bundle_id,
                model_version=self._model_version,
                status="invalid_argument",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return
        predictions = outputs[0].flatten().tolist()
        log_inference_audit(
            logger,
            context,
            bundle_id=self._bundle_id,
            model_version=self._model_version,
            status="ok",
            duration_ms=(time.perf_counter() - started) * 1000,
        )

        # Server streaming: return predictions one by one, suitable for real-time intermediate result display
        for pred in predictions:
            response = self._response([pred], context=context)
            response.confidence = pred
            yield response

    async def BatchPredict(
        self,
        request_stream: gRPCInputStream,
        grpc_context: RayServegRPCContext,
    ) -> inference_pb2.PredictResponse:
        """Client streaming batch inference.

        Args:
            request_stream: Request stream.
            grpc_context: gRPC context (Ray Serve convention parameter name).

        Returns:
            Merged predict results.
        """
        context = InferenceContext.from_grpc(grpc_context)
        started = time.perf_counter()
        inputs_list = []
        async for request in request_stream:
            inputs = _prepare_inputs_or_invalid(
                request,
                grpc_context,
                self._input_names,
                self._input_dtypes or None,
                self._input_shapes or None,
            )
            if inputs is None:
                log_inference_audit(
                    logger,
                    context,
                    bundle_id=self._bundle_id,
                    model_version=self._model_version,
                    status="invalid_argument",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                return self._response(context=context)
            inputs_list.append(inputs)

        if not inputs_list:
            _set_invalid_argument(grpc_context, "request stream cannot be empty")
            log_inference_audit(
                logger,
                context,
                bundle_id=self._bundle_id,
                model_version=self._model_version,
                status="invalid_argument",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return self._response(context=context)

        if isinstance(inputs_list[0], dict):
            # Versioned inputs: every request must use the same mode, the
            # same input names, and matching per-tensor shapes (non-batch
            # axes) — anything else is a client contract violation.
            for item in inputs_list[1:]:
                if not isinstance(item, dict):
                    _set_invalid_argument(
                        grpc_context,
                        "Mixed input modes across batch requests: versioned "
                        "inputs and legacy features cannot be combined",
                    )
                    log_inference_audit(
                        logger,
                        context,
                        bundle_id=self._bundle_id,
                        model_version=self._model_version,
                        status="invalid_argument",
                        duration_ms=(time.perf_counter() - started) * 1000,
                    )
                    return self._response(context=context)
            expected_names = set(inputs_list[0])
            for item in inputs_list[1:]:
                assert isinstance(item, dict)
                if set(item) != expected_names:
                    _set_invalid_argument(
                        grpc_context,
                        "Inconsistent input names across batch requests: "
                        f"expected {sorted(expected_names)}, got {sorted(item)}",
                    )
                    log_inference_audit(
                        logger,
                        context,
                        bundle_id=self._bundle_id,
                        model_version=self._model_version,
                        status="invalid_argument",
                        duration_ms=(time.perf_counter() - started) * 1000,
                    )
                    return self._response(context=context)
            first = inputs_list[0]
            for item in inputs_list[1:]:
                assert isinstance(item, dict)
                for name, arr in item.items():
                    if arr.shape[1:] != first[name].shape[1:]:
                        _set_invalid_argument(
                            grpc_context,
                            f"Input {name!r} has shape {arr.shape} but the "
                            f"first request carries shape {first[name].shape} "
                            "— non-batch dimensions must match across "
                            "batch requests",
                        )
                        log_inference_audit(
                            logger,
                            context,
                            bundle_id=self._bundle_id,
                            model_version=self._model_version,
                            status="invalid_argument",
                            duration_ms=(time.perf_counter() - started) * 1000,
                        )
                        return self._response(context=context)
            named: dict[str, list[np.ndarray]] = {}
            for item in inputs_list:
                assert isinstance(item, dict)
                for name, arr in item.items():
                    named.setdefault(name, []).append(arr)
            batch = {
                name: np.concatenate(parts, axis=0) for name, parts in named.items()
            }
            try:
                outputs = _run_or_invalid(self._run, batch, grpc_context)
            except Exception:
                log_inference_audit(
                    logger,
                    context,
                    bundle_id=self._bundle_id,
                    model_version=self._model_version,
                    status="error",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                raise
        else:
            for item in inputs_list[1:]:
                if not isinstance(item, np.ndarray):
                    _set_invalid_argument(
                        grpc_context,
                        "Mixed input modes across batch requests: legacy "
                        "features and versioned inputs cannot be combined",
                    )
                    log_inference_audit(
                        logger,
                        context,
                        bundle_id=self._bundle_id,
                        model_version=self._model_version,
                        status="invalid_argument",
                        duration_ms=(time.perf_counter() - started) * 1000,
                    )
                    return self._response(context=context)
                if item.shape[1:] != inputs_list[0].shape[1:]:
                    _set_invalid_argument(
                        grpc_context,
                        f"Legacy feature matrix has shape {item.shape} but the "
                        f"first request carries shape {inputs_list[0].shape} "
                        "— non-batch dimensions must match across batch "
                        "requests",
                    )
                    log_inference_audit(
                        logger,
                        context,
                        bundle_id=self._bundle_id,
                        model_version=self._model_version,
                        status="invalid_argument",
                        duration_ms=(time.perf_counter() - started) * 1000,
                    )
                    return self._response(context=context)
            batch = np.concatenate([np.asarray(i) for i in inputs_list], axis=0)
            try:
                outputs = _run_or_invalid(self._run, batch, grpc_context)
            except Exception:
                log_inference_audit(
                    logger,
                    context,
                    bundle_id=self._bundle_id,
                    model_version=self._model_version,
                    status="error",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                raise

        if outputs is None:
            log_inference_audit(
                logger,
                context,
                bundle_id=self._bundle_id,
                model_version=self._model_version,
                status="invalid_argument",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return self._response(context=context)

        all_predictions = outputs[0].flatten().tolist()
        log_inference_audit(
            logger,
            context,
            bundle_id=self._bundle_id,
            model_version=self._model_version,
            status="ok",
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        return self._response(all_predictions, context=context)

    def close(self) -> None:
        """Release bundle resources (idempotent).

        Call when the gRPC deployment is torn down; prediction keeps
        working after close (in-memory model contract).  No-op on the
        legacy session path.
        """
        if self._runtime is not None:
            self._runtime.close()

    async def health(self) -> dict[str, Any]:
        """Health check.

        Returns:
            Health status information.
        """
        return {
            "status": "healthy",
            "model_loaded": self._runtime is not None or self._session is not None,
            "input_names": list(self._input_names),
            "bundle_id": self._bundle_id,
            "model_version": self._model_version,
        }
