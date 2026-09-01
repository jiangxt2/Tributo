"""Unit tests for strict, credential-free inference contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tributo.data import (
    DataWriteTargetRequest,
    IngestionPlanReceipt,
    IngestionRequest,
    WriteMode,
)
from tributo.data.source_config import ParquetSourceConfig, ProviderSourceConfig
from tributo.exporting.models import BundleRef
from tributo.inference.contracts import (
    ArtifactModelReference,
    BundleModelReference,
    FailureDiagnostic,
    InferenceRequest,
    InferenceResult,
    InputBindingSpec,
    OutputBindingSpec,
    ParquetResultSinkRequest,
    RegistryModelReference,
    ResultSinkReceipt,
    TensorInputBinding,
    TensorOutputBinding,
)


def _request(**updates) -> InferenceRequest:
    values = {
        "model": BundleModelReference(uri="/models/bundle"),
        "input": IngestionRequest(
            source=ParquetSourceConfig(path="/data/input.parquet"),
            engine="ray",
        ),
        "input_binding": InputBindingSpec(
            tensors=(
                TensorInputBinding(
                    tensor_name="float_input",
                    columns=("feature_b", "feature_a"),
                    dtype="float32",
                ),
            ),
            passthrough_columns=("entity_id", "feature_a"),
        ),
        "output_binding": OutputBindingSpec(
            tensors=(
                TensorOutputBinding(
                    tensor_name="probabilities",
                    column="score",
                    semantic="probability",
                ),
            )
        ),
        "result_sink": ParquetResultSinkRequest(uri="/data/output"),
    }
    values.update(updates)
    return InferenceRequest(**values)


class TestBundleModelReference:
    def test_binds_existing_beta_bundle_ref_without_extending_it(self) -> None:
        ref = BundleRef(
            canonical_uri="/models/bundle",
            bundle_id="bundle-1",
            manifest_sha256="a" * 64,
        )

        selection = BundleModelReference.from_bundle_ref(
            ref, role="inference", storage_profile="model-store"
        )

        assert selection.uri == ref.canonical_uri
        assert selection.expected_manifest_sha256 == ref.manifest_sha256
        assert set(ref.model_dump()) == {
            "canonical_uri",
            "bundle_id",
            "manifest_sha256",
        }


def test_generic_data_write_result_sink_is_target_neutral() -> None:
    request = DataWriteTargetRequest(
        target_kind="clickhouse",
        target="analytics.events",
        options={"operation": "append"},
    )

    assert request.sink_id == "data-write-v1"
    assert request.target_kind == "clickhouse"
    assert request.mode is WriteMode.APPEND


def test_generic_data_write_result_sink_rejects_target_credentials() -> None:
    with pytest.raises(ValidationError, match="credential-free"):
        DataWriteTargetRequest(
            target_kind="doris",
            target="doris://user:secret@host/events",
        )


def test_generic_data_write_result_sink_rejects_inline_runtime_credentials() -> None:
    with pytest.raises(ValidationError, match="credential"):
        DataWriteTargetRequest(
            target_kind="doris",
            target="analytics.events",
            runtime_options={"password": "secret"},
        )


class TestArtifactModelReference:
    @pytest.mark.parametrize(
        ("variant", "expected_format"),
        (("ubj", "ubj"), ("json", "xgboost-json")),
    )
    def test_first_party_legacy_xgboost_reference_is_normalised(
        self, variant: str, expected_format: str
    ) -> None:
        with pytest.warns(DeprecationWarning, match="deprecated"):
            reference = ArtifactModelReference(
                provider_id="tributo.artifact",
                uri="/models/model.xgb",
                format_id="xgboost",
                flavor_id="xgboost-native-v1",
                import_bundle_uri="/bundles/imported",
                options={"variant": variant, "input_fields": [], "custom": True},
            )

        assert reference.format_id == expected_format
        assert reference.flavor_id == "xgboost-native-v1"
        assert reference.options == {"input_fields": [], "custom": True}

    def test_third_party_xgboost_contract_is_not_rewritten(self) -> None:
        reference = ArtifactModelReference(
            provider_id="external.store",
            uri="/models/model.xgb",
            format_id="xgboost",
            flavor_id="xgboost-native-v1",
            import_bundle_uri="/bundles/imported",
            options={"variant": "json"},
        )

        assert reference.format_id == "xgboost"
        assert reference.options == {"variant": "json"}

    @pytest.mark.parametrize("variant", ("", "binary", 1, ["ubj"]))
    def test_first_party_legacy_xgboost_rejects_unknown_variant(
        self, variant: object
    ) -> None:
        with pytest.raises(ValidationError, match="must be 'ubj' or 'json'"):
            ArtifactModelReference(
                provider_id="tributo.artifact",
                uri="/models/model.xgb",
                format_id="xgboost",
                flavor_id="xgboost-native-v1",
                import_bundle_uri="/bundles/imported",
                options={"variant": variant},
            )


class TestBindings:
    def test_projection_is_feature_then_passthrough_and_deduplicated(self) -> None:
        request = _request()

        assert request.input_binding.projected_columns() == (
            "feature_b",
            "feature_a",
            "entity_id",
        )

    @pytest.mark.parametrize(
        "binding",
        [
            InputBindingSpec(
                tensors=(
                    TensorInputBinding(tensor_name="x", columns=("a",)),
                    TensorInputBinding(tensor_name="y", columns=("b",)),
                )
            ).model_dump()
            | {
                "tensors": [
                    {"tensor_name": "x", "columns": ["a"]},
                    {"tensor_name": "x", "columns": ["b"]},
                ]
            },
            {"tensors": [{"tensor_name": "x", "columns": ["a", "a"]}]},
        ],
    )
    def test_duplicate_input_bindings_fail(self, binding: dict) -> None:
        with pytest.raises(ValidationError):
            InputBindingSpec.model_validate(binding)

    def test_duplicate_result_columns_fail(self) -> None:
        with pytest.raises(ValidationError, match="result columns must be unique"):
            OutputBindingSpec(
                tensors=(
                    TensorOutputBinding(
                        tensor_name="label", column="result", semantic="label"
                    ),
                    TensorOutputBinding(
                        tensor_name="score", column="result", semantic="score"
                    ),
                )
            )

    def test_empty_passthrough_column_fails_during_contract_validation(self) -> None:
        with pytest.raises(ValidationError, match="passthrough columns"):
            InputBindingSpec(
                tensors=(TensorInputBinding(tensor_name="x", columns=("a",)),),
                passthrough_columns=("",),
            )

    def test_result_collision_with_retained_input_fails_before_execution(
        self,
    ) -> None:
        with pytest.raises(ValidationError, match="collide with retained"):
            _request(
                output_binding=OutputBindingSpec(
                    tensors=(
                        TensorOutputBinding(
                            tensor_name="probabilities",
                            column="entity_id",
                            semantic="probability",
                        ),
                    )
                )
            )

    @pytest.mark.parametrize("dtype", ["float", "object", "not-a-dtype"])
    def test_binding_dtype_must_be_canonical_and_supported(self, dtype: str) -> None:
        with pytest.raises(ValidationError, match="unsupported binding dtype"):
            TensorInputBinding(tensor_name="x", columns=("a",), dtype=dtype)
        with pytest.raises(ValidationError, match="unsupported binding dtype"):
            TensorOutputBinding(
                tensor_name="y", column="result", semantic="tensor", dtype=dtype
            )

    def test_single_column_mode_is_explicit_and_json_stable(self) -> None:
        vector = TensorInputBinding(tensor_name="x", columns=("a",))
        scalar = TensorInputBinding(
            tensor_name="y",
            columns=("b",),
            single_column_mode="scalar",
        )

        assert vector.single_column_mode == "vector"
        assert scalar.single_column_mode == "scalar"
        assert (
            TensorInputBinding.model_validate_json(scalar.model_dump_json()) == scalar
        )

    def test_scalar_single_column_mode_rejects_invalid_contracts(self) -> None:
        with pytest.raises(ValidationError, match="requires exactly one column"):
            TensorInputBinding(
                tensor_name="x",
                columns=("a", "b"),
                single_column_mode="scalar",
            )
        with pytest.raises(
            ValidationError, match="Input should be 'vector' or 'scalar'"
        ):
            TensorInputBinding.model_validate(
                {
                    "tensor_name": "x",
                    "columns": ["a"],
                    "single_column_mode": "matrix",
                }
            )

    def test_nan_policy_is_explicit_and_fail_closed_by_default(self) -> None:
        assert _request().input_binding.nan_policy == "error"
        assert (
            InputBindingSpec(
                tensors=(TensorInputBinding(tensor_name="x", columns=("a",)),),
                nan_policy="allow",
            ).nan_policy
            == "allow"
        )


class TestCredentialBoundary:
    @pytest.mark.parametrize(
        "updates, secret",
        [
            (
                {
                    "input": IngestionRequest(
                        source=ProviderSourceConfig(
                            provider="tributo.parquet",
                            uri="s3://bucket/input",
                            options={"s3": {"secret_access_key": "input-secret"}},
                        ),
                        engine="ray",
                    )
                },
                "input-secret",
            ),
            (
                {
                    "model": ArtifactModelReference(
                        provider_id="external.store",
                        uri="s3://bucket/model.onnx",
                        format_id="onnx",
                        flavor_id="onnx-runtime-v1",
                        import_bundle_uri="s3://bucket/imported",
                        options={"api_token": "model-secret"},
                    )
                },
                "model-secret",
            ),
            (
                {
                    "model": ArtifactModelReference(
                        provider_id="external.store",
                        uri="/model.onnx",
                        format_id="onnx",
                        flavor_id="onnx-runtime-v1",
                        import_bundle_uri="/bundle",
                        options={"headers": {"Authorization": "Bearer model-secret"}},
                    )
                },
                "model-secret",
            ),
        ],
    )
    def test_plaintext_credentials_are_rejected_without_echoing_values(
        self, updates: dict, secret: str
    ) -> None:
        with pytest.raises(ValidationError) as error:
            _request(**updates)

        assert "plaintext credentials" in str(error.value)
        assert secret not in str(error.value)

    def test_separate_profile_references_are_serializable(self) -> None:
        request = _request(
            model=BundleModelReference(
                uri="s3://models/bundle", storage_profile="model-domain"
            ),
            input=IngestionRequest(
                source=ParquetSourceConfig(path="s3://source/input.parquet"),
                engine="ray",
                storage_profile="source-domain",
            ),
            result_sink=ParquetResultSinkRequest(
                uri="s3://results/output", storage_profile="sink-domain"
            ),
        )

        payload = request.model_dump_json()
        assert "model-domain" in payload
        assert "source-domain" in payload
        assert "sink-domain" in payload
        assert "secret_access_key" not in payload

    def test_non_json_model_options_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="JSON-serializable"):
            _request(
                model=ArtifactModelReference(
                    provider_id="external.store",
                    uri="/model.onnx",
                    format_id="onnx",
                    flavor_id="onnx-runtime-v1",
                    import_bundle_uri="/bundle",
                    options={"object": object()},
                )
            )

    @pytest.mark.parametrize(
        "uri",
        [
            "https://results.example/output",
            "s3:///missing-bucket",
            "s3://bucket/output?version=1",
            "file://remote-host/output",
        ],
    )
    def test_parquet_sink_uri_rejects_unsupported_or_ambiguous_targets(
        self, uri: str
    ) -> None:
        with pytest.raises(ValidationError):
            ParquetResultSinkRequest(uri=uri)

    def test_parquet_sink_uri_error_does_not_echo_signed_credentials(self) -> None:
        with pytest.raises(ValidationError) as error:
            ParquetResultSinkRequest(
                uri="s3://bucket/output?X-Amz-Signature=must-not-leak"
            )

        assert "must-not-leak" not in str(error.value)

    def test_daft_input_is_rejected_without_implicit_conversion(self) -> None:
        with pytest.raises(ValidationError, match="Daft-to-Ray conversion"):
            _request(
                input=IngestionRequest(
                    source=ParquetSourceConfig(path="/data/input.parquet"),
                    engine="daft",
                )
            )

    def test_input_trace_context_is_reserved(self) -> None:
        with pytest.raises(ValidationError, match="trace_context is reserved"):
            _request(
                input=IngestionRequest(
                    source=ParquetSourceConfig(path="/data/input.parquet"),
                    engine="ray",
                    trace_context={"trace_id": "trace-1"},
                )
            )


class TestInferenceResult:
    def test_success_requires_sink_receipt(self) -> None:
        with pytest.raises(ValidationError, match="requires a sink receipt"):
            InferenceResult(
                run_id="run-1",
                attempt_id="attempt-1",
                submission_id="submission-1",
                plan_digest="a" * 64,
                bundle_id="bundle-1",
                manifest_sha256="b" * 64,
                role="inference",
                flavor_id="onnx-runtime-v1",
                source_ref_id="c" * 64,
                status="succeeded",
            )

    def test_success_requires_ingestion_receipt(self) -> None:
        with pytest.raises(ValidationError, match="requires an ingestion receipt"):
            InferenceResult(
                run_id="run-1",
                attempt_id="attempt-1",
                submission_id="submission-1",
                plan_digest="a" * 64,
                bundle_id="bundle-1",
                manifest_sha256="b" * 64,
                role="inference",
                flavor_id="onnx-runtime-v1",
                source_ref_id="c" * 64,
                sink_receipt=ResultSinkReceipt(
                    sink_id="parquet-v1",
                    result_id="d" * 64,
                    uri="/data/output",
                ),
                status="succeeded",
            )

    def test_success_carries_both_domain_receipts(self) -> None:
        result = InferenceResult(
            run_id="run-1",
            attempt_id="attempt-1",
            submission_id="submission-1",
            plan_digest="a" * 64,
            bundle_id="bundle-1",
            manifest_sha256="b" * 64,
            role="inference",
            flavor_id="onnx-runtime-v1",
            source_ref_id="c" * 64,
            ingestion_receipt=_ingestion_receipt(),
            sink_receipt=ResultSinkReceipt(
                sink_id="parquet-v1",
                result_id="d" * 64,
                uri="/data/output",
            ),
            status="succeeded",
        )

        assert result.ingestion_receipt is not None
        assert result.ingestion_receipt.source_ref == result.source_ref_id

    def test_success_cannot_be_retryable(self) -> None:
        with pytest.raises(
            ValidationError, match="succeeded inference cannot be retryable"
        ):
            InferenceResult(
                run_id="run-1",
                attempt_id="attempt-1",
                submission_id="submission-1",
                plan_digest="a" * 64,
                bundle_id="bundle-1",
                manifest_sha256="b" * 64,
                role="inference",
                flavor_id="onnx-runtime-v1",
                source_ref_id="c" * 64,
                ingestion_receipt=_ingestion_receipt(),
                sink_receipt=ResultSinkReceipt(
                    sink_id="parquet-v1",
                    result_id="d" * 64,
                    uri="/data/output",
                ),
                status="succeeded",
                retryable=True,
            )

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ({"retryable": True}, "cancelled inference cannot be retryable"),
            (
                {
                    "failure": FailureDiagnostic(
                        phase="execution",
                        code="inference_execution_failed",
                        error_type="RuntimeError",
                    )
                },
                "cancelled inference cannot carry a failure",
            ),
        ],
    )
    def test_cancelled_has_no_failure_or_retry_semantics(
        self, payload: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValidationError, match=message):
            InferenceResult(
                run_id="run-1",
                attempt_id="attempt-1",
                submission_id="submission-1",
                plan_digest="a" * 64,
                bundle_id="bundle-1",
                manifest_sha256="b" * 64,
                role="inference",
                flavor_id="onnx-runtime-v1",
                source_ref_id="c" * 64,
                status="cancelled",
                **payload,
            )

    def test_receipt_uri_rejects_signed_credentials(self) -> None:
        with pytest.raises(ValidationError, match="credential-free"):
            ResultSinkReceipt(
                sink_id="parquet-v1",
                result_id="d" * 64,
                uri="https://sink/out?X-Amz-Credential=secret",
            )

    def test_receipt_metadata_rejects_plaintext_credentials(self) -> None:
        with pytest.raises(ValidationError, match="credential-free") as error:
            ResultSinkReceipt(
                sink_id="parquet-v1",
                result_id="d" * 64,
                uri="/data/output",
                metadata={"access_token": "must-not-leak"},
            )

        assert "must-not-leak" not in str(error.value)

    def test_result_metrics_reject_plaintext_credentials(self) -> None:
        with pytest.raises(ValidationError, match="credential-free") as error:
            InferenceResult(
                run_id="run-1",
                attempt_id="attempt-1",
                submission_id="submission-1",
                plan_digest="a" * 64,
                bundle_id="bundle-1",
                manifest_sha256="b" * 64,
                role="inference",
                flavor_id="onnx-runtime-v1",
                source_ref_id="c" * 64,
                ingestion_receipt=_ingestion_receipt(),
                sink_receipt=ResultSinkReceipt(
                    sink_id="parquet-v1",
                    result_id="d" * 64,
                    uri="/data/output",
                ),
                metrics={"client_secret": "must-not-leak"},
                status="succeeded",
            )

        assert "must-not-leak" not in str(error.value)


class TestModelReferences:
    def test_registry_selector_must_be_exactly_version_or_alias(self) -> None:
        with pytest.raises(ValidationError, match="exactly one"):
            RegistryModelReference(
                provider_id="mlflow.v2",
                model_name="classifier",
                import_bundle_uri="/imported",
            )
        with pytest.raises(ValidationError, match="exactly one"):
            RegistryModelReference(
                provider_id="mlflow.v2",
                model_name="classifier",
                version="4",
                alias="champion",
                import_bundle_uri="/imported",
            )

    def test_unknown_request_field_fails(self) -> None:
        payload = _request().model_dump(mode="python")
        payload["unexpected"] = True

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            InferenceRequest.model_validate(payload)


def _ingestion_receipt() -> IngestionPlanReceipt:
    return IngestionPlanReceipt(
        request_digest="1" * 64,
        engine_id="tributo.ray_data",
        engine_version="2.55.1",
        provider_id="tributo.parquet",
        connector_id="parquet",
        binding_id="tributo.ray.parquet",
        scan_kind="file",
        logical_plan_version=1,
        logical_plan_digest="2" * 64,
        source_ref="c" * 64,
        dataset_ref="3" * 64,
        transform_ir_version=1,
        transform_digest="4" * 64,
        binding_distribution="tributo",
        binding_distribution_version="1.0.0",
        reader_api="ray.data.read_parquet",
        transport_id="ray-data",
    )
