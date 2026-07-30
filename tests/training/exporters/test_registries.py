"""Tests for export registries — registration, candidate selection, diagnostics."""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

import pytest
from pydantic import BaseModel

from tributo.exceptions import JobConfigurationError
from tributo.training.exporters.models import (
    ExportContext,
    ExportSource,
    ExportTarget,
    PlannedTarget,
    ResolvedArtifact,
    SupportRequest,
    SupportResult,
    ValidationResult,
    ValidatorBinding,
)
from tributo.training.exporters.registries import (
    ExportRegistry,
    FlavorRegistry,
    ModelFactoryRegistry,
    SourceProviderRegistry,
    ValidatorRegistry,
    select_candidate,
)

# ── Fake implementations for testing ─────────────────────────────────────────


class _MinimalOptions(BaseModel):
    """Concrete options model for test exporters."""

    model_config = {"extra": "forbid"}


class _FakeONNXExporter:
    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "torch-onnx-v1"
    priority: ClassVar[int] = 10
    output_format: ClassVar[str] = "onnx"
    options_model: ClassVar[type[BaseModel]] = _MinimalOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = False

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        if request.source_kind == "pytorch_result":
            return SupportResult(supported=True, code="OK")
        return SupportResult(
            supported=False,
            code="UNSUPPORTED_SOURCE",
            reason=f"Cannot handle source_kind={request.source_kind!r}",
        )

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> Any:
        raise NotImplementedError


class _FakeXGBoostONNXExporter:
    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "xgboost-onnx-v1"
    priority: ClassVar[int] = 10
    output_format: ClassVar[str] = "onnx"
    options_model: ClassVar[type[BaseModel]] = _MinimalOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = True  # mutates feature_names

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        if request.source_kind == "xgboost_result":
            task = request.source_metadata.get("task_type", "")
            if task == "regression":
                return SupportResult(
                    supported=False,
                    code="UNSUPPORTED_TASK",
                    reason="Only classification is supported",
                )
            return SupportResult(supported=True, code="OK")
        return SupportResult(
            supported=False,
            code="UNSUPPORTED_SOURCE",
            reason="Only xgboost_result is supported",
        )

    def export(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _FakeLowPriorityONNXExporter:
    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "legacy-onnx-v1"
    priority: ClassVar[int] = 5
    output_format: ClassVar[str] = "onnx"
    options_model: ClassVar[type[BaseModel]] = _MinimalOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = False

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        return SupportResult(supported=True, code="OK")

    def export(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _FakeTiedPriorityONNXExporter:
    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "another-onnx-v1"
    priority: ClassVar[int] = 10  # same as torch-onnx-v1
    output_format: ClassVar[str] = "onnx"
    options_model: ClassVar[type[BaseModel]] = _MinimalOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = False

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        return SupportResult(supported=True, code="OK")

    def export(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _FakeSourceProvider:
    api_version: ClassVar[int] = 1
    provider_id: ClassVar[str] = "xgboost-provider-v1"
    trainer_type: ClassVar[str] = "xgboost"
    priority: ClassVar[int] = 10

    def open_source(self, result: Any, config: BaseModel) -> Any:
        raise NotImplementedError


class _FakeValidator:
    api_version: ClassVar[int] = 1
    validator_id: ClassVar[str] = "structure-v1"
    options_model: ClassVar[type[BaseModel]] = _MinimalOptions

    def validate(self, *args: Any, **kwargs: Any) -> ValidationResult:
        raise NotImplementedError


class _FakeFactory:
    api_version: ClassVar[int] = 1
    architecture_id: ClassVar[str] = "tributo-dnn-v1"

    def build(self, model_config: dict[str, Any]) -> Any:
        raise NotImplementedError


# ── Tests ────────────────────────────────────────────────────────────────────


class TestExportRegistry:
    def test_register_and_get(self) -> None:
        reg = ExportRegistry()
        reg.register(_FakeONNXExporter)
        assert reg.contains("torch-onnx-v1")
        assert reg.get("torch-onnx-v1") is _FakeONNXExporter

    def test_unknown_raises(self) -> None:
        reg = ExportRegistry()
        with pytest.raises(JobConfigurationError, match="Unknown exporter"):
            reg.get("nonexistent")

    def test_list_all(self) -> None:
        reg = ExportRegistry()
        reg.register(_FakeONNXExporter)
        reg.register(_FakeXGBoostONNXExporter)
        assert set(reg.list_all()) == {"torch-onnx-v1", "xgboost-onnx-v1"}

    def test_list_candidates_filters_by_format(self) -> None:
        reg = ExportRegistry()
        reg.register(_FakeONNXExporter)
        reg.register(_FakeXGBoostONNXExporter)
        candidates = reg.list_candidates(
            source_kind="pytorch_result", output_format="onnx"
        )
        assert len(candidates) == 2

    def test_list_candidates_sorted_by_priority(self) -> None:
        reg = ExportRegistry()
        reg.register(_FakeLowPriorityONNXExporter)  # priority 5
        reg.register(_FakeONNXExporter)  # priority 10
        candidates = reg.list_candidates(source_kind="any", output_format="onnx")
        assert candidates[0].exporter_id == "torch-onnx-v1"
        assert candidates[1].exporter_id == "legacy-onnx-v1"

    def test_duplicate_detection_disables_both(self) -> None:
        reg = ExportRegistry()
        reg.register(_FakeONNXExporter)
        reg.register(_FakeONNXExporter)  # same exporter_id
        assert not reg.contains("torch-onnx-v1")
        assert len(reg.diagnostics()) == 1
        assert "Duplicate" in reg.diagnostics()[0].reason

    def test_unregister(self) -> None:
        reg = ExportRegistry()
        reg.register(_FakeONNXExporter)
        reg.unregister("torch-onnx-v1")
        assert not reg.contains("torch-onnx-v1")

    def test_no_candidates(self) -> None:
        reg = ExportRegistry()
        candidates = reg.list_candidates(source_kind="any", output_format="onnx")
        assert candidates == []


class TestSourceProviderRegistry:
    def test_register_and_resolve(self) -> None:
        reg = SourceProviderRegistry()
        reg.register(_FakeSourceProvider)
        resolved = reg.resolve("xgboost")
        assert resolved is _FakeSourceProvider

    def test_no_provider_raises(self) -> None:
        reg = SourceProviderRegistry()
        with pytest.raises(JobConfigurationError, match="No SourceProvider"):
            reg.resolve("unknown")

    def test_duplicate_detection(self) -> None:
        reg = SourceProviderRegistry()
        reg.register(_FakeSourceProvider)
        reg.register(_FakeSourceProvider)
        assert len(reg.diagnostics()) > 0

    def test_ambiguous_priority(self) -> None:
        class _ProviderA:
            api_version: ClassVar[int] = 1
            provider_id: ClassVar[str] = "a-v1"
            trainer_type: ClassVar[str] = "xgboost"
            priority: ClassVar[int] = 10

        class _ProviderB:
            api_version: ClassVar[int] = 1
            provider_id: ClassVar[str] = "b-v1"
            trainer_type: ClassVar[str] = "xgboost"
            priority: ClassVar[int] = 10

        reg = SourceProviderRegistry()
        reg.register(_ProviderA)
        reg.register(_ProviderB)
        with pytest.raises(JobConfigurationError, match="Ambiguous"):
            reg.resolve("xgboost")


class TestValidatorRegistry:
    def test_register_and_get(self) -> None:
        reg = ValidatorRegistry()
        reg.register(_FakeValidator)
        assert reg.get("structure-v1") is _FakeValidator

    def test_unknown_raises(self) -> None:
        reg = ValidatorRegistry()
        with pytest.raises(JobConfigurationError, match="Unknown validator"):
            reg.get("nonexistent")


class TestFlavorRegistry:
    def test_register_and_get(self) -> None:
        class _FakeFlavor:
            flavor_id: ClassVar[str] = "onnx-runtime-v1"

        reg = FlavorRegistry()
        reg.register(_FakeFlavor)
        assert reg.get("onnx-runtime-v1") is _FakeFlavor

    def test_duplicate_detection(self) -> None:
        class _FakeFlavorA:
            flavor_id: ClassVar[str] = "onnx-runtime-v1"

        class _FakeFlavorB:
            flavor_id: ClassVar[str] = "onnx-runtime-v1"

        reg = FlavorRegistry()
        reg.register(_FakeFlavorA)
        reg.register(_FakeFlavorB)
        assert not reg.list_all()
        assert len(reg.diagnostics()) > 0


class TestModelFactoryRegistry:
    def test_register_and_get(self) -> None:
        reg = ModelFactoryRegistry()
        reg.register(_FakeFactory)
        assert reg.get("tributo-dnn-v1") is _FakeFactory

    def test_unknown_raises(self) -> None:
        reg = ModelFactoryRegistry()
        with pytest.raises(JobConfigurationError, match="Unknown architecture"):
            reg.get("unknown-arch")


class TestSelectCandidate:
    def test_selects_highest_priority(self) -> None:
        target = ExportTarget(name="fp32", format="onnx")
        request = SupportRequest(source_kind="pytorch_result")
        validator_reg = ValidatorRegistry()

        result = select_candidate(
            [_FakeONNXExporter, _FakeLowPriorityONNXExporter],
            target,
            request,
            validator_reg,
        )
        assert result.exporter_id == "torch-onnx-v1"

    def test_explicit_exporter_id(self) -> None:
        target = ExportTarget(name="fp32", format="onnx", exporter_id="legacy-onnx-v1")
        request = SupportRequest(source_kind="pytorch_result")
        validator_reg = ValidatorRegistry()

        result = select_candidate(
            [_FakeONNXExporter, _FakeLowPriorityONNXExporter],
            target,
            request,
            validator_reg,
        )
        assert result.exporter_id == "legacy-onnx-v1"

    def test_explicit_not_found(self) -> None:
        target = ExportTarget(name="fp32", format="onnx", exporter_id="nonexistent")
        request = SupportRequest(source_kind="pytorch_result")
        validator_reg = ValidatorRegistry()

        with pytest.raises(JobConfigurationError, match="not found"):
            select_candidate([_FakeONNXExporter], target, request, validator_reg)

    def test_explicit_not_supported(self) -> None:
        target = ExportTarget(name="fp32", format="onnx", exporter_id="xgboost-onnx-v1")
        request = SupportRequest(source_kind="pytorch_result")
        validator_reg = ValidatorRegistry()

        with pytest.raises(JobConfigurationError, match="does not support"):
            select_candidate(
                [_FakeONNXExporter, _FakeXGBoostONNXExporter],
                target,
                request,
                validator_reg,
            )

    def test_regression_rejected(self) -> None:
        target = ExportTarget(name="fp32", format="onnx")
        request = SupportRequest(
            source_kind="xgboost_result",
            source_metadata={"task_type": "regression"},
        )
        validator_reg = ValidatorRegistry()

        with pytest.raises(JobConfigurationError, match="UNSUPPORTED_TASK"):
            select_candidate([_FakeXGBoostONNXExporter], target, request, validator_reg)

    def test_tied_priority_ambiguous(self) -> None:
        target = ExportTarget(name="fp32", format="onnx")
        request = SupportRequest(source_kind="pytorch_result")
        validator_reg = ValidatorRegistry()

        with pytest.raises(JobConfigurationError, match="Ambiguous"):
            select_candidate(
                [_FakeONNXExporter, _FakeTiedPriorityONNXExporter],
                target,
                request,
                validator_reg,
            )

    def test_no_supported_candidates(self) -> None:
        target = ExportTarget(name="fp32", format="onnx")
        request = SupportRequest(source_kind="unknown_kind")
        validator_reg = ValidatorRegistry()

        with pytest.raises(JobConfigurationError, match="No exporter supports"):
            select_candidate(
                [_FakeONNXExporter, _FakeXGBoostONNXExporter],
                target,
                request,
                validator_reg,
            )
