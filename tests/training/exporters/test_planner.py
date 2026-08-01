"""Tests for ExportPlanner — plan construction, implicit nodes, cycles."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from tributo.exceptions import JobConfigurationError
from tributo.exporting.models import (
    BundleOutputConfig,
    ExportSource,
    ExportTarget,
    SupportRequest,
    SupportResult,
    UpstreamRequirement,
    ValidatorBinding,
)
from tributo.exporting.planner import ExportPlanner
from tributo.exporting.registries import (
    ExportRegistry,
    ValidatorRegistry,
)

# ── Fake exporters ───────────────────────────────────────────────────────────


class _MinOpts(BaseModel):
    model_config = {"extra": "forbid"}


class _QuantizerOpts(BaseModel):
    """Options model for the fake quantizer — accepts quantization config."""

    model_config = {"extra": "forbid"}
    quantization: dict[str, Any] | None = None


class _ExporterONNX:
    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "torch-onnx-v1"
    priority: ClassVar[int] = 100
    output_format: ClassVar[str] = "onnx"
    options_model: ClassVar[type[BaseModel]] = _MinOpts
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[UpstreamRequirement, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        return SupportResult(supported=True, code="OK")

    def export(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _ExporterXGBoost:
    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "xgboost-native-v1"
    priority: ClassVar[int] = 100
    output_format: ClassVar[str] = "xgboost"
    options_model: ClassVar[type[BaseModel]] = _MinOpts
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[UpstreamRequirement, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        return SupportResult(supported=True, code="OK")

    def export(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _ExporterONNXQuantizer:
    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "onnx-quantizer-v1"
    priority: ClassVar[int] = 90
    output_format: ClassVar[str] = "onnx"
    options_model: ClassVar[type[BaseModel]] = _QuantizerOpts
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[UpstreamRequirement, ...]] = (
        UpstreamRequirement(
            name="model",
            format="onnx",
            options={"quantization": None},
        ),
    )

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        # Accept any source_kind for testing purposes.
        return SupportResult(supported=True, code="OK")

    def export(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


# ── Helper ───────────────────────────────────────────────────────────────────


def _make_planner() -> ExportPlanner:
    er = ExportRegistry()
    er.register(_ExporterONNX)
    er.register(_ExporterXGBoost)
    er.register(_ExporterONNXQuantizer)
    return ExportPlanner(er, ValidatorRegistry())


def _make_source() -> ExportSource:
    return ExportSource(
        source_kind="pytorch_result",
        metadata={"task_type": "classification"},
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestPlanSimple:
    def test_single_target(self) -> None:
        planner = _make_planner()
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[ExportTarget(name="fp32", format="onnx")],
        )
        plan = planner.plan(cfg, _make_source())
        assert len(plan.nodes) == 1
        assert plan.nodes[0].target.name == "fp32"
        assert plan.nodes[0].exporter_id == "torch-onnx-v1"

    def test_two_independent_targets(self) -> None:
        planner = _make_planner()
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[
                ExportTarget(name="native", format="xgboost"),
                ExportTarget(name="onnx", format="onnx"),
            ],
        )
        plan = planner.plan(cfg, _make_source())
        assert len(plan.nodes) == 2

    def test_legacy_mode_rejects(self) -> None:
        planner = _make_planner()
        cfg = BundleOutputConfig()
        with pytest.raises(JobConfigurationError, match="legacy mode"):
            planner.plan(cfg, _make_source())


class TestCycleDetection:
    def test_simple_cycle_detected_at_plan_time(self) -> None:
        planner = _make_planner()
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[
                ExportTarget(name="a", format="onnx", depends_on=("b",)),
                ExportTarget(name="b", format="xgboost", depends_on=("a",)),
            ],
        )
        with pytest.raises(JobConfigurationError, match="Cycle detected"):
            planner.plan(cfg, _make_source())

    def test_self_cycle_rejected_at_config_time(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="cannot depend on itself"):
            BundleOutputConfig(
                bundle_uri="/tmp/bundle",
                targets=[ExportTarget(name="a", format="onnx", depends_on=("a",))],
            )


class TestDependencyOrdering:
    def test_linear_dag_order(self) -> None:
        planner = _make_planner()
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[
                ExportTarget(name="step2", format="onnx", depends_on=("step1",)),
                ExportTarget(name="step1", format="xgboost"),
            ],
        )
        plan = planner.plan(cfg, _make_source())
        # step1 must execute before step2.
        names = [n.target.name for n in plan.nodes]
        step1_idx = names.index("step1")
        step2_idx = names.index("step2")
        assert step1_idx < step2_idx

    def test_diamond_dag(self) -> None:
        planner = _make_planner()
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[
                ExportTarget(name="left", format="onnx"),
                ExportTarget(name="right", format="xgboost"),
                ExportTarget(name="sink", format="onnx", depends_on=("left", "right")),
            ],
        )
        plan = planner.plan(cfg, _make_source())
        names = [n.target.name for n in plan.nodes]
        sink_idx = names.index("sink")
        left_idx = names.index("left")
        right_idx = names.index("right")
        assert left_idx < sink_idx
        assert right_idx < sink_idx


class TestNoCandidate:
    def test_unknown_format(self) -> None:
        planner = _make_planner()
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[ExportTarget(name="a", format="unknown_fmt")],
        )
        with pytest.raises(JobConfigurationError, match="No candidates"):
            planner.plan(cfg, _make_source())


class TestExplicitTargets:
    def test_explicit_target_map(self) -> None:
        planner = _make_planner()
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[ExportTarget(name="fp32", format="onnx")],
        )
        plan = planner.plan(cfg, _make_source())
        assert "fp32" in plan.explicit_node_map


class TestImplicitInjection:
    """Tests for upstream_requirements-based implicit node injection."""

    def test_quantizer_injects_fp32_onnx_node(self) -> None:
        """Quantizer declares upstream_requirements → planner injects FP32 node."""
        planner = _make_planner()
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[
                ExportTarget(
                    name="quantized",
                    format="onnx",
                    exporter_id="onnx-quantizer-v1",
                    depends_on=("model",),
                    options={"quantization": {"bits": 8}},
                ),
            ],
        )
        plan = planner.plan(cfg, _make_source())
        # Should have two nodes: implicit FP32 + explicit quantized.
        assert len(plan.nodes) == 2

        implicit = plan.nodes[0]
        explicit = plan.nodes[1]

        assert implicit.implicit is True
        assert implicit.publish is False
        assert implicit.target.format == "onnx"
        # Quantization should be stripped from implicit node options.
        assert "quantization" not in implicit.target.options
        # Implicit node should not use the quantizer.
        assert implicit.exporter_id != "onnx-quantizer-v1"

        assert explicit.implicit is False
        assert explicit.publish is True
        assert explicit.exporter_id == "onnx-quantizer-v1"
        # Quantized node depends on the implicit node.
        assert explicit.target.depends_on == (implicit.target.name,)

    def test_missing_requirement_raises(self) -> None:
        """depends_on names an unknown dep → error."""
        planner = _make_planner()
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[
                ExportTarget(
                    name="fp32",
                    format="onnx",
                    depends_on=("nonexistent",),
                ),
            ],
        )
        with pytest.raises(JobConfigurationError, match="nonexistent"):
            planner.plan(cfg, _make_source())

    def test_implicit_node_not_in_explicit_map(self) -> None:
        """Implicit nodes are excluded from explicit_node_map."""
        planner = _make_planner()
        cfg = BundleOutputConfig(
            bundle_uri="/tmp/bundle",
            targets=[
                ExportTarget(
                    name="quantized",
                    format="onnx",
                    exporter_id="onnx-quantizer-v1",
                    depends_on=("model",),
                    options={"quantization": {"bits": 8}},
                ),
            ],
        )
        plan = planner.plan(cfg, _make_source())
        assert "quantized" in plan.explicit_node_map
        # Implicit node name starts with _implicit__
        for name in plan.explicit_node_map:
            assert not name.startswith("_implicit__")
