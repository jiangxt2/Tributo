"""Focused Core tests for the unified Torch v1 public contract."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    DistributionStrategy,
    InputBinding,
    MetricReduction,
    ResultPolicy,
    SingleStageTorchPlan,
    TorchAccumulationWindow,
    TorchBackwardContext,
    TorchCheckpointDescriptor,
    TorchCheckpointPayloadDraft,
    TorchCompositeLossContribution,
    TorchDatasetRoute,
    TorchGlobalLossReduction,
    TorchLossContribution,
    TorchMetricContribution,
    TorchMetricPolicy,
    TorchMetricReductionContext,
    TorchPolicy,
    TorchStageRunIdentity,
    TorchStageSpec,
    apply_torch_loss_backward,
    reduce_torch_metrics,
    report_torch_checkpoint,
)
from tributo.algorithms.spi import (
    TorchOptimizationPlan,
    TorchRuntimeContext,
    TorchStageContext,
)


class _Scalar:
    ndim = 0

    def __init__(self, value: float) -> None:
        self.value = value

    def detach(self) -> "_Scalar":
        return self

    def item(self) -> float:
        return self.value


def test_torch_policy_and_run_name_are_deterministic() -> None:
    route = TorchDatasetRoute("train", "split_exact")
    execution_plan = SingleStageTorchPlan(stage=TorchStageSpec("train", ("train",)))
    policy = TorchPolicy(
        torch_runtime_api_version=1,
        loop_owner="core_recipe",
        parallelism_id="torch.ddp.replicated",
        dataset_routing=(route,),
        execution_plan=execution_plan,
        state_layout="replicated",
        metric_reducers={"train_loss": MetricReduction.SUM_COUNT},
    )
    assert policy.digest == TorchPolicy.from_dict(policy.to_dict()).digest
    identity = TorchStageRunIdentity(
        "aabbccdd",
        "11223344",
        "train",
        1,
        "example",
        "example.recipe",
        "0" * 64,
        policy.digest,
        execution_plan.digest,
    )
    assert identity.run_config_name.startswith("tributo-torch-v1-")


def test_loss_contribution_requires_zero_dimensional_scalar() -> None:
    assert TorchLossContribution(_Scalar(2.0), 3).normalizer == 3.0
    with pytest.raises(AlgorithmConfigurationError):
        TorchLossContribution(2.0, 3)
    composite = TorchCompositeLossContribution(
        "schema",
        {"loss_a": _Scalar(2.0), "loss_b": _Scalar(1.0), "loss_c": _Scalar(3.0)},
        {"count_a": 2, "count_b": 4},
    )
    assert set(composite.differentiable_components) == {"loss_a", "loss_b", "loss_c"}


@pytest.mark.parametrize(
    "max_gradient_norm",
    [True, 0, -1, float("nan"), float("inf"), float("-inf")],
)
def test_torch_optimization_plan_rejects_invalid_gradient_clip_norm(
    max_gradient_norm: float | int | bool,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        TorchOptimizationPlan(
            optimizer=object(),
            max_gradient_norm=max_gradient_norm,
        )


def test_torch_optimization_plan_accepts_optional_positive_gradient_clip_norm() -> None:
    assert (
        TorchOptimizationPlan(
            optimizer=object(), max_gradient_norm=None
        ).max_gradient_norm
        is None
    )
    assert (
        TorchOptimizationPlan(
            optimizer=object(), max_gradient_norm=1.0
        ).max_gradient_norm
        == 1.0
    )


def test_backward_and_metric_helpers_use_explicit_normalizers() -> None:
    events: list[object] = []
    result = apply_torch_loss_backward(
        TorchLossContribution(_Scalar(1.0), 2),
        TorchAccumulationWindow(index=0, expected_micro_batches=1),
        TorchBackwardContext(
            world_size=2,
            backward=lambda value: events.append(("backward", value)),
            reduce_normalizer=lambda value: value * 3,
            finalize_window=lambda scale: events.append(("scale", scale)),
        ),
    )
    assert result.global_normalizer == 6
    assert events[-1] == ("scale", 2 / 6)
    metrics = reduce_torch_metrics(
        {"loss": TorchMetricContribution(4, 2)},
        TorchMetricPolicy({"loss": "sum_count"}),
        TorchMetricReductionContext(
            lambda name, value, reducer: value.numerator / value.normalizer
        ),
    )
    assert metrics.values == {"loss": 2.0}


def test_backward_helper_reduces_only_the_accumulation_window_total() -> None:
    reductions: list[float] = []
    scales: list[float] = []
    context = TorchBackwardContext(
        world_size=2,
        backward=lambda value: None,
        reduce_normalizer=lambda value: reductions.append(value) or value * 2,
        finalize_window=scales.append,
    )
    first = apply_torch_loss_backward(
        TorchLossContribution(_Scalar(1), 2),
        TorchAccumulationWindow(0, 2),
        context,
    )
    assert not first.window_complete
    assert reductions == []
    second = apply_torch_loss_backward(
        TorchLossContribution(_Scalar(1), 3),
        TorchAccumulationWindow(0, 2, 1, first.global_normalizer),
        context,
    )
    assert second.window_complete
    assert reductions == [5]
    assert scales == [2 / 10]


def test_policy_replicate_budget_is_explicit() -> None:
    route = TorchDatasetRoute(
        "nodes", "replicate", max_rows=10, max_bytes_per_worker=10
    )
    plan = SingleStageTorchPlan(stage=TorchStageSpec("train", ("nodes",)))
    with pytest.raises(AlgorithmConfigurationError):
        TorchPolicy(
            1,
            "core_recipe",
            "torch.ddp.replicated",
            (route,),
            plan,
            "replicated",
            {"train_loss": MetricReduction.SUM_COUNT},
        )


def test_torch_v1_rejects_cross_run_recovery() -> None:
    route = TorchDatasetRoute("train", "split_exact")
    plan = SingleStageTorchPlan(stage=TorchStageSpec("train", ("train",)))
    with pytest.raises(AlgorithmConfigurationError, match="cross-Run"):
        TorchPolicy(
            1,
            "core_recipe",
            "torch.ddp.replicated",
            (route,),
            plan,
            "replicated",
            {"train_loss": MetricReduction.SUM_COUNT},
            resume_supported=True,
        )


def test_composite_global_state_keeps_component_and_normalizer_names_independent() -> (
    None
):
    from tributo.algorithms.api import TorchCompositeGlobalState

    state = TorchCompositeGlobalState(
        components={"positive": 2.0, "negative": -1.0},
        normalizers={"positive_count": 3.0, "negative_count": 4.0},
    )
    assert set(state.components) == {"positive", "negative"}
    assert set(state.normalizers) == {"positive_count", "negative_count"}


def test_role_evidence_requires_an_actual_role_binding() -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        _binding_digest_for_role,
    )

    class Descriptors:
        def get(self, role: str) -> object:
            raise AlgorithmConfigurationError(f"unknown resolved input role: {role}")

    plan = SimpleNamespace(
        input_descriptors=Descriptors(),
    )
    with pytest.raises(AlgorithmConfigurationError, match="no binding digest"):
        _binding_digest_for_role(plan, "val")


def test_worker_evidence_defaults_only_missing_declared_resources() -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        _normalize_worker_evidence,
    )

    plan = SimpleNamespace(
        runtime=SimpleNamespace(
            num_cpus=2.0,
            num_gpus=1.0,
            custom_resources={"accelerator": 1.0},
            memory_bytes=1024,
        )
    )
    existing = {"num_cpus": 9.0, "num_gpus": 0.0, "custom": {}}
    records = _normalize_worker_evidence(
        [{"worker_id": "a"}, {"worker_id": "b", "resources": existing}],
        plan,
    )
    assert records[0]["resources"] == {
        "num_cpus": 2.0,
        "num_gpus": 1.0,
        "custom": {"accelerator": 1.0},
        "memory_bytes": 1024,
    }
    assert records[1]["resources"] is existing


def test_component_state_details_project_stage_coverage() -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        _component_state_details,
    )

    stages = (
        SimpleNamespace(
            stage_id="pretrain",
            state_digest="a" * 64,
            roles=(SimpleNamespace(role="train", present=True, observed_rows=16),),
            workers=(),
            to_dict=lambda: {"stage_id": "pretrain", "state": "a" * 64},
        ),
        SimpleNamespace(
            stage_id="finetune",
            state_digest="b" * 64,
            roles=(SimpleNamespace(role="train", present=True, observed_rows=12),),
            workers=(),
            to_dict=lambda: {"stage_id": "finetune", "state": "b" * 64},
        ),
    )
    details = _component_state_details(stages, "pretrain")
    assert details["component_stage_count"] == 2
    assert details["component_stages"] == "pretrain,finetune"
    assert details["anchor_stage"] == "pretrain"
    assert details["stage.pretrain.rows"] == 16
    assert details["stage.finetune.rows"] == 12
    assert len(details["composition_digest"]) == 64


def test_component_stage_metrics_keep_named_metrics_and_final_train_loss() -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        _record_stage_metrics,
    )

    metrics: dict[str, float] = {}
    declared = {"train_loss", "teacher_loss", "student_loss"}
    _record_stage_metrics(
        metrics,
        {"train_loss": 2.0, "teacher_loss": 2.0},
        declared,
        is_final=False,
    )
    _record_stage_metrics(
        metrics,
        {"train_loss": 1.0, "student_loss": 1.0},
        declared,
        is_final=True,
    )

    assert metrics == {
        "train_loss": 1.0,
        "teacher_loss": 2.0,
        "student_loss": 1.0,
    }


def test_source_state_details_preserve_adapter_declared_scalars() -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        _source_state_details,
    )

    assert _source_state_details(
        {
            "sampling": "full_neighborhood",
            "topology_kind": "relational",
            "sparse_routing": "all_to_all_single_owner_mod",
            "framework_versions": {"torch": "2.5"},
        }
    ) == {
        "sampling": "full_neighborhood",
        "topology_kind": "relational",
        "routing": "all_to_all_single_owner_mod",
        "jagged": True,
    }


def test_torch_ray_config_rejects_removed_resume_options() -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        _torch_ray_config,
    )

    plan = SimpleNamespace(
        algorithm_config={"ray": {"storage_path": "/tmp/ray", "max_failures": 1}}
    )
    assert _torch_ray_config(plan) == ("/tmp/ray", 1)
    plan.algorithm_config["ray"]["resume"] = {"checkpoint_interval": 1}
    with pytest.raises(AlgorithmConfigurationError, match="unsupported key"):
        _torch_ray_config(plan)


def test_runtime_rejects_reserved_torch_policy_features() -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        _validate_runtime_policy,
    )

    supported = SimpleNamespace(
        state_layout="replicated",
        dataset_routing=(SimpleNamespace(mode="split_exact"),),
        checkpoint_owner_rank=0,
        checkpoint_adapter_ref=None,
        evidence_adapter_ref=None,
    )
    _validate_runtime_policy(supported)
    for policy, message in (
        (
            SimpleNamespace(**{**vars(supported), "state_layout": "sharded"}),
            "sharded",
        ),
        (
            SimpleNamespace(**{**vars(supported), "checkpoint_owner_rank": 1}),
            "checkpoint_owner_rank",
        ),
        (
            SimpleNamespace(
                **{
                    **vars(supported),
                    "dataset_routing": (SimpleNamespace(mode="split_framework"),),
                }
            ),
            "split_framework",
        ),
        (
            SimpleNamespace(
                **{
                    **vars(supported),
                    "checkpoint_adapter_ref": "example:checkpoint",
                }
            ),
            "adapter references",
        ),
    ):
        with pytest.raises(AlgorithmConfigurationError, match=message):
            _validate_runtime_policy(policy)


def test_replicated_role_evidence_uses_per_rank_rows() -> None:
    from tributo.algorithms.api import TorchRoleExecutionEvidence

    evidence = TorchRoleExecutionEvidence(
        role="nodes",
        mode="replicate",
        required=True,
        present=True,
        empty_rank_policy="reject",
        expected_rows=8,
        observed_rows=8,
        rows_per_rank=(8, 8),
        replicated_bytes_per_worker=123,
    )
    assert evidence.rows_per_rank == (8, 8)
    with pytest.raises(AlgorithmConfigurationError):
        TorchRoleExecutionEvidence(
            role="nodes",
            mode="replicate",
            required=True,
            present=True,
            empty_rank_policy="reject",
            expected_rows=8,
            observed_rows=8,
            rows_per_rank=(),
            replicated_bytes_per_worker=123,
        )


def test_stage_route_validation_returns_actual_replicated_bytes() -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        _validate_stage_routes,
    )

    calls: list[str] = []

    class MaterializedDataset:
        def count(self) -> int:
            calls.append("count")
            return 8

        def size_bytes(self) -> int:
            calls.append("size_bytes")
            return 123

    materialized = MaterializedDataset()

    class LimitedDataset:
        def materialize(self) -> MaterializedDataset:
            calls.append("materialize")
            return materialized

    class Dataset:
        def count(self) -> int:
            raise AssertionError("the unbounded Dataset must not be counted")

        def limit(self, count: int) -> LimitedDataset:
            calls.append(f"limit:{count}")
            assert count == 9
            return LimitedDataset()

    route = TorchDatasetRoute(
        "nodes", "replicate", max_rows=8, max_bytes_per_worker=256
    )
    datasets: dict[str, object] = {"nodes": Dataset()}
    rows, replicated_bytes = _validate_stage_routes(
        SimpleNamespace(dataset_routing=(route,), max_replicated_bytes_per_worker=256),
        SimpleNamespace(input_roles=("nodes",)),
        datasets,
        2,
    )
    assert rows == {"nodes": 8}
    assert replicated_bytes == {"nodes": 123}
    assert datasets["nodes"] is materialized
    assert calls == ["limit:9", "materialize", "count", "size_bytes"]


def test_adapter_worker_config_cannot_carry_core_paths() -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        _validate_adapter_worker_config,
    )

    with pytest.raises(AlgorithmConfigurationError, match="Core-owned path"):
        _validate_adapter_worker_config({"ray": {"storage_path": "s3://secret"}})


def test_torch_adapter_context_contains_bindings_but_not_core_control_config() -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        _torch_algorithm_context_config,
        _torch_input_bindings,
        _torch_output_config,
    )

    binding = InputBinding(
        name="train",
        resolver_id="example.resolver",
        reference="memory://train",
        feature_names=("feature",),
        label_name="label",
    )
    plan = SimpleNamespace(
        algorithm_config={
            "model": {"width": 4},
            "ray": {"storage_path": "/core/path", "resume": {"uri": "s3://x"}},
            "output": {"bundle_uri": "/core/bundle"},
        },
        input_bindings=SimpleNamespace(bindings=(binding,)),
    )
    assert _torch_algorithm_context_config(plan) == {"model": {"width": 4}}
    assert _torch_input_bindings(plan)["train"]["feature_names"] == ["feature"]
    assert _torch_output_config(plan) == {"bundle_uri": "/core/bundle"}


def test_composite_backward_records_reducer_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    import tributo.integrations.algorithm_runtimes.ray_train_torch as runtime

    class Reducer:
        api_version = 1
        reducer_id = "example.reducer"
        component_schema_id = "example.schema"
        code_digest = "4" * 64

        def reduce(self, config, global_state, context):
            del config, global_state, context
            return TorchGlobalLossReduction(
                "accepted",
                coefficients={"loss": 1.0},
                metrics={"train_loss": TorchMetricContribution(3.0, 1.0)},
            )

    reducer = Reducer()
    monkeypatch.setattr(runtime, "_load_reference", lambda reference: reducer)
    monkeypatch.setattr(
        runtime, "_validate_module_digest", lambda reference, digest: None
    )
    loss = TorchCompositeLossContribution(
        "example.schema",
        {"loss": torch.tensor(2.0, requires_grad=True)},
        {"count": 1.0},
    )
    metric_totals: dict[str, list[float]] = {}
    value = runtime._composite_backward(
        loss,
        config={
            "_core_global_loss_reducer_ref": "example.reducer:Reducer",
            "_core_composite_loss_schema_id": "example.schema",
            "_core_global_loss_reducer_api_version": 1,
            "_core_global_loss_reducer_code_digest": reducer.code_digest,
            "_core_policy_digest": "1" * 64,
            "_core_execution_plan_digest": "2" * 64,
        },
        world_size=1,
        device=torch.device("cpu"),
        dist=torch.distributed,
        metric_totals=metric_totals,
    )
    assert float(value.detach().item()) == 2.0
    assert metric_totals == {"train_loss": [3.0, 1.0]}


def test_component_export_result_contains_composition_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import tributo.exporting.service as service_module
    import tributo.integrations.algorithm_runtimes.ray_train_torch as runtime
    import tributo.integrations.sources.ray_torch as source_module
    from tributo.exporting.models import ExportSource

    class FakeProvider:
        def open_source(self, result, options):
            del result, options

            @contextmanager
            def opened():
                yield ExportSource(
                    source_kind="torch_module",
                    metadata={
                        "artifact_plan": {
                            "targets": ({"name": "model", "format": "onnx"},),
                            "roles": {"inference": "model"},
                        }
                    },
                )

            return opened()

    class FakeService:
        def export_bundle(self, source, config, *, tributo_version):
            del source, config, tributo_version
            return SimpleNamespace(
                bundle_id="bundle-1",
                canonical_uri=str(tmp_path / "bundle"),
                execution_id="execution-1",
                manifest_sha256="5" * 64,
            )

    monkeypatch.setattr(source_module, "RayTorchSourceProvider", FakeProvider)
    monkeypatch.setattr(service_module, "BundleExportService", FakeService)
    policy = SimpleNamespace(
        loop_owner="core_recipe",
        digest="1" * 64,
        state_layout="component",
        execution_plan=SimpleNamespace(
            digest="2" * 64,
            stages=(SimpleNamespace(input_roles=("train",)),),
        ),
    )
    binding = InputBinding(
        name="train",
        resolver_id="example.resolver",
        reference="memory://train",
        feature_names=("feature",),
        label_name="label",
    )
    plan = SimpleNamespace(
        plan_id="2" * 64,
        distribution_spec=SimpleNamespace(
            strategy=DistributionStrategy.RAY_TRAIN_TORCH,
            result_policy=ResultPolicy.BUNDLE_REQUIRED,
            policy=policy,
        ),
        algorithm_config={"output": {"bundle_uri": str(tmp_path / "bundle")}},
        implementation=SimpleNamespace(
            implementation_ref="example:Implementation",
            code_digest="3" * 64,
            implementation_id="example.implementation",
        ),
        input_bindings=SimpleNamespace(bindings=(binding,)),
    )
    result = SimpleNamespace(
        checkpoint=object(),
        metrics={"torch_evidence": {"composition_digest": "a" * 64}},
        core_evidence_attested=True,
    )
    execution = runtime.export_ray_train_torch_result(
        result=result,
        plan=plan,
        run_id="aabbccdd",
    )
    assert execution.outputs["composition_digest"] == "a" * 64


def test_checkpoint_report_builds_descriptor_from_payload(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = TorchStageRunIdentity(
        "aabbccdd",
        "11223344",
        "train",
        1,
        "example",
        "example.recipe",
        "0" * 64,
        "1" * 64,
        "2" * 64,
    )
    runtime = TorchRuntimeContext(
        algorithm_config={},
        implementation_id=identity.implementation_id,
        world_size=1,
        policy_digest=identity.policy_digest,
        execution_plan_digest=identity.execution_plan_digest,
        run_identity=identity,
        input_binding_digest="3" * 64,
    )
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    payload = tmp_path / "model.pt"
    payload.write_bytes(b"model")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "ray.train.get_context",
        lambda: SimpleNamespace(get_world_rank=lambda: 0),
    )
    monkeypatch.setattr(
        "ray.train.report",
        lambda metrics, checkpoint=None: captured.update(metrics),
    )

    report_torch_checkpoint(
        {"train_loss": 0.5},
        TorchCheckpointPayloadDraft(tmp_path),
        stage,
        1,
    )
    descriptor = TorchCheckpointDescriptor.from_dict(captured["checkpoint_descriptor"])
    assert descriptor.identity == identity
    assert descriptor.payload_files == {
        "model.pt": __import__("hashlib").sha256(b"model").hexdigest()
    }
    assert (tmp_path / "torch_checkpoint_descriptor.json").is_file()


def test_checkpoint_payload_rejects_symlinked_descriptor(tmp_path) -> None:
    identity = TorchStageRunIdentity(
        "aabbccdd",
        "11223344",
        "train",
        1,
        "example",
        "example.recipe",
        "0" * 64,
        "1" * 64,
        "2" * 64,
    )
    runtime = TorchRuntimeContext(
        algorithm_config={},
        implementation_id=identity.implementation_id,
        world_size=1,
        policy_digest=identity.policy_digest,
        execution_plan_digest=identity.execution_plan_digest,
        run_identity=identity,
        input_binding_digest="3" * 64,
    )
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    (tmp_path / "model.pt").write_bytes(b"model")
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    (tmp_path / "torch_checkpoint_descriptor.json").symlink_to(target)

    with pytest.raises(AlgorithmExecutionError):
        report_torch_checkpoint({}, TorchCheckpointPayloadDraft(tmp_path), stage, 1)


def test_checkpoint_report_rejects_core_metadata_fields(tmp_path) -> None:
    identity = TorchStageRunIdentity(
        "aabbccdd",
        "11223344",
        "train",
        1,
        "example",
        "example.recipe",
        "0" * 64,
        "1" * 64,
        "2" * 64,
    )
    runtime = TorchRuntimeContext(
        algorithm_config={},
        implementation_id=identity.implementation_id,
        world_size=1,
        policy_digest=identity.policy_digest,
        execution_plan_digest=identity.execution_plan_digest,
        run_identity=identity,
        input_binding_digest="3" * 64,
    )
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    (tmp_path / "model.pt").write_bytes(b"model")

    with pytest.raises(AlgorithmConfigurationError):
        report_torch_checkpoint(
            {"checkpoint_locator": "s3://bucket/private"},
            TorchCheckpointPayloadDraft(tmp_path),
            stage,
            1,
        )


def test_checkpoint_descriptor_omits_unsupported_resume_state() -> None:
    identity = TorchStageRunIdentity(
        "aabbccdd",
        "11223344",
        "train",
        1,
        "example",
        "example.recipe",
        "0" * 64,
        "1" * 64,
        "2" * 64,
    )
    descriptor = TorchCheckpointDescriptor(
        schema_version=1,
        identity=identity,
        run_config_name=identity.run_config_name,
        state_layout="component",
        world_size=1,
        completed_step=7,
        policy_digest=identity.policy_digest,
        execution_plan_digest=identity.execution_plan_digest,
        input_binding_digest="3" * 64,
        implementation_code_digest=identity.implementation_code_digest,
        payload_files={"model.pt": "4" * 64},
        resume_supported=False,
    )
    assert "same_world_size_resume" not in descriptor.to_dict()
    assert TorchCheckpointDescriptor.from_dict(descriptor.to_dict()) == descriptor


def test_runtime_context_omits_unsupported_resume_state() -> None:
    runtime = TorchRuntimeContext(
        algorithm_config={},
        implementation_id="example.adapter",
        world_size=1,
        policy_digest="1" * 64,
        execution_plan_digest="2" * 64,
        resume_supported=False,
    )
    payload = runtime.to_dict()
    assert "same_world_size_resume" not in payload
    restored = TorchStageContext.from_dict(
        {
            "runtime": payload,
            "stage_id": "train",
            "stage_index": 0,
            "is_final": True,
            "input_roles": ["train"],
        }
    )
    assert not hasattr(restored.runtime, "same_world_size_resume")


def test_worker_stage_context_omits_driver_output_paths() -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        _worker_stage_context,
    )

    stage = TorchStageContext(
        TorchRuntimeContext(
            algorithm_config={},
            implementation_id="example.adapter",
            world_size=1,
            policy_digest="1" * 64,
            execution_plan_digest="2" * 64,
            output_config={"bundle_uri": "/driver/model"},
        ),
        "train",
        0,
        True,
        ("train",),
    )

    worker_context = _worker_stage_context(stage)

    assert stage.runtime.output_config == {"bundle_uri": "/driver/model"}
    assert worker_context.runtime.output_config == {}


def test_composite_zero_global_normalizer_fails_before_reducer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tributo.integrations.algorithm_runtimes.ray_train_torch as runtime

    torch = pytest.importorskip("torch")
    dist = pytest.importorskip("torch.distributed")
    invoked = False

    class Reducer:
        api_version = 1
        reducer_id = "example.reducer"
        component_schema_id = "example.components"
        code_digest = "0" * 64

        def reduce(self, config, global_state, context):
            nonlocal invoked
            del config, global_state, context
            invoked = True
            return TorchGlobalLossReduction(
                "accepted",
                coefficients={"loss": 1.0},
                metrics={"train_loss": TorchMetricContribution(1.0, 1.0)},
            )

    monkeypatch.setattr(runtime, "_validate_module_digest", lambda *args: None)
    monkeypatch.setattr(runtime, "_load_reference", lambda reference: Reducer)
    loss = TorchCompositeLossContribution(
        "example.components",
        {"loss": torch.tensor(1.0, requires_grad=True)},
        {"rows": 0.0},
    )
    with pytest.raises(AlgorithmExecutionError, match="normalizer"):
        runtime._reduce_composite_loss(
            loss,
            config={
                "_core_global_loss_reducer_ref": "example:Reducer",
                "_core_global_loss_reducer_api_version": 1,
                "_core_global_loss_reducer_code_digest": "0" * 64,
                "_core_composite_loss_schema_id": "example.components",
                "_core_policy_digest": "1" * 64,
                "_core_execution_plan_digest": "2" * 64,
            },
            world_size=1,
            device=torch.device("cpu"),
            dist=dist,
        )
    assert not invoked


def test_helper_signatures_are_public_and_stable() -> None:
    from tributo.algorithms.api.torch_runtime import report_torch_checkpoint

    assert list(inspect.signature(report_torch_checkpoint).parameters) == [
        "metrics",
        "payload_draft",
        "stage_context",
        "completed_step",
    ]


def test_removed_torch_public_surfaces_are_not_exported() -> None:
    import tributo.algorithms as algorithms

    assert not hasattr(algorithms, "TorchTrainingRecipe")
    assert not hasattr(algorithms, "TrainingRecipeV2")
    assert not hasattr(algorithms.AlgorithmBuilder, "from_torch_recipe")
    assert not hasattr(algorithms.AlgorithmBuilder, "from_training_recipe_v2")
    for name in (
        "TorchCheckpointLocator",
        "TorchCheckpointProgress",
        "TorchPreflightLease",
        "TorchPreflightTokenData",
        "TorchRankProgressStatistics",
        "TorchRecoveryEnvelope",
        "TorchRuntimeExecutionEnvelope",
        "TorchWorkerControlEnvelope",
        "claim_torch_run_directory",
        "describe_torch_checkpoint",
        "torch_run_config_name",
        "validate_torch_retry_identity",
    ):
        assert not hasattr(algorithms, name)
