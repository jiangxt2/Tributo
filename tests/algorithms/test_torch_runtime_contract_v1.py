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
    TorchCheckpointLocator,
    TorchCheckpointProgress,
    TorchCompositeLossContribution,
    TorchDatasetRoute,
    TorchGlobalLossReduction,
    TorchLossContribution,
    TorchMetricContribution,
    TorchMetricPolicy,
    TorchMetricReductionContext,
    TorchPolicy,
    TorchPreflightLease,
    TorchPreflightTokenData,
    TorchRankProgressStatistics,
    TorchRecoveryEnvelope,
    TorchStageRunIdentity,
    TorchStageSpec,
    apply_torch_loss_backward,
    reduce_torch_metrics,
    report_torch_checkpoint,
    torch_run_config_name,
)
from tributo.algorithms.spi import TorchRuntimeContext, TorchStageContext


class _Scalar:
    ndim = 0

    def __init__(self, value: float) -> None:
        self.value = value

    def detach(self) -> "_Scalar":
        return self

    def item(self) -> float:
        return self.value


def _identity_kwargs() -> dict[str, object]:
    return {
        "run_id": "aabbccdd",
        "invocation_id": "11223344",
        "algorithm": "example",
        "implementation_ref": "example:Recipe",
        "implementation_code_digest": "0" * 64,
        "policy_digest": "1" * 64,
        "execution_plan_digest": "2" * 64,
        "runtime_id": "tributo.ray_train_torch",
        "plan_digest": "3" * 64,
    }


def test_torch_policy_and_run_name_are_deterministic() -> None:
    route = TorchDatasetRoute("train", "split_exact")
    execution_plan = SingleStageTorchPlan(
        stage=TorchStageSpec("train", "example:loop", ("train",))
    )
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
    assert torch_run_config_name(identity) == identity.run_config_name


def test_preflight_lease_is_one_shot_and_identity_bound() -> None:
    lease = TorchPreflightLease(TorchPreflightTokenData(**_identity_kwargs()))
    lease.claim(
        run_id="aabbccdd",
        invocation_id="11223344",
        plan_digest="3" * 64,
        runtime_id="tributo.ray_train_torch",
    )
    lease.consume(
        run_id="aabbccdd",
        invocation_id="11223344",
        plan_digest="3" * 64,
        runtime_id="tributo.ray_train_torch",
    )
    with pytest.raises(AlgorithmExecutionError):
        lease.consume(
            run_id="aabbccdd",
            invocation_id="11223344",
            plan_digest="3" * 64,
            runtime_id="tributo.ray_train_torch",
        )


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


def test_locator_rejects_local_paths_and_policy_replicate_budget_is_explicit() -> None:
    with pytest.raises(AlgorithmConfigurationError):
        TorchCheckpointLocator("/tmp/checkpoint", "0" * 64)
    route = TorchDatasetRoute(
        "nodes", "replicate", max_rows=10, max_bytes_per_worker=10
    )
    plan = SingleStageTorchPlan(
        stage=TorchStageSpec("train", "example:loop", ("nodes",))
    )
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


def test_stage_dependency_is_allowed_when_external_recovery_is_disabled() -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        _control_for_stage,
    )

    control = _control_for_stage(
        SimpleNamespace(
            algorithm_config={},
            runtime=SimpleNamespace(resume_from=None),
        ),
        SimpleNamespace(
            resume_supported=False,
            digest="1" * 64,
            execution_plan=SimpleNamespace(digest="2" * 64),
        ),
        SimpleNamespace(stage_id="student", checkpoint_from_stage="teacher"),
        run_id="aabbccdd",
        invocation_id="11223344",
        predecessor={
            "locator": "s3://bucket/teacher-checkpoint",
            "descriptor_digest": "3" * 64,
        },
    )
    assert control is not None
    assert control["purpose"] == "stage_dependency"
    assert control["source_stage_id"] == "teacher"


def test_role_evidence_falls_back_to_primary_binding_for_alias_roles() -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        _binding_digest_for_role,
    )

    class Descriptors:
        def get(self, role: str) -> object:
            raise AlgorithmConfigurationError(f"unknown resolved input role: {role}")

    primary = SimpleNamespace(binding_digest="3" * 64)
    plan = SimpleNamespace(
        input_descriptors=Descriptors(), primary_input_descriptor=primary
    )
    assert _binding_digest_for_role(plan, "val") == "3" * 64


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
    details = _component_state_details(stages)
    assert details["component_stage_count"] == 2
    assert details["component_stages"] == "pretrain,finetune"
    assert details["anchor_stage"] == "finetune"
    assert details["stage.pretrain.rows"] == 16
    assert details["stage.finetune.rows"] == 12
    assert len(details["composition_digest"]) == 64


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
        )


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


def test_checkpoint_report_builds_descriptor_from_payload(tmp_path) -> None:
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

    class Draft:
        checkpoint_dir = tmp_path

        def report(self, *, metrics, stage_context, completed_step) -> None:
            captured.update(metrics)
            assert stage_context is stage
            assert completed_step == 1

    report_torch_checkpoint(
        {"train_loss": 0.5},
        Draft(),
        stage,
        1,
    )
    descriptor = TorchCheckpointDescriptor.from_dict(captured["checkpoint_descriptor"])
    assert descriptor.identity == identity
    assert descriptor.payload_files == {
        "model.pt": __import__("hashlib").sha256(b"model").hexdigest()
    }
    assert (tmp_path / "torch_checkpoint_descriptor.json").is_file()


def test_recovery_envelope_roundtrip_and_locator_digest_binding() -> None:
    locator = TorchCheckpointLocator("s3://bucket/stage", "4" * 64)
    envelope = TorchRecoveryEnvelope(
        completed_stage_ids=("pretrain",),
        stage_checkpoints={"pretrain": locator},
        active_stage_id="finetune",
        active_checkpoint=TorchCheckpointLocator("s3://bucket/active", "5" * 64),
    )
    restored = TorchRecoveryEnvelope.from_dict(envelope.to_dict())
    assert restored == envelope
    with pytest.raises(AlgorithmConfigurationError):
        TorchRecoveryEnvelope(
            completed_stage_ids=("pretrain",),
            stage_checkpoints={
                "pretrain": TorchCheckpointLocator("s3://bucket/stage", "4" * 64)
            },
            active_stage_id="pretrain",
            active_checkpoint=TorchCheckpointLocator("s3://bucket/active", "5" * 64),
        )
    with pytest.raises(AlgorithmConfigurationError):
        TorchRecoveryEnvelope.from_dict(
            {
                "completed_stage_ids": ["pretrain"],
                "stage_checkpoints": {"pretrain": "not-a-locator"},
            }
        )


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

    class Draft:
        checkpoint_dir = tmp_path

        def report(self, *, metrics, stage_context, completed_step) -> None:
            del metrics, stage_context, completed_step

    with pytest.raises(AlgorithmExecutionError):
        report_torch_checkpoint({}, Draft(), stage, 1)


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

    class Draft:
        checkpoint_dir = tmp_path

        def report(self, *, metrics, stage_context, completed_step) -> None:
            del metrics, stage_context, completed_step

    with pytest.raises(AlgorithmConfigurationError):
        report_torch_checkpoint(
            {"checkpoint_locator": "s3://bucket/private"}, Draft(), stage, 1
        )


def test_checkpoint_progress_roundtrip_and_conditional_resume_serialization() -> None:
    progress = TorchCheckpointProgress(
        epoch=2,
        micro_batch_cursor=3,
        optimizer_step=7,
        scheduler_step=2,
        accumulation_steps=4,
        dataset_cursor_by_rank={"0": 3, "1": 3},
        shuffle_seed=44,
    )
    assert TorchCheckpointProgress.from_dict(progress.to_dict()) == progress
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
        same_world_size_resume=None,
    )
    assert "same_world_size_resume" not in descriptor.to_dict()
    assert TorchCheckpointDescriptor.from_dict(descriptor.to_dict()) == descriptor


def test_rank_progress_statistics_and_runtime_context_are_typed() -> None:
    statistics = TorchRankProgressStatistics(
        rows_processed=4,
        coverage_totals={"coverage.positive": 2},
        loss_numerator_total=3.0,
        loss_normalizer_total=4.0,
        metric_totals={"accuracy": (2.0, 4.0)},
        reducer_observation={"branch": "nnpu_normal"},
    )
    assert TorchRankProgressStatistics.from_dict(statistics.to_dict()) == statistics
    with pytest.raises(AlgorithmConfigurationError):
        TorchRankProgressStatistics.from_dict({"rows_processed": "four"})

    runtime = TorchRuntimeContext(
        algorithm_config={},
        implementation_id="example.adapter",
        world_size=1,
        policy_digest="1" * 64,
        execution_plan_digest="2" * 64,
        resume_supported=False,
        same_world_size_resume=None,
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
    assert restored.runtime.same_world_size_resume is None


def test_scheduler_boundary_and_recovery_commit_are_fail_closed(tmp_path) -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        _require_checkpoint_commit,
        _should_apply_epoch_scheduler,
    )

    assert _should_apply_epoch_scheduler(
        restore_same_stage=True,
        epoch=1,
        restored_epoch=1,
        restored_epoch_scheduler_applied=False,
    )
    assert not _should_apply_epoch_scheduler(
        restore_same_stage=True,
        epoch=1,
        restored_epoch=1,
        restored_epoch_scheduler_applied=True,
    )
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
    (tmp_path / "model.pt").write_bytes(b"model")
    descriptor = TorchCheckpointDescriptor(
        schema_version=1,
        identity=identity,
        run_config_name=identity.run_config_name,
        state_layout="replicated",
        world_size=1,
        completed_step=1,
        policy_digest=identity.policy_digest,
        execution_plan_digest=identity.execution_plan_digest,
        input_binding_digest="3" * 64,
        implementation_code_digest=identity.implementation_code_digest,
        payload_files={"model.pt": "4" * 64},
    )
    with pytest.raises(AlgorithmExecutionError, match="commit"):
        _require_checkpoint_commit(tmp_path, descriptor)


def test_local_stage_staging_ignores_prior_partial_attempt(tmp_path) -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        _persist_stage_checkpoint,
    )

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
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.pt").write_bytes(b"model")
    run_root = tmp_path / identity.run_config_name
    run_root.mkdir()

    class Checkpoint:
        @contextmanager
        def as_directory(self):
            yield source

    # A previous attempt with the same digest must not block a fresh staging
    # attempt; only the committed destination is authoritative.
    stale_path = run_root / f".stage_checkpoint.staging-{'4' * 64}-old"
    stale_path.mkdir()
    locator = _persist_stage_checkpoint(
        Checkpoint(),
        identity=identity,
        storage_path=tmp_path,
        descriptor_digest="4" * 64,
    )
    assert locator == f"ray://{run_root / 'stage_checkpoint'}"
    assert (run_root / "stage_checkpoint" / "torch_stage_commit.json").is_file()


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
