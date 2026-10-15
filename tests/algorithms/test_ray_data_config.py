"""Fixed-Ray contract tests for exact-coverage Train dataset assignment."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from ray.data import ExecutionOptions, ExecutionResources
from ray.data.context import DataContext

from tributo.integrations.algorithm_runtimes.ray_data_config import (
    ExactCoverageDataConfig,
)


class _Dataset:
    def __init__(self) -> None:
        self.name = None
        self.context = SimpleNamespace(execution_options=None)
        self.split_calls: list[dict[str, Any]] = []

    def set_name(self, name: str) -> None:
        self.name = name

    def copy(self, other: object) -> _Dataset:
        assert other is self
        return self

    def streaming_split(self, count: int, **kwargs: Any) -> list[str]:
        self.split_calls.append({"count": count, **kwargs})
        return [f"iterator-{rank}" for rank in range(count)]

    def iterator(self) -> str:
        return "replicated"


def test_exact_coverage_uses_public_unequal_streaming_split_and_locality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _Dataset()
    config = ExactCoverageDataConfig()
    options = ExecutionOptions(
        exclude_resources=ExecutionResources(cpu=2), preserve_order=True
    )
    monkeypatch.setattr(DataContext.get_current(), "execution_options", options)

    result = config.configure(
        {"train": dataset},
        world_size=2,
        worker_handles=None,
        worker_node_ids=["node-a", "node-b"],
    )

    assert result == [
        {"train": "iterator-0"},
        {"train": "iterator-1"},
    ]
    assert dataset.name == "train"
    assert dataset.split_calls == [
        {
            "count": 2,
            "equal": False,
            "locality_hints": ["node-a", "node-b"],
        }
    ]
    assert dataset.context.execution_options.exclude_resources.cpu == 2
    assert dataset.context.execution_options.preserve_order is True
    assert dataset.context.execution_options is not options


def test_exact_coverage_can_replicate_declared_small_datasets() -> None:
    dataset = _Dataset()
    config = ExactCoverageDataConfig(datasets_to_split=[])

    result = config.configure(
        {"vocabulary": dataset},
        world_size=2,
        worker_handles=None,
        worker_node_ids=["node-a", "node-b"],
    )

    assert result == [
        {"vocabulary": "replicated"},
        {"vocabulary": "replicated"},
    ]
    assert dataset.split_calls == []


def test_exact_coverage_compatibility_is_locked_to_audited_ray() -> None:
    import ray
    from ray.train import DataConfig

    assert ray.__version__ == "2.59.0"
    assert "equal=True" in inspect.getsource(DataConfig.configure)
    source = inspect.getsource(ExactCoverageDataConfig.configure)
    assert "equal=False" in source
    assert "ray.data._internal" not in source


@pytest.mark.parametrize("per_dataset", [False, True])
def test_exact_coverage_copies_user_options_without_aliasing(
    per_dataset: bool,
) -> None:
    options = ExecutionOptions(
        resource_limits=ExecutionResources(cpu=3),
        exclude_resources=ExecutionResources(cpu=1),
        preserve_order=True,
    )
    config = ExactCoverageDataConfig(
        execution_options={"train": options} if per_dataset else options,
        enable_shard_locality=False,
    )
    train, validation = _Dataset(), _Dataset()

    config.configure(
        {"train": train, "validation": validation},
        world_size=2,
        worker_handles=None,
        worker_node_ids=["node-a", "node-b"],
    )

    configured = train.context.execution_options
    assert configured.resource_limits.cpu == 3
    assert configured.exclude_resources.cpu == 1
    assert configured.preserve_order is True
    assert train.split_calls[0]["locality_hints"] is None
    assert configured is not options
    assert configured is not validation.context.execution_options
    configured.exclude_resources = ExecutionResources(cpu=4)
    assert options.exclude_resources.cpu == 1
    if not per_dataset:
        assert validation.context.execution_options.exclude_resources.cpu == 1


def test_exact_coverage_uses_current_context_for_unspecified_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExactCoverageDataConfig(
        execution_options={"train": ExecutionOptions(preserve_order=False)}
    )
    options = ExecutionOptions(
        preserve_order=True, exclude_resources=ExecutionResources(cpu=2)
    )
    monkeypatch.setattr(DataContext.get_current(), "execution_options", options)
    train, validation = _Dataset(), _Dataset()

    config.configure(
        {"train": train, "validation": validation},
        world_size=1,
        worker_handles=None,
        worker_node_ids=None,
    )

    assert train.context.execution_options.preserve_order is False
    assert validation.context.execution_options.preserve_order is True
    assert validation.context.execution_options.exclude_resources.cpu == 2
