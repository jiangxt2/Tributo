"""Fixed-Ray contract tests for exact-coverage Train dataset assignment."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest

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
    monkeypatch.setattr(config, "_is_v2_autoscaler", lambda: False)
    config.set_train_total_resources(num_train_cpus=2, num_train_gpus=0)

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

    assert ray.__version__ == "2.55.1"
    assert "equal=True" in inspect.getsource(DataConfig.configure)
    source = inspect.getsource(ExactCoverageDataConfig.configure)
    assert "equal=False" in source
    assert "ray.data._internal" not in source
