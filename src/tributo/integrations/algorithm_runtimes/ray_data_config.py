"""Ray 2.55.1 DataConfig compatibility for exact row coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ray.train import DataConfig

from tributo.util.annotations import DeveloperAPI

if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from ray.data import DataIterator, Dataset, NodeIdStr


@DeveloperAPI
class ExactCoverageDataConfig(DataConfig):
    """Keep every input row when Ray Train assigns Dataset iterators."""

    @DeveloperAPI
    def configure(
        self,
        datasets: dict[str, Dataset],
        world_size: int,
        worker_handles: list[ActorHandle] | None,
        worker_node_ids: list[NodeIdStr] | None,
        **kwargs: Any,
    ) -> list[dict[str, DataIterator]]:
        """Mirror Ray 2.55.1 ``DataConfig`` except for ``equal=False``."""
        del worker_handles, kwargs
        from ray.data import ExecutionResources

        output: list[dict[str, DataIterator]] = [{} for _ in range(world_size)]
        for dataset_name, dataset in datasets.items():
            if dataset.name is None:
                dataset.set_name(dataset_name)

        datasets_to_split = (
            set(datasets)
            if self._datasets_to_split == "all"
            else set(self._datasets_to_split)
        )
        locality_hints = worker_node_ids if self._enable_shard_locality else None
        for name, dataset in datasets.items():
            execution_options = self._get_execution_options(name)
            if (
                execution_options.is_resource_limits_default()
                and not self._is_v2_autoscaler()
            ):
                execution_options.exclude_resources = (
                    execution_options.exclude_resources.add(
                        ExecutionResources(
                            cpu=self._num_train_cpus,
                            gpu=self._num_train_gpus,
                        )
                    )
                )

            configured = dataset.copy(dataset)
            configured.context.execution_options = execution_options
            if name in datasets_to_split:
                splits = configured.streaming_split(
                    world_size,
                    equal=False,
                    locality_hints=locality_hints,
                )
                for rank, split in enumerate(splits):
                    output[rank][name] = split
            else:
                for rank in range(world_size):
                    output[rank][name] = configured.iterator()
        return output


__all__ = ["ExactCoverageDataConfig"]
