"""Exercise portable Tune's intermediate reporter with Ray Train and ASHA."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def train_worker(config: dict) -> None:
    import ray
    from ray import train

    rank = train.get_context().get_world_rank()
    actor_id = ray.get_runtime_context().get_actor_id()
    ray.get(config["tracker"].register.remote(config["loss"], rank, actor_id))
    for round_number in range(1, 9):
        if rank == 0:
            ray.get(config["tracker"].record.remote(config["loss"], round_number))
        metrics = {
            "train-logloss": config["loss"],
            "training_rounds": round_number,
            "_tributo_xgboost_recovery": {
                "phase": "final" if round_number == 8 else "training"
            },
        }
        if round_number == 8 and rank == 0:
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                (Path(checkpoint_dir) / "model").write_text("fixture")
                train.report(
                    metrics, checkpoint=train.Checkpoint.from_directory(checkpoint_dir)
                )
        else:
            train.report(metrics)
        time.sleep(0.3)


def trainable(config: dict, tracker: object) -> None:
    from ray import train, tune
    from ray.train.v2.api.data_parallel_trainer import DataParallelTrainer

    import tributo.algorithms
    import tributo.integrations.algorithm_runtimes.framework_native as native
    import tributo.training.portable_tune as portable_tune
    from tests.algorithms.conftest import request_for
    from tributo.algorithms.api import (
        AlgorithmOperation,
        DistributionStrategy,
        ExecutionProfile,
        ExecutionRequest,
        FrameworkNativePolicy,
        ResultPolicy,
    )
    from tributo.algorithms.spi import (
        InputExecutionContext,
        InputResolutionContext,
        PreparedInput,
    )
    from tributo.training.tune_config import TuneSearchConfig

    trial_id = tune.get_context().get_trial_id()
    policy = FrameworkNativePolicy(
        framework="xgboost-rabit", evidence_collector_ref="tests.example:collect"
    )
    plan = SimpleNamespace(
        input_bindings=SimpleNamespace(bindings=(SimpleNamespace(name="train"),)),
        distribution_spec=SimpleNamespace(
            strategy=DistributionStrategy.FRAMEWORK_NATIVE,
            result_policy=ResultPolicy.FIT_ONLY,
            policy=policy,
        ),
        runtime=SimpleNamespace(
            worker_input_adapter_ref=None,
            worker_count=2,
            num_cpus=1.0,
            num_gpus=0.0,
            memory_bytes=None,
            custom_resources={},
        ),
        algorithm_config={},
        environment=SimpleNamespace(python="3.12", dependencies=()),
    )

    class Dataset:
        def count(self) -> int:
            return 8

    class Algorithm:
        def validate_environment(self) -> None:
            pass

        def bind_datasets(self, _views: object) -> dict:
            return {"train": Dataset()}

        def build_trainer(self, _config: object, _datasets: object) -> object:
            return DataParallelTrainer(
                train_worker,
                train_loop_config={**config, "tracker": tracker},
                scaling_config=train.ScalingConfig(num_workers=2),
                run_config=train.RunConfig(
                    name=f"inner-train-{trial_id}",
                    storage_path="/workspace/tributo-work",
                ),
            )

        def collect_evidence(self, _result: object) -> dict:
            return {}

    class Dispatcher:
        def plan(self, *_args: object) -> object:
            return plan

        def execute_plan(self, _plan: object, _context: object) -> object:
            envelope = SimpleNamespace(
                cancelled=False, plan=plan, input_payloads=(None,), run_id="test"
            )
            return native.FrameworkNativeRuntime().execute(envelope)

    runner = object.__new__(portable_tune.PortableTuneRunner)
    runner._request = ExecutionRequest(
        algorithm_request=request_for("external_map_reduce", AlgorithmOperation.FIT),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
    )
    runner._input_context = InputExecutionContext({})
    runner._resolution_context = InputResolutionContext({})
    runner._tune_config = TuneSearchConfig(
        metric="train-logloss", progress_attr="training_rounds"
    )
    with (
        patch.object(tributo.algorithms, "build_algorithm_dispatcher", Dispatcher),
        patch.object(portable_tune, "_fit_only_plan", lambda value: value),
        patch.object(native, "_algorithm", lambda _envelope: Algorithm()),
        patch.object(
            native,
            "_load_reference",
            lambda _reference: lambda _payload: PreparedInput({"train": object()}),
        ),
        patch.object(native, "_validated_framework_evidence", lambda *a, **k: ([], {})),
        patch.object(native, "_actual_environment_versions", lambda *a: {}),
    ):
        runner._trainable()({})


def main() -> None:
    import ray
    from ray import tune
    from ray.tune.execution.placement_groups import PlacementGroupFactory
    from ray.tune.logger import CSVLoggerCallback
    from ray.tune.schedulers import ASHAScheduler
    from ray.util.state import get_actor

    os.chdir("/workspace/tributo-work")
    storage = Path.cwd() / f"portable-tune-{uuid.uuid4().hex}"

    @ray.remote(num_cpus=0)
    class Tracker:
        def __init__(self) -> None:
            self.rounds: dict[str, list[int]] = {}
            self.actors: dict[str, dict[int, str]] = {}

        def register(self, loss: float, rank: int, actor_id: str) -> None:
            self.actors.setdefault(str(loss), {})[rank] = actor_id

        def record(self, loss: float, round_number: int) -> None:
            self.rounds.setdefault(str(loss), []).append(round_number)

        def read(self) -> dict[str, list[int]]:
            return self.rounds

        def actor_ids(self, loss: float) -> dict[int, str]:
            return self.actors[str(loss)]

    tracker = Tracker.remote()
    scheduler = ASHAScheduler(
        time_attr="training_rounds",
        max_t=16,
        grace_period=1,
        reduction_factor=2,
    )
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(trainable, tracker=tracker),
            PlacementGroupFactory([{"CPU": 1}, {"CPU": 1}, {"CPU": 1}]),
        ),
        param_space={"loss": tune.grid_search([0.1, 1.0])},
        tune_config=tune.TuneConfig(
            metric="train-logloss",
            mode="min",
            scheduler=scheduler,
            max_concurrent_trials=1,
        ),
        run_config=tune.RunConfig(
            name="intermediate",
            storage_path=str(storage),
            callbacks=[CSVLoggerCallback()],
        ),
    )
    results = tuner.fit()
    assert results.num_errors == 0
    histories = {}
    for result in results:
        frame = result.metrics_dataframe
        assert frame is not None
        histories[str(result.config["loss"])] = [
            int(value) for value in frame["training_rounds"].dropna().tolist()
        ]
    assert max(histories["0.1"]) == 8
    assert max(histories["1.0"]) < 8
    assert all(values == sorted(values) for values in histories.values())
    actor_ids = ray.get(tracker.actor_ids.remote(1.0))
    assert set(actor_ids) == {0, 1}
    deadline = time.monotonic() + 20
    while True:
        states = [get_actor(actor_id, timeout=3) for actor_id in actor_ids.values()]
        if all(state is not None and state.state == "DEAD" for state in states):
            break
        assert time.monotonic() < deadline, "ASHA-stopped workers remained alive"
        time.sleep(1)
    worker_rounds = ray.get(tracker.read.remote())
    assert max(worker_rounds["1.0"]) < 8
    print(
        "RESULT: "
        + json.dumps(
            {
                "histories": histories,
                "worker_rounds": worker_rounds,
                "nodes": len(ray.nodes()),
            }
        )
    )


if __name__ == "__main__":
    main()
