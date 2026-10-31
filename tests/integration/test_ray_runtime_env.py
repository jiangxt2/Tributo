"""Explicit Ray uv runtime-environment propagation tests."""

from __future__ import annotations

import json
import logging
import os
import shlex
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.ray_runtime_env


def test_uv_run_runtime_environment_is_detected() -> None:
    """The feature remains tested separately from local Ray data contracts."""
    from ray._private import ray_constants
    from ray._private.worker import _maybe_modify_runtime_env

    assert ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV is True
    modified = _maybe_modify_runtime_env({}, _skip_env_hook=False)

    assert modified.get("py_executable", "").startswith("uv run")


def _user_setup() -> None:
    assert logging.getLogger("tributo").level == logging.DEBUG
    os.environ["TRIBUTO_TEST_HOOK_COUNT"] = str(
        int(os.environ.get("TRIBUTO_TEST_HOOK_COUNT", "0")) + 1
    )


def _worker_probe() -> None:
    from tributo._common.logging import emit_event

    assert os.environ["TRIBUTO_TEST_HOOK_COUNT"] == "1"
    for name in ("tributo.test.runtime_env", "ray_doris.test"):
        logger = logging.getLogger(name)
        assert logger.isEnabledFor(logging.DEBUG)
        emit_event(
            logger,
            "tributo.test.worker",
            level=logging.DEBUG,
            phase="test",
            status="running",
        )


def _logging_job() -> None:
    import numpy as np
    import ray
    import ray.train
    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchConfig, TorchTrainer

    from tributo._common.logging import bootstrap_logging
    from tributo._common.logging_policy import driver_logging_runtime_env
    from tributo.inference.contracts import (
        InputBindingSpec,
        OutputBindingSpec,
        TensorInputBinding,
        TensorOutputBinding,
    )
    from tributo.inference.kernel import KernelBatchPredictor

    policy = bootstrap_logging()
    ray.init(address="auto", **driver_logging_runtime_env())
    try:
        ray.get(ray.remote(_worker_probe).remote())

        class Kernel:
            def predict(self, inputs):
                return {"score": inputs["x"][:, 0]}

            def close(self):
                pass

        class Factory:
            factory_id = "logging.identity"

            def create(self):
                _worker_probe()
                return Kernel()

        @ray.remote
        class Actor(KernelBatchPredictor):
            def check(self):
                return self({"feature": np.array([1.0])})["score"].tolist()

        actor = Actor.remote(
            Factory(),
            InputBindingSpec(
                tensors=(TensorInputBinding(tensor_name="x", columns=("feature",)),)
            ),
            OutputBindingSpec(
                tensors=(
                    TensorOutputBinding(
                        tensor_name="score", column="score", semantic="score"
                    ),
                )
            ),
            _logging_policy=policy.serialize(),
        )
        assert ray.get(actor.check.remote()) == [1.0]
        ray.kill(actor)

        def worker(config):
            from tributo.integrations.algorithm_runtimes import (
                ray_train_torch as runtime,
            )

            if ray.train.get_context().get_world_rank() == config["rank"]:
                wrapper = (
                    runtime.torch_recipe_train_loop_per_worker
                    if config["mode"] == "recipe"
                    else runtime.ray_torch_adapter_train_loop_per_worker
                )
                wrapper({"_core_logging_policy": config["policy"]})
            else:
                time.sleep(2)
                ray.train.report({"ready": 1})

        for mode in ("recipe", "adapter"):
            for rank in (0, 1):
                trainer = TorchTrainer(
                    train_loop_per_worker=worker,
                    train_loop_config={
                        "mode": mode,
                        "rank": rank,
                        "policy": policy.serialize(),
                    },
                    scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
                    torch_config=TorchConfig(backend="gloo"),
                    run_config=RunConfig(
                        name=f"logging-{mode}-{rank}",
                        storage_path=os.environ["TRIBUTO_TEST_ROOT"],
                        failure_config=FailureConfig(max_failures=0),
                    ),
                )
                try:
                    trainer.fit()
                except Exception:
                    pass  # Assert on the four real rank-specific records below.
                else:
                    raise AssertionError("Injected worker error did not fail training")
        print("TRIBUTO_LOGGING_PROBE_PASSED", flush=True)
    finally:
        ray.shutdown()


@pytest.mark.timeout(480)
def test_logging_reaches_owned_job_task_actor_and_all_training_ranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    import ray
    from ray._private import ray_constants
    from ray.job_submission import JobSubmissionClient

    from tributo._common.logging_policy import LoggingPolicy, logging_runtime_env

    if ray.is_initialized():
        raise RuntimeError("This test must own its local runtime")
    previous = ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV
    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    # Dashboard/Jobs processes read the flag independently on import.
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    try:
        context = ray.init(
            address="local",
            num_cpus=4,
            include_dashboard=True,
            dashboard_host="127.0.0.1",
            dashboard_port=0,
        )
        # Ray 2.59 leaves the reaper stdin open on macOS. Retain it until
        # pytest teardown so close happens after ray.shutdown stops the reaper.
        for info in ray._private.worker._global_node.all_processes.get(
            ray_constants.PROCESS_TYPE_REAPER, ()
        ):
            if info.process.stdin is not None:
                request.addfinalizer(info.process.stdin.close)
        client = JobSubmissionClient("http://" + context.dashboard_url)
        runtime = logging_runtime_env(
            {
                "worker_process_setup_hook": "tests.integration.test_ray_runtime_env._user_setup",
                "env_vars": {
                    "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                    "TRIBUTO_TEST_ROOT": str(tmp_path),
                    "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
                },
            },
            policy=LoggingPolicy(
                level="DEBUG", logger_levels=(("ray_doris", "DEBUG"),)
            ),
            jobs=True,
        )
        job = client.submit_job(
            entrypoint=shlex.join(
                [sys.executable, "-m", "tests.integration.test_ray_runtime_env"]
            ),
            runtime_env=runtime,
        )
        deadline = time.monotonic() + 480
        while str(client.get_job_status(job)) not in {"SUCCEEDED", "FAILED", "STOPPED"}:
            if time.monotonic() > deadline:
                client.stop_job(job)
                raise TimeoutError("Owned logging probe timed out")
            time.sleep(0.5)
        output = client.get_job_logs(job)
        (tmp_path / "logging-job.log").write_text(output)
        assert str(client.get_job_status(job)) == "SUCCEEDED", output
        records = []
        for line in output.splitlines():
            start = line.find('{"timestamp"')
            if start >= 0:
                try:
                    records.append(json.loads(line[start:]))
                except ValueError:
                    continue
        failures = [
            r
            for r in records
            if r.get("event") == "tributo.training.finished"
            and r.get("status") == "failed"
        ]
        assert {(r["mode"], r["rank"]) for r in failures} == {
            (mode, rank) for mode in ("recipe", "adapter") for rank in (0, 1)
        }, output
        assert all(
            r["world_size"] == 2 and r.get("ray_job_id") and r.get("worker_id")
            for r in failures
        )
        assert any(
            r.get("event") == "tributo.model.ready" and r.get("actor_id")
            for r in records
        ), output
        assert any(
            r.get("event") == "tributo.test.worker" and r.get("task_id")
            for r in records
        ), output
        assert "TRIBUTO_LOGGING_PROBE_PASSED" in output
    finally:
        ray.shutdown()
        ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = previous
    assert not ray.is_initialized()


if __name__ == "__main__":
    _logging_job()
