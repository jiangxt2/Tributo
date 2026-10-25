"""Owned Docker Ray Jobs evidence for application logging and physical workers."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ray

from tributo._common.logging import bootstrap_logging, configure_logging
from tributo._common.logging_events import emit_event
from tributo._common.logging_policy import (
    POLICY_ENV,
    LoggingPolicy,
    driver_logging_runtime_env,
    logging_runtime_env,
)

logger = logging.getLogger("tributo.test.logging_gate")


def user_setup() -> None:
    """A real user hook, executed after the application hook in each process."""
    assert logging.getLogger("tributo").level == logging.DEBUG
    os.environ["TRIBUTO_TEST_HOOK_COUNT"] = str(
        int(os.environ.get("TRIBUTO_TEST_HOOK_COUNT", "0")) + 1
    )


def serve_user_setup() -> None:
    assert logging.getLogger("tributo").level == logging.DEBUG
    logging.getLogger("tributo").setLevel(logging.ERROR)
    os.environ["TRIBUTO_TEST_SERVE_HOOK_COUNT"] = str(
        int(os.environ.get("TRIBUTO_TEST_SERVE_HOOK_COUNT", "0")) + 1
    )


def _worker_evidence() -> dict[str, str]:
    assert os.environ["TRIBUTO_TEST_HOOK_COUNT"] == "1"
    for name in ("tributo.test.logging_gate", "ray_doris.test", "ray_clickhouse.test"):
        target = logging.getLogger(name)
        assert target.isEnabledFor(logging.DEBUG)
        emit_event(
            target,
            "tributo.batch.progress",
            level=logging.DEBUG,
            phase="gate",
            status="running",
            batch_index=1,
        )
    return {
        "job_id": ray.get_runtime_context().get_job_id(),
        "worker_id": ray.get_runtime_context().get_worker_id(),
    }


def _job() -> None:
    import numpy as np

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
        emit_event(logger, "tributo.stage.finished", phase="driver", status="succeeded")
        task = ray.remote(num_cpus=1)(_worker_evidence)
        evidence = ray.get(task.remote())
        assert evidence["job_id"] == ray.get_runtime_context().get_job_id()

        class Kernel:
            def predict(self, inputs):
                return {"score": np.asarray(inputs["x"])[:, 0]}

            def close(self):
                pass

        class Factory:
            factory_id = "gate.identity"

            def create(self):
                _worker_evidence()
                return Kernel()

        dataset = ray.data.from_items([{"feature": float(i)} for i in range(8)])
        result = dataset.map_batches(
            KernelBatchPredictor,
            batch_format="numpy",
            batch_size=2,
            compute=ray.data.ActorPoolStrategy(size=1),
            num_cpus=1,
            fn_constructor_args=(
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
            ),
            fn_constructor_kwargs={"_logging_policy": policy.serialize()},
        ).take_all()
        assert sorted(row["score"] for row in result) == [float(i) for i in range(8)]
        _serve_replicas()
    finally:
        ray.shutdown()


def _torch_failures(policy: LoggingPolicy) -> None:
    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchConfig, TorchTrainer

    def worker(config):
        import ray.train

        from tributo.integrations.algorithm_runtimes.ray_train_torch import (
            ray_torch_adapter_train_loop_per_worker,
            torch_recipe_train_loop_per_worker,
        )

        rank = ray.train.get_context().get_world_rank()
        if rank == config["fail_rank"]:
            wrapper = (
                torch_recipe_train_loop_per_worker
                if config["kind"] == "recipe"
                else ray_torch_adapter_train_loop_per_worker
            )
            wrapper({"_core_logging_policy": config["policy"]})
        else:
            time.sleep(2)
            ray.train.report({"ready": 1})

    for kind, rank in (("recipe", 0), ("recipe", 1), ("adapter", 0), ("adapter", 1)):
        trainer = TorchTrainer(
            train_loop_per_worker=worker,
            train_loop_config={
                "fail_rank": rank,
                "kind": kind,
                "policy": policy.serialize(),
            },
            scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
            torch_config=TorchConfig(backend="gloo"),
            run_config=RunConfig(
                name=f"logging-{kind}-rank-{rank}",
                storage_path=os.environ["TRIBUTO_LOGGING_TEST_ROOT"],
                failure_config=FailureConfig(max_failures=0),
            ),
        )
        try:
            trainer.fit()
        except Exception:
            pass  # The assertion is on the rank-specific structured failure below.
        else:
            raise AssertionError(
                "The injected worker configuration failure did not fail training"
            )


def _serve_replicas() -> None:
    import requests
    from ray import serve

    from tributo._common.serve_utils import deploy_serve_app
    from tributo.serving.observability import (
        InferenceContext,
        inference_metrics_snapshot,
        log_inference_audit,
    )

    class Echo:
        def __init__(self):
            bootstrap_logging()
            assert logging.getLogger("tributo").level == logging.ERROR
            assert os.environ["TRIBUTO_TEST_SERVE_HOOK_COUNT"] == "1"
            assert os.environ["KEEP"] == "kept"
            assert os.environ["EXTRA"] == "extra"

        async def __call__(self, request):
            context = InferenceContext.from_http(request)
            log_inference_audit(
                logging.getLogger("tributo.test.serving"),
                context,
                bundle_id="gate-bundle",
                model_version="1",
                status="ok",
                duration_ms=1,
            )
            return {
                "level": logging.getLogger("tributo").level,
                "requests": inference_metrics_snapshot()["inference_requests_total"],
            }

    decorated = serve.deployment(
        ray_actor_options={
            "num_cpus": 0.1,
            "runtime_env": {"env_vars": {"KEEP": "kept"}},
        }
    )(Echo)
    hook = "tests.integrations.test_debug_logging.serve_user_setup"
    decorated = decorated.options(
        ray_actor_options={
            **decorated.ray_actor_options,
            "runtime_env": {
                **decorated.ray_actor_options["runtime_env"],
                "worker_process_setup_hook": hook,
            },
        }
    )
    try:
        for suffix, deployment in (("existing", decorated), ("plain", Echo)):
            app = "logging-" + suffix
            deploy_serve_app(
                deployment,
                app_name=app,
                route_prefix="/" + app,
                runtime_env={
                    "env_vars": {"KEEP": "kept", "EXTRA": "extra"},
                    "worker_process_setup_hook": hook,
                },
            )
            try:
                response = requests.get(
                    "http://127.0.0.1:8000/" + app,
                    timeout=30,
                    headers={"x-request-id": "logging-" + uuid.uuid4().hex},
                )
                response.raise_for_status()
                assert response.json() == {"level": logging.ERROR, "requests": 1}
            finally:
                serve.delete(app)
        assert decorated.ray_actor_options["num_cpus"] == 0.1
        assert POLICY_ENV not in decorated.ray_actor_options["runtime_env"]["env_vars"]
    finally:
        serve.shutdown()


def _local_child() -> None:
    policy = bootstrap_logging()
    from tributo.inference.batch_job import main as batch_main

    def probe(_env):
        assert (
            ray.get_runtime_context().get_node_id()
            == os.environ["TRIBUTO_TEST_PARENT_NODE"]
        )
        assert (
            ray.get_runtime_context().get_job_id()
            != os.environ["TRIBUTO_TEST_PARENT_JOB"]
        )

        @ray.remote
        def check_main():
            assert logging.getLogger("tributo").level == logging.DEBUG
            return ray.get_runtime_context().get_job_id()

        identity = ray.get_runtime_context().get_job_id()
        assert ray.get(check_main.remote()) == identity
        return SimpleNamespace(status="succeeded")

    try:
        with patch("tributo.inference.batch_job._run_resolved_plan", side_effect=probe):
            assert (
                batch_main(argparse.Namespace(config=None, resolved_plan_env="PROBE"))
                == 0
            )
        main_job = ray.get_runtime_context().get_job_id()
    finally:
        ray.shutdown()
    ray.init(
        address="auto",
        runtime_env=logging_runtime_env(
            {"worker_process_setup_hook": user_setup}, policy=policy
        ),
    )
    try:
        assert (
            ray.get_runtime_context().get_node_id()
            == os.environ["TRIBUTO_TEST_PARENT_NODE"]
        )

        @ray.remote
        def check():
            assert logging.getLogger("tributo").level == logging.DEBUG
            assert os.environ["TRIBUTO_TEST_HOOK_COUNT"] == "1"
            return ray.get_runtime_context().get_job_id()

        child_id = ray.get_runtime_context().get_job_id()
        assert ray.get(check.remote()) == child_id
        assert len({child_id, main_job, os.environ["TRIBUTO_TEST_PARENT_JOB"]}) == 3
        print(
            "TRIBUTO_LOCAL_RESULT="
            + json.dumps(
                {
                    "parent": os.environ["TRIBUTO_TEST_PARENT_JOB"],
                    "child": child_id,
                    "main_job": main_job,
                    "address": os.environ["RAY_ADDRESS"],
                    "node_id": os.environ["TRIBUTO_TEST_PARENT_NODE"],
                }
            ),
            flush=True,
        )
    finally:
        ray.shutdown()


def _local_chain() -> None:
    from tributo.runtime_providers import run_local_entrypoint

    configure_logging(level="DEBUG", log_format="json")

    original_init = ray.init
    original_run = subprocess.run
    results = []

    def track_init(**kwargs):
        result = original_init(**kwargs)
        os.environ["TRIBUTO_TEST_PARENT_JOB"] = ray.get_runtime_context().get_job_id()
        os.environ["TRIBUTO_TEST_PARENT_NODE"] = ray.get_runtime_context().get_node_id()
        return result

    def capture(*args, **kwargs):
        command = args[0] if args else kwargs.get("args", ())
        if "--local-child" not in command:
            return original_run(*args, **kwargs)
        result = original_run(*args, **kwargs, capture_output=True, text=True)
        results.append(result)
        print(result.stdout, flush=True)
        print(result.stderr, file=sys.stderr, flush=True)
        return result

    with (
        patch("ray.init", side_effect=track_init),
        patch("tributo.runtime_providers.subprocess.run", side_effect=capture),
    ):
        run_local_entrypoint(
            f"{sys.executable} -m tests.integrations.test_debug_logging --local-child",
            num_cpus=2,
            timeout=180,
        )
    assert len(results) == 1 and results[0].returncode == 0
    assert "TRIBUTO_LOCAL_RESULT=" in results[0].stdout
    assert not ray.is_initialized()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="store_true")
    parser.add_argument("--local-child", action="store_true")
    parser.add_argument("--channel", choices=("data", "torch"), default="data")
    args = parser.parse_args()
    if args.job:
        if args.channel == "data":
            _job()
        else:
            policy = bootstrap_logging()
            ray.init(address="auto", **driver_logging_runtime_env())
            try:
                _torch_failures(policy)
            finally:
                ray.shutdown()
        return
    if args.local_child:
        _local_child()
        return
    if args.channel == "torch":
        if os.environ.get("TRIBUTO_LOGGING_TORCH_LOCAL") != "1":
            raise RuntimeError("Use scripts/run_debug_logging_torch_it.sh")
        if ray.is_initialized():
            raise RuntimeError("Refusing to reuse a caller-owned runtime")
        with tempfile.TemporaryDirectory(prefix="tributo-logging-torch-") as directory:
            context = ray.init(
                address="local",
                num_cpus=4,
                include_dashboard=True,
                dashboard_host="127.0.0.1",
                dashboard_port=0,
            )
            try:
                _run_control(
                    args.channel, "http://" + context.dashboard_url, Path(directory)
                )
            finally:
                ray.shutdown()
        assert not ray.is_initialized()
        return
    if os.environ.get("TRIBUTO_DOCKER_RAY_TEST") != "1":
        raise RuntimeError("Use scripts/run_debug_logging_it.sh")
    _run_control(args.channel, "http://127.0.0.1:8265", Path("/workspace/tributo-work"))


def _run_control(channel: str, address: str, root: Path) -> None:
    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient(address)
    policy = LoggingPolicy(
        level="DEBUG",
        batches=True,
        steps=True,
        logger_levels=(("ray_doris", "DEBUG"), ("ray_clickhouse", "DEBUG")),
    )
    runtime = logging_runtime_env(
        {
            "worker_process_setup_hook": "tests.integrations.test_debug_logging.user_setup",
            "env_vars": {"TRIBUTO_LOGGING_TEST_ROOT": str(root)},
            **(
                {"working_dir": str(Path(__file__).resolve().parents[2])}
                if channel == "torch"
                else {}
            ),
        },
        policy=policy,
        jobs=True,
    )
    identity = client.submit_job(
        entrypoint=shlex.join(
            [
                sys.executable,
                "-m",
                "tests.integrations.test_debug_logging",
                "--job",
                "--channel",
                channel,
            ]
        ),
        runtime_env=runtime,
        submission_id="tributo-logging-" + uuid.uuid4().hex,
    )
    deadline = time.monotonic() + 480
    while str(client.get_job_status(identity)) not in {
        "SUCCEEDED",
        "FAILED",
        "STOPPED",
    }:
        if time.monotonic() > deadline:
            client.stop_job(identity)
            raise TimeoutError("Logging Ray Job exceeded its owned deadline")
        time.sleep(1)
    logs = client.get_job_logs(identity)
    print(logs, flush=True)
    (root / "logging-job.log").write_text(logs)
    assert str(client.get_job_status(identity)) == "SUCCEEDED"
    records = []
    for line in logs.splitlines():
        start = line.find('{"timestamp"')
        if start < 0:
            continue
        try:
            records.append(json.loads(line[start:]))
        except ValueError:
            continue
    failures = [
        r
        for r in records
        if r.get("phase") == "training_worker" and r.get("status") == "failed"
    ]
    if channel == "torch":
        assert {(r["mode"], r["rank"]) for r in failures} == {
            ("recipe", 0),
            ("recipe", 1),
            ("adapter", 0),
            ("adapter", 1),
        }, failures
        assert all(r["world_size"] == 2 for r in failures)
    else:
        assert any(r.get("event") == "tributo.model.ready" for r in records)
        assert {r["logger"].split(".")[0] for r in records} >= {
            "tributo",
            "ray_doris",
            "ray_clickhouse",
        }
        _local_chain()
    print("TRIBUTO_LOGGING_GATE_PASSED:" + channel, flush=True)


if __name__ == "__main__":
    main()
