"""Real process/Ray workload used only by the owned reporting IT runner."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any


def emit(event: str, count: int = 1) -> dict[str, Any]:
    from tributo._common.reporting.session import active_session

    logger = logging.getLogger("tributo.acceptance")
    logger.setLevel(logging.INFO)
    for _ in range(count):
        logger.info(
            "private SQL/password must not reach Doris",
            extra={"event": event, "status": "succeeded"},
        )
    session = active_session()
    if session:
        assert session.state == "ready"
        assert session._sender is not None and session._sender.flush(5.0)
    return {
        "pid": os.getpid(),
        "count": count,
        "role": session.role if session else "disabled",
        "logging_level": logging.getLogger("tributo").level,
        "snapshot": session.snapshot() if session else {},
        "owner_credentials": [
            key for key in os.environ if key.startswith("REPORTING_IT_SECRET_")
        ],
    }


def original_hook() -> None:
    os.environ["REPORTING_IT_HOOK_RAN"] = "1"


def task(event: str, count: int = 1) -> dict[str, Any]:
    import ray

    result = emit(event, count)
    result["node_id"] = ray.get_runtime_context().get_node_id()
    result["original_hook"] = os.environ.get("REPORTING_IT_HOOK_RAN")
    return result


class Actor:
    def report(self, event: str) -> dict[str, Any]:
        return task(event)


def train_loop(config: dict[str, Any]) -> None:
    from ray import train

    from tributo._common.reporting.session import reporting_worker

    def work(values: Any) -> None:
        emit("tributo.it.train")
        train.report({"rank": train.get_context().get_world_rank(), "ok": 1})

    reporting_worker(work)(config)


def tune_trial(config: dict[str, Any]) -> None:
    from ray import tune

    from tributo._common.reporting.session import producer_scope

    with producer_scope(config["policy"], ray_worker=True):
        emit("tributo.it.tune")
        tune.report({"ok": 1})


def core_reporting_fragment(context: Any) -> None:
    from tests.support.portable_algorithms import custom_training_fragment
    from tributo._common.reporting.session import active_session

    result = emit("tributo.fix3.core.worker")
    session = active_session()
    assert (
        session
        and session.role == "producer"
        and session._collector is None
        and session.config is None
    )
    assert not result["owner_credentials"]
    assert os.environ.get("REPORTING_IT_HOOK_RAN") == "1"
    custom_training_fragment(context)


def core_reporting_reducer(plan: Any, results: Any) -> Any:
    from tests.support.portable_algorithms import reduce_custom_training_results
    from tributo._common.reporting.session import active_session

    result = emit("tributo.fix3.core.reducer")
    session = active_session()
    assert session and session.role == "producer" and session._collector is None
    assert not result["owner_credentials"]
    assert os.environ.get("REPORTING_IT_HOOK_RAN") == "1"
    return reduce_custom_training_results(plan, results)


class PolicyScopeActor:
    def __init__(self, first: str, second: str) -> None:
        from tests.reporting.conftest import reporting_kernel
        from tributo._common.reporting.session import bootstrap_producer

        self.initial = bootstrap_producer(first, ray_worker=True)
        assert self.initial and self.initial._sender
        self.sender = self.initial._sender

        def record(name):
            emit("tributo.fix3.kernel." + name)

        self.a, self.batch = reporting_kernel(first, record)
        self.b, _ = reporting_kernel(second, record)

    def run(self) -> list[list[int]]:
        import asyncio

        from tests.reporting.conftest import reporting_serving_request
        from tributo._common.reporting.sender import CONTEXT
        from tributo._common.reporting.session import active_session

        outputs = []
        for kernel in (self.b, self.a, self.b):
            outputs.append(kernel(self.batch)["prediction"].tolist())
            assert CONTEXT.get() is None
        self.a.close()
        self.b.close()

        async def serving():
            for kind in ("http", "grpc", "batch", "stream"):
                response = await reporting_serving_request(
                    kind,
                    "serving-" + kind,
                    lambda: emit("tributo.fix3.serving.request"),
                )
                if kind == "stream":
                    await anext(response)
                    await response.aclose()
                assert CONTEXT.get() is None

        logging.getLogger("tributo.serving.grpc_deployment").setLevel(logging.INFO)
        asyncio.run(serving())
        assert active_session()._sender is self.sender
        assert self.sender.flush(5.0)
        return outputs


def main(mode: str, path: str) -> None:
    import ray

    from tributo.observability import ReportingConfig, reporting_session

    payload = json.loads(Path(path).read_text())
    os.environ.update(payload["credentials"])
    config = ReportingConfig.model_validate(payload["config"])
    if mode in {"matrix", "light"}:
        from tests.integrations.test_log_reporting import _run_ray_matrix

        _run_ray_matrix(config)
        return
    if mode == "regressions":
        from tests.integrations.test_log_reporting import _run_review_regressions

        _run_review_regressions(config)
        return
    with reporting_session(config):
        result = emit("tributo.it." + mode + ".driver")
        from tributo._common.reporting.session import active_session, ray_options

        session = active_session()
        assert session is not None
        collector = session._collector
        assert collector and collector.process
        pid = collector.process.pid
        ray.init(address="auto", **ray_options(driver=True))
        try:
            result["worker"] = ray.get(
                ray.remote(task).remote("tributo.it." + mode + ".worker")
            )
            # Submitter is allowed to exit while this owner remains alive.
            if mode == "job":
                Path("/workspace/tributo-work/job-ready").write_text(str(pid))
                time.sleep(2)
        finally:
            ray.shutdown()
    assert not Path(f"/proc/{pid}").exists()
    Path("/workspace/tributo-work/artifacts/" + mode + ".json").write_text(
        json.dumps(result)
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
