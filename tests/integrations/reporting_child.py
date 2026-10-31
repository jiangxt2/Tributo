"""Child entrypoint for the real local runtime ownership test."""

import json
from pathlib import Path

import ray

from tests.integrations.reporting_workload import emit, task
from tributo._common.logging import bootstrap_logging
from tributo._common.reporting.session import ray_options
from tributo.observability import reporting_session

if __name__ == "__main__":
    bootstrap_logging()
    with reporting_session() as session:
        result = emit("tributo.it.local.child")
        ray.init(address="auto", **ray_options(driver=True))
        try:
            result["worker"] = ray.get(
                ray.remote(task).remote("tributo.it.local.worker")
            )
        finally:
            ray.shutdown()
    Path("/workspace/tributo-work/artifacts/local.json").write_text(json.dumps(result))
