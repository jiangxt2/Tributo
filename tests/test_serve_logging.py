"""Deployment merging and caller hook precedence without Ray processes."""

import logging
import sys
import types
from unittest.mock import patch

import pytest

from tributo._common import worker_logging
from tributo._common.logging import bootstrap_logging
from tributo._common.logging_policy import (
    POLICY_ENV,
    SETUP_HOOK,
    USER_HOOK_ENV,
    LoggingPolicy,
)
from tributo._common.serve_utils import deploy_serve_app

pytestmark = pytest.mark.usefixtures("isolated_logging")


class _Deployment:
    def __init__(self, options):
        self.ray_actor_options = options

    def options(self, *, ray_actor_options):
        return _Deployment(ray_actor_options)

    def bind(self, **kwargs):
        return self


@pytest.mark.parametrize("decorated", [False, True])
def test_replica_preserves_options_and_user_hook_final_control(monkeypatch, decorated):
    calls = []

    def user_hook():
        calls.append(logging.getLogger("tributo").level)
        logging.getLogger("tributo").setLevel(logging.ERROR)

    module = types.ModuleType("serve_logging_test_hook")
    module.setup = user_hook
    monkeypatch.setitem(sys.modules, module.__name__, module)
    hook_name = module.__name__ + ".setup"

    original = {
        "num_cpus": 0.25,
        "runtime_env": {
            "env_vars": {"KEEP": "kept"},
            "worker_process_setup_hook": hook_name,
        },
    }
    deployment = _Deployment(original) if decorated else type("Service", (), {})
    runtime = {"env_vars": {"TRIBUTO_LOG_LEVEL": "DEBUG", "EXTRA": "extra"}}
    if not decorated:
        runtime["worker_process_setup_hook"] = hook_name
    with (
        patch(
            "tributo._common.serve_utils.active_logging_policy",
            return_value=LoggingPolicy(),
        ),
        patch("tributo._common.serve_utils.ray.is_initialized", return_value=True),
        patch("tributo._common.serve_utils.serve.start"),
        patch(
            "tributo._common.serve_utils.serve.deployment",
            side_effect=lambda **kw: lambda cls: _Deployment(kw["ray_actor_options"]),
        ),
        patch("tributo._common.serve_utils.serve.run") as run,
    ):
        deploy_serve_app(
            deployment, app_name="logging", route_prefix="/logging", runtime_env=runtime
        )
    options = run.call_args.args[0].ray_actor_options
    selected = options["runtime_env"]
    assert LoggingPolicy.deserialize(selected["env_vars"][POLICY_ENV]).level == "DEBUG"
    assert selected["env_vars"]["EXTRA"] == "extra"
    if decorated:
        assert options["num_cpus"] == 0.25
        assert selected["env_vars"]["KEEP"] == "kept"
    assert POLICY_ENV not in runtime["env_vars"]
    assert original["runtime_env"]["worker_process_setup_hook"] == hook_name
    assert POLICY_ENV not in original["runtime_env"]["env_vars"]
    hook = selected["worker_process_setup_hook"]
    assert hook == SETUP_HOOK
    monkeypatch.setenv(POLICY_ENV, selected["env_vars"][POLICY_ENV])
    monkeypatch.setenv(USER_HOOK_ENV, selected["env_vars"][USER_HOOK_ENV])
    worker_logging.setup()
    bootstrap_logging()
    bootstrap_logging()
    assert calls == [logging.DEBUG]
    assert logging.getLogger("tributo").level == logging.ERROR


def test_bad_replica_policy_fails_before_ray_or_serve_start():
    with (
        patch("tributo._common.serve_utils.ray.init") as init,
        patch("tributo._common.serve_utils.serve.start") as start,
        pytest.raises(ValueError, match="Invalid application logging configuration"),
    ):
        deploy_serve_app(
            type("Service", (), {}),
            app_name="logging",
            route_prefix="/logging",
            runtime_env={"env_vars": {"TRIBUTO_LOG_LEVEL": "invalid"}},
        )
    init.assert_not_called()
    start.assert_not_called()


def test_serve_rejects_non_json_callable_hook_before_start():
    with (
        patch("tributo._common.serve_utils.serve.start") as start,
        pytest.raises(ValueError, match="string hook"),
    ):
        deploy_serve_app(
            type("Service", (), {}),
            app_name="logging",
            route_prefix="/logging",
            runtime_env={"worker_process_setup_hook": lambda: None},
        )
    start.assert_not_called()
