"""Application logging contracts, without Ray processes or external services."""

from __future__ import annotations

import io
import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tributo._common.logging import (
    JsonFormatter,
    TextFormatter,
    apply_logging_policy,
    bootstrap_logging,
    configure_logging,
    emit_event,
)
from tributo._common.logging_policy import POLICY_ENV, LoggingPolicy, resolve_policy

pytestmark = pytest.mark.usefixtures("isolated_logging")


def _without_external_output():
    # Pytest 9 attaches handlers to non-propagating loggers at the call phase.
    # Establish the no-caller-handler scenario only after that phase starts.
    logger = logging.getLogger("tributo")
    logger.handlers = []
    logger.propagate = False


def test_owned_handler_is_idempotent_and_module_debug_reaches_output(monkeypatch):
    _without_external_output()
    output = io.StringIO()
    monkeypatch.setenv("TRIBUTO_LOGGER_LEVELS", '{"tributo.test":"DEBUG"}')
    configure_logging(stream=output)
    configure_logging(stream=output)
    emit_event(
        logging.getLogger("tributo.test.child"),
        "tributo.test",
        level=logging.DEBUG,
        phase="prediction",
        status="running",
        rank=1,
    )
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(logging.getLogger("tributo").handlers) == 1
    assert records[-1]["rank"] == 1
    assert logging.getLogger("tributo").level == logging.INFO
    assert logging.getLogger("tributo").handlers[0].level == logging.DEBUG


def test_external_handler_format_threshold_filter_and_propagation_survive():
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("caller:%(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger("tributo")
    logger.handlers = [handler]
    logger.propagate = False
    configure_logging(level="DEBUG", log_format="json")
    configure_logging(level="DEBUG", log_format="json")
    assert logger.handlers == [handler]
    assert handler.formatter is formatter
    assert handler.level == logging.INFO
    assert not logger.propagate
    assert output.getvalue().count("DEBUG is filtered") == 1
    logger.debug("hidden")
    assert "hidden" not in output.getvalue()


@pytest.mark.parametrize("name", ["tributo.test", "ray_doris.reader"])
@pytest.mark.parametrize("existing", [False, True])
def test_child_handler_owns_output_without_disabling_sibling_fallback(
    monkeypatch, name, existing
):
    caller, fallback = io.StringIO(), io.StringIO()
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(caller)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("caller:%(message)s")
    handler.setFormatter(formatter)
    handler.addFilter(lambda record: record.getMessage() != "filtered")
    monkeypatch.setenv("TRIBUTO_LOGGER_LEVELS", json.dumps({name: "DEBUG"}))
    with patch.object(logging.getLogger(), "handlers", []):
        if existing:
            logger.addHandler(handler)
        configure_logging(stream=fallback)
        if not existing:
            logger.addHandler(handler)
        configure_logging(stream=fallback)
        configure_logging(stream=fallback)
        logger.debug("hidden")
        logger.info("visible")
        logger.info("filtered")
        logging.getLogger(name.split(".")[0] + ".sibling").info("sibling")
    assert [
        json.loads(line)["message"] for line in fallback.getvalue().splitlines()
    ] == ["sibling"]
    assert caller.getvalue().splitlines() == [
        "caller:Application DEBUG is filtered by existing handler thresholds",
        "caller:visible",
    ]
    assert logger.handlers == [handler] and logger.propagate
    assert handler.level == logging.INFO and handler.formatter is formatter


def test_explicit_parameters_override_environment_and_frozen_policy(monkeypatch):
    _without_external_output()
    monkeypatch.setenv(
        POLICY_ENV, LoggingPolicy(level="ERROR", log_format="text").serialize()
    )
    monkeypatch.setenv("TRIBUTO_LOG_LEVEL", "CRITICAL")
    output = io.StringIO()
    configure_logging(level="DEBUG", log_format="json", stream=output)
    emit_event(
        logging.getLogger("tributo.test"),
        "tributo.test",
        level=logging.DEBUG,
        phase="test",
        status="ok",
    )
    assert json.loads(output.getvalue().splitlines()[-1])["level"] == "DEBUG"


def test_matching_handler_name_does_not_grant_ownership():
    handler = logging.StreamHandler(io.StringIO())
    handler.name = "tributo.console"
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter("caller:%(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger("tributo")
    logger.handlers = [handler]
    logger.propagate = False
    configure_logging(level="INFO")
    assert logger.handlers == [handler]
    assert handler.level == logging.WARNING
    assert handler.formatter is formatter


def test_reconfiguration_removes_previously_owned_module_debug(monkeypatch):
    _without_external_output()
    monkeypatch.setenv("TRIBUTO_LOGGER_LEVELS", '{"tributo.test":"DEBUG"}')
    configure_logging(stream=io.StringIO())
    assert logging.getLogger("tributo.test").isEnabledFor(logging.DEBUG)
    monkeypatch.setenv("TRIBUTO_LOGGER_LEVELS", "{}")
    configure_logging(level="WARNING")
    assert not logging.getLogger("tributo.test").isEnabledFor(logging.DEBUG)


@pytest.mark.parametrize(
    "env",
    [
        {"TRIBUTO_LOG_LEVEL": "password=do-not-echo"},
        {"TRIBUTO_LOG_FORMAT": "yaml"},
        {"TRIBUTO_LOGGER_LEVELS": '{"root":"DEBUG"}'},
        {"TRIBUTO_LOGGER_LEVELS": '{"torch.distributed":"DEBUG"}'},
        {"TRIBUTO_LOGGER_LEVELS": '{"handler":{"class":"bad"}}'},
        {"TRIBUTO_LOGGER_LEVELS": "x" * 4097},
        {POLICY_ENV: '{"version":true}'},
    ],
)
def test_invalid_policy_is_bounded_and_never_echoes_input(env):
    with pytest.raises(
        ValueError, match="Invalid application logging configuration"
    ) as error:
        resolve_policy(env)
    assert "do-not-echo" not in str(error.value)


def test_json_and_text_share_safe_fields_without_stringifying_objects():
    class Dangerous:
        def __str__(self):
            raise AssertionError("arbitrary conversion")

    record = logging.LogRecord(
        "tributo.test", logging.ERROR, __file__, 1, "fixed", (), None
    )
    record.__dict__.update(
        event="tributo.stage.finished",
        phase="fit",
        status="failed",
        operation_id="a" * 32,
        rank=1,
        duration_ms=1.25,
        password="private",
        config=Dangerous(),
        error_type="RuntimeError",
        request_id="bad\nforged",
        model_id="x" * 129,
    )
    payload = json.loads(JsonFormatter().format(record))
    text = TextFormatter().format(record)
    assert payload["rank"] == 1
    assert payload["fields_truncated"] is True
    assert "password" not in payload and "config" not in payload
    assert "request_id" not in payload and "model_id" not in payload
    assert "private" not in text and "forged" not in text
    assert (
        text.index("event=")
        < text.index("phase=")
        < text.index("status=")
        < text.index("rank=")
    )


def test_explicit_sdk_configuration_survives_owned_bootstrap():
    _without_external_output()
    configure_logging(level="DEBUG", log_format="text", stream=io.StringIO())
    policy = bootstrap_logging()
    assert policy.level == "DEBUG" and policy.log_format == "text"
    assert logging.getLogger("tributo").level == logging.DEBUG


def test_explicit_arguments_override_invalid_lower_priority_leaf_options(monkeypatch):
    _without_external_output()
    monkeypatch.setenv("TRIBUTO_LOG_LEVEL", "invalid-lower-priority")
    monkeypatch.setenv("TRIBUTO_LOG_FORMAT", "invalid-lower-priority")
    output = io.StringIO()
    configure_logging(level="DEBUG", log_format="json", stream=output)
    assert logging.getLogger("tributo").level == logging.DEBUG
    assert isinstance(logging.getLogger("tributo").handlers[0].formatter, JsonFormatter)


def test_wrapper_reapplication_preserves_explicit_user_hook_configuration():
    _without_external_output()
    original = LoggingPolicy(level="INFO")
    apply_logging_policy(original)
    configure_logging(level="ERROR", log_format="text", stream=io.StringIO())
    apply_logging_policy(original)
    assert logging.getLogger("tributo").level == logging.ERROR


def test_runtime_metadata_failure_does_not_drop_error_event(monkeypatch):
    import ray

    _without_external_output()
    output = io.StringIO()
    configure_logging(stream=output)
    monkeypatch.setattr(ray, "is_initialized", lambda: True)

    def unavailable():
        raise RuntimeError("unavailable")

    monkeypatch.setattr(ray, "get_runtime_context", unavailable)
    emit_event(
        logging.getLogger("tributo.test"),
        "tributo.stage.finished",
        level=logging.ERROR,
        phase="fit",
        status="failed",
        error_type="ValueError",
    )
    record = json.loads(output.getvalue())
    assert record["status"] == "failed" and record["error_type"] == "ValueError"
    assert "ray_job_id" not in record


def test_text_and_json_keep_the_same_safe_exception_type():
    error = ValueError("password=private")
    record = logging.LogRecord(
        "tributo.test",
        logging.ERROR,
        __file__,
        1,
        "operation failed",
        (),
        (type(error), error, None),
    )
    payload = json.loads(JsonFormatter().format(record))
    text = TextFormatter().format(record)
    assert payload["error_type"] == "ValueError"
    assert "error_type=ValueError" in text
    assert "private" not in text and "private" not in json.dumps(payload)


def test_local_entrypoint_preserves_explicit_sdk_policy_and_pins_owned_address(
    monkeypatch,
):
    from tributo.runtime_providers import run_local_entrypoint

    _without_external_output()
    configure_logging(level="DEBUG", log_format="text", stream=io.StringIO())
    runtime = SimpleNamespace(address_info={"gcs_address": "127.0.0.1:23456"})
    with (
        patch("ray.is_initialized", return_value=False),
        patch("ray.init", return_value=runtime),
        patch("ray.shutdown") as shutdown,
        patch("tributo.runtime_providers.subprocess.run") as run,
    ):
        run.return_value.returncode = 0
        run_local_entrypoint("python child.py", env_vars={"RAY_ADDRESS": "wrong:1234"})
    env = run.call_args.kwargs["env"]
    selected = LoggingPolicy.deserialize(env[POLICY_ENV])
    assert selected.level == "DEBUG" and selected.log_format == "text"
    assert logging.getLogger("tributo").level == logging.DEBUG
    assert env["RAY_ADDRESS"] == "127.0.0.1:23456"
    shutdown.assert_called_once()


def test_owned_runtime_without_address_is_closed_before_child_launch():
    from tributo.exceptions import JobConfigurationError
    from tributo.runtime_providers import run_local_entrypoint

    with (
        patch("ray.is_initialized", return_value=False),
        patch("ray.init", return_value=SimpleNamespace()),
        patch("ray.shutdown") as shutdown,
        patch("tributo.runtime_providers.subprocess.run") as run,
        pytest.raises(JobConfigurationError, match="did not provide an address"),
    ):
        run_local_entrypoint("python child.py")
    shutdown.assert_called_once()
    run.assert_not_called()


def test_broken_handler_cannot_replace_business_failure():
    class Broken(logging.Handler):
        def emit(self, record):
            raise OSError("sink failed")

    logger = logging.getLogger("tributo.test")
    logger.handlers = [Broken()]
    logger.setLevel(logging.ERROR)
    original = ValueError("business failure")
    with pytest.raises(ValueError) as caught:
        try:
            raise original
        except ValueError:
            emit_event(
                logger,
                "tributo.test.failed",
                level=logging.ERROR,
                error_type="ValueError",
            )
            raise
    assert caught.value is original
