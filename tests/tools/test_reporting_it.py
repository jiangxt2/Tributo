"""Manual reporting CI contracts without starting Docker."""

from pathlib import Path
from unittest.mock import Mock, call

import pytest
import yaml

from tools import reporting_it


@pytest.mark.parametrize("mode", list(reporting_it.it.RuntimeBuildInputMode))
def test_image_source_reaches_runtime_and_doris(tmp_path, monkeypatch, mode):
    binary = tmp_path / "otelcol-contrib"
    binary.write_bytes(b"unit fixture")
    binary.chmod(0o700)
    prepared = object()
    prepare = Mock(return_value=prepared)
    ensure = Mock()
    cleanup = Mock()
    monkeypatch.setattr(reporting_it.it, "_assert_project_absent", Mock())
    monkeypatch.setattr(reporting_it.it, "docker_platform", lambda: "linux/arm64")
    monkeypatch.setattr(reporting_it.it, "prepare_runtime", prepare)
    monkeypatch.setattr(reporting_it.it, "ensure_digest_image", ensure)
    monkeypatch.setattr(reporting_it.it, "_local_digest_reference", lambda ref: ref)
    monkeypatch.setattr(reporting_it, "_image_diagnostics", Mock())
    monkeypatch.setattr(reporting_it, "_cleanup_built_runtime", cleanup)
    monkeypatch.setattr(
        reporting_it,
        "_available_subnet",
        Mock(side_effect=RuntimeError("after images")),
    )
    options = (
        {}
        if mode == reporting_it.it.RuntimeBuildInputMode.LOCAL_MIRROR
        else {"build_input_mode": mode}
    )
    with pytest.raises(RuntimeError, match="after images"):
        reporting_it.run(tmp_path, tmp_path / "evidence", **options)
    assert prepare.call_args.kwargs["build_input_mode"] == mode
    profile = reporting_it.it.load_profile("log-reporting")
    assert ensure.call_args_list == [
        call(
            profile.definition[name],
            use_mirror=mode == reporting_it.it.RuntimeBuildInputMode.LOCAL_MIRROR,
        )
        for name in ("doris_fe_image", "doris_be_image")
    ]
    cleanup.assert_called_once_with(prepared)


def test_workflow_is_manual_only_and_retains_failure_evidence():
    path = Path(__file__).parents[2] / ".github/workflows/log-reporting-e2e.yml"
    workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["reporting"]
    assert job["runs-on"] == "ubuntu-24.04-arm"
    command = next(
        s["run"] for s in job["steps"] if "tools.reporting_it" in s.get("run", "")
    )
    assert "--cases all --build-input-mode canonical" in command
    assert "timeout --signal=TERM --kill-after=60s 30m" in command
    upload = next(
        s
        for s in job["steps"]
        if s.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["if"] == "always()"
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["retention-days"] == "7"
