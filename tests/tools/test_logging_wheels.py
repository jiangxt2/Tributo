"""Target selection and integrity tests without network or Docker."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.prepare_logging_wheels import download_wheel, prepare_wheels, select_wheels


def test_selects_linux_arm_wheel_and_respects_python_markers():
    url = "https://files.pythonhosted.org/packages/example/"
    lock = {
        "package": [
            {
                "name": "example",
                "version": "1.0",
                "wheels": [
                    {
                        "url": url + "example-1.0-cp312-cp312-macosx_12_0_arm64.whl",
                        "hash": "sha256:" + "a" * 64,
                        "size": 1,
                    },
                    {
                        "url": url
                        + "example-1.0-cp312-cp312-manylinux_2_28_aarch64.whl",
                        "hash": "sha256:" + "b" * 64,
                        "size": 2,
                    },
                ],
            },
        ]
    }
    requirements = b'example==1.0 ; sys_platform == "linux" \\\n --hash=sha256:example\nmissing==1 ; python_version >= "3.13"\n'
    result = select_wheels(requirements, lock, "aarch64")
    assert len(result) == 1
    assert result[0]["hash"] == "sha256:" + "b" * 64
    assert "manylinux_2_28_aarch64" in result[0]["filename"]


def test_verified_cached_wheel_does_not_download(tmp_path: Path):
    data = b"verified wheel"
    wheel = {
        "filename": "example.whl",
        "url": "https://files.pythonhosted.org/packages/example.whl",
        "size": len(data),
        "hash": "sha256:" + hashlib.sha256(data).hexdigest(),
    }
    (tmp_path / "example.whl").write_bytes(data)
    with patch("tools.prepare_logging_wheels.subprocess.run") as run:
        assert download_wheel(wheel, tmp_path) == wheel
    run.assert_not_called()


def test_truncated_download_is_rejected_without_replacing_artifact(tmp_path: Path):
    wheel = {
        "filename": "example.whl",
        "url": "https://files.pythonhosted.org/packages/example.whl",
        "size": 100,
        "hash": "sha256:" + "a" * 64,
    }

    def truncated(command, **kwargs):
        Path(command[command.index("-o") + 1]).write_bytes(b"truncated")

    with patch("tools.prepare_logging_wheels.subprocess.run", side_effect=truncated):
        with pytest.raises(ValueError, match="checksum/size mismatch"):
            download_wheel(wheel, tmp_path)
    assert not (tmp_path / "example.whl").exists()


def test_preparation_uses_existing_export_and_records_lock_digest(tmp_path, capsys):
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n")
    directory = tmp_path / "wheels"
    requirements = b"locked export"
    wheel = {"filename": "example.whl"}
    with (
        patch(
            "tools.prepare_logging_wheels.select_wheels", return_value=[wheel]
        ) as select,
        patch("tools.prepare_logging_wheels.download_wheel") as download,
    ):
        prepare_wheels(requirements, lock, "linux/arm64", directory)
    select.assert_called_once_with(requirements, {"version": 1}, "aarch64")
    download.assert_called_once_with(wheel, directory)
    assert directory.is_dir() and not (directory / "manifest.json").exists()
    assert hashlib.sha256(lock.read_bytes()).hexdigest() in capsys.readouterr().out
