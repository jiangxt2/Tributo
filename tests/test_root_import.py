"""Root-package import and lazy Ray Jobs boundary tests."""

from __future__ import annotations

import subprocess
import sys


def test_root_import_does_not_eagerly_import_ray_job_submission() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, tributo; "
                "assert 'ray.job_submission' not in sys.modules; "
                "assert 'TributoClient' in dir(tributo)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_lazy_root_client_preserves_public_identity() -> None:
    import tributo
    from tributo.job import RayJob, TributoClient

    assert tributo.TributoClient is TributoClient
    assert tributo.RayJob is RayJob
