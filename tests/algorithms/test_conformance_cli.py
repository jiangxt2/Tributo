"""Installed-Wheel conformance CLI contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import tributo.algorithms.conformance as conformance
import tributo.algorithms.conformance_cli as conformance_cli
from tributo.algorithms.conformance import AlgorithmPackageConformanceReport


def test_installed_conformance_matches_identity_and_contract_manifest(
    monkeypatch, tmp_path
) -> None:
    manifest = tmp_path / "identities.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entry_points": {
                    "example": {
                        "algorithm_id": "example",
                        "distribution": "tributo-algorithms-example",
                        "implementation_id": "example.implementation",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    descriptor = SimpleNamespace(
        registration=SimpleNamespace(
            contract_bindings=SimpleNamespace(
                config=object(), input=object(), output=object(), coverage=object()
            )
        )
    )

    class Distribution:
        metadata = {"Name": "tributo-algorithms-example"}

    class EntryPoint:
        name = "example"
        dist = Distribution()

        @staticmethod
        def load():
            return descriptor

    monkeypatch.setattr(
        conformance.importlib.metadata,
        "entry_points",
        lambda **kwargs: (EntryPoint(),),
    )
    monkeypatch.setattr(
        conformance,
        "validate_installed_algorithm_package",
        lambda descriptor, entry_point_name: AlgorithmPackageConformanceReport(
            algorithm_id="example",
            implementation_id="example.implementation",
            distribution="tributo-algorithms-example",
            package_version="1.0.0",
            entry_point_name=entry_point_name,
            contract_ids=("config", "input", "output", "coverage"),
        ),
    )

    reports = conformance._run_installed_conformance(
        distribution_prefix="tributo-algorithms-",
        expected_count=1,
        identity_manifest=manifest,
        required_contracts=("config", "input", "output", "coverage"),
        forbidden_imports=(),
    )

    assert len(reports) == 1
    assert reports[0].entry_point_name == "example"


def test_conformance_cli_prints_deterministic_report(monkeypatch, capsys) -> None:
    report = AlgorithmPackageConformanceReport(
        algorithm_id="example",
        implementation_id="example.implementation",
        distribution="tributo-algorithms-example",
        package_version="1.0.0",
        entry_point_name="example",
        contract_ids=("config", "input", "output", "coverage"),
    )
    monkeypatch.setattr(
        conformance_cli,
        "_run_installed_conformance",
        lambda **kwargs: (report,),
    )

    assert (
        conformance_cli.main(
            [
                "--distribution-prefix",
                "tributo-algorithms-",
                "--expected-count",
                "1",
                "--identity-manifest",
                "identities.json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["count"] == 1
