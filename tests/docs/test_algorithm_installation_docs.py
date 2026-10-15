from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_algorithm_installation_documents_the_aggregate_distribution() -> None:
    documentation = (_ROOT / "docs/algorithms/getting-started.md").read_text(
        encoding="utf-8"
    )
    current, historical = documentation.split("## Historical source snapshot", 1)

    assert "tributo-algorithms[boosting]" in current
    assert "tributo-algorithms[causal-xlearner]" in current
    assert "37" in current
    assert "Core 1.1" in current
    assert '--project "${TRIBUTO_ALGORITHMS_ROOT}"' in current
    assert "tributo_algorithms-0.1.0-py3-none-any.whl[boosting]" in current
    assert "--package" not in current
    assert '"${TRIBUTO_ALGORITHMS_ROOT}/packages/boosting"' not in current
    assert "Once the selected aggregate version is published" in current
    assert "71a6a16c3fd3676aba2225a0c528817634927fc7" in historical
