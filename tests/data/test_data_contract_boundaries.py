"""Static checks for neutral ingestion/writing contract boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

WORKTREE_ROOT = Path(__file__).parents[2]
SOURCE_ROOT = WORKTREE_ROOT / "src" / "tributo"
WRITING_ROOT = SOURCE_ROOT / "data" / "writing"

# runtime_credentials.py intentionally remains outside WRITING_ROOT as the
# sole neutral bridge for approved engine-native credential objects.  It must
# not become a route to legacy connector execution or storage-format writes.


def _production_python_files() -> tuple[Path, ...]:
    return tuple(sorted(WRITING_ROOT.glob("*.py")))


def test_writing_core_does_not_depend_on_ingestion_or_legacy_base() -> None:
    forbidden = (
        "from tributo.data.ingestion import",
        "import tributo.data.ingestion",
        "from tributo.data.base import",
        "import tributo.data.base",
    )
    violations = [
        f"{path.name}: {marker}"
        for path in _production_python_files()
        if path.name != "compatibility.py"
        for marker in forbidden
        if marker in path.read_text()
    ]

    assert violations == []


def test_production_writers_do_not_reimplement_data_plane_commit() -> None:
    """Keep file and snapshot commits owned by Ray or Daft native writers."""
    forbidden = (
        "LanceDataset.commit(",
        "lance.fragment.write_fragments(",
        "table.append(arrow_table",
        "table.overwrite(arrow_table",
        "_write_lance_distributed(",
    )
    violations = [
        f"{path.relative_to(WORKTREE_ROOT)}: {marker}"
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
        for marker in forbidden
        if marker in path.read_text()
    ]

    assert violations == []


def _data_handle_to_arrow_calls(path: Path) -> tuple[str, ...]:
    """Find data-handle materialization calls without banning schema conversion."""
    tree = ast.parse(path.read_text(), filename=str(path))
    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "to_arrow":
            continue
        receiver = ast.unparse(node.func.value)
        receiver_name = receiver.rsplit(".", maxsplit=1)[-1].lower()
        if receiver_name not in {"schema", "arrow_schema", "field", "field_type"}:
            calls.append(receiver)
    return tuple(calls)


def test_production_writers_do_not_materialize_data_handles_to_arrow() -> None:
    """Allow schema-only conversion while blocking full data materialization."""
    violations = [
        f"{path.relative_to(WORKTREE_ROOT)}: {receiver}.to_arrow()"
        for path in _production_python_files()
        if path.name != "compatibility.py"
        for receiver in _data_handle_to_arrow_calls(path)
    ]

    assert violations == []
