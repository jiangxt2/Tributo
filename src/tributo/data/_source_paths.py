"""Shared local-path resolution for canonical file sources."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlsplit

from tributo._common import find_project_root
from tributo.data.source_config import (
    CanonicalSourceInput,
    CsvSourceConfig,
    ParquetSourceConfig,
    ProviderSourceConfig,
)
from tributo.util.annotations import DeveloperAPI

_LOCAL_FILE_PROVIDERS = frozenset({"tributo.parquet", "parquet", "tributo.csv", "csv"})


@DeveloperAPI
def resolve_file_source_path(
    source: CanonicalSourceInput,
    project_root_path: Path | None,
) -> CanonicalSourceInput:
    """Resolve relative built-in file sources against one explicit base path.

    The returned source carries the resolved path into Provider normalization,
    so the canonical source identity and the engine-native plan cannot diverge.
    URI sources and absolute local paths are returned unchanged.
    """
    field_name: str | None = None
    raw_path: str | None = None
    if isinstance(source, (ParquetSourceConfig, CsvSourceConfig)):
        field_name = "path"
        raw_path = source.path
    elif (
        isinstance(source, ProviderSourceConfig)
        and source.provider in _LOCAL_FILE_PROVIDERS
    ):
        field_name = "uri"
        raw_path = source.uri

    if field_name is None or raw_path is None or urlsplit(raw_path).scheme:
        return source
    path = Path(raw_path)
    if path.is_absolute():
        return source
    # Preserve the compatibility entrypoint's exact project-root semantics;
    # in particular, do not collapse a caller-supplied symlinked root.
    root = project_root_path or find_project_root()
    return source.model_copy(update={field_name: str(root / path)})


@DeveloperAPI
def require_local_file_source_exists(source: CanonicalSourceInput) -> None:
    """Preserve the compatibility loader's early local-path validation.

    The canonical Gateway leaves runtime connectivity to Bindings. Existing
    Ray-only loader APIs historically raised ``FileNotFoundError`` before Ray
    started, so they invoke this check after resolving relative paths. Globs
    remain the selected engine's responsibility because ``Path.exists()``
    cannot validate their expansion semantics.
    """
    raw_path: str | None = None
    if isinstance(source, (ParquetSourceConfig, CsvSourceConfig)):
        raw_path = source.path
    elif (
        isinstance(source, ProviderSourceConfig)
        and source.provider in _LOCAL_FILE_PROVIDERS
    ):
        raw_path = source.uri
    if raw_path is None or any(marker in raw_path for marker in ("*", "?", "[")):
        return

    parts = urlsplit(raw_path)
    if parts.scheme.lower() not in {"", "file"}:
        return
    path = Path(unquote(parts.path) if parts.scheme else raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")


__all__ = ["require_local_file_source_exists", "resolve_file_source_path"]
