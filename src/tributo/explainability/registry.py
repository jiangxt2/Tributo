"""Lazy explainability adapter registry and entry-point discovery."""

from __future__ import annotations

import logging
from functools import lru_cache
from importlib.metadata import entry_points

from tributo.explainability.conformance import validate_adapter_conformance
from tributo.explainability.protocols import ExplainerAdapter
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="alpha")
class ExplainerRegistry:
    """Registry that validates adapter metadata without importing dependencies."""

    def __init__(self) -> None:
        self._adapters: dict[str, type[ExplainerAdapter]] = {}
        self._diagnostics: list[str] = []

    def register(self, adapter_cls: type[ExplainerAdapter]) -> None:
        adapter_id = getattr(adapter_cls, "adapter_id", None)
        validate_adapter_conformance(adapter_cls)
        if not isinstance(adapter_id, str):
            raise ValueError("explainer adapter_id must be a non-empty string")
        if adapter_id in self._adapters:
            raise ValueError(f"duplicate explainer adapter_id {adapter_id!r}")
        self._adapters[adapter_id] = adapter_cls

    def get(self, adapter_id: str) -> type[ExplainerAdapter]:
        try:
            return self._adapters[adapter_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown explainer adapter {adapter_id!r}; available: "
                f"{sorted(self._adapters)}"
            ) from exc

    def list_all(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def diagnostics(self) -> tuple[str, ...]:
        return tuple(self._diagnostics)

    def discover_entry_points(self) -> None:
        """Load only adapter classes; adapters load heavy dependencies later."""
        for ep in sorted(
            entry_points(group="tributo.explainers"), key=lambda e: e.name
        ):
            try:
                adapter_cls = ep.load()
                adapter_id = getattr(adapter_cls, "adapter_id", None)
                existing = (
                    self._adapters.get(adapter_id)
                    if isinstance(adapter_id, str)
                    else None
                )
                if (
                    existing is not None
                    and existing.__module__ == adapter_cls.__module__
                    and existing.__qualname__ == adapter_cls.__qualname__
                ):
                    continue
                self.register(adapter_cls)
            except ImportError as exc:
                message = f"{ep.name}: {type(exc).__name__}: {exc}"
                logger.warning("Failed to load explainability plugin %s", message)
                self._diagnostics.append(message)
            except Exception:
                # Invalid metadata and duplicate IDs are configuration errors;
                # hiding them could make a different adapter execute instead.
                raise


@PublicAPI(stability="alpha")
@lru_cache(maxsize=1)
def get_default_explainer_registry() -> ExplainerRegistry:
    """Return the lazily initialized first-party/entry-point registry."""
    from tributo.explainability.shap import ShapAdapter

    registry = ExplainerRegistry()
    registry.register(ShapAdapter)
    registry.discover_entry_points()
    return registry


__all__ = ["ExplainerRegistry", "get_default_explainer_registry"]
