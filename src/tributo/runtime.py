"""Top-level runtime composition helpers.

The composition root may depend on concrete integrations; domain modules do
not import this module at import time.
"""

from __future__ import annotations

from tributo.inference.contracts import ResultSinkProvider
from tributo.util.annotations import DeveloperAPI


@DeveloperAPI
def default_result_sink_provider() -> ResultSinkProvider:
    """Build the default sink provider from installed integrations."""
    from tributo.integrations.sinks.registry import default_result_sink_registry

    return default_result_sink_registry()


__all__ = ["default_result_sink_provider"]
