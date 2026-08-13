"""Verify ``@PublicAPI`` annotations are consistent with the stability levels
declared in ``docs/STABILITY.md``.

Two checks:

1. **Prototype/developer purity**: modules declared ``prototype`` or ``developer``
   in STABILITY_MAP must not carry ``@PublicAPI("beta")`` or ``@PublicAPI("stable")``
   symbols.  (Deprecated modules are exempt — they were once public.)

2. **Inventory completeness**: any module with ``@PublicAPI`` symbols must have an
   explicit entry in STABILITY_MAP (not just fallback to ``developer`` default).

This is the "API inventory check" gate.
"""

from __future__ import annotations

import ast
import os
from typing import Optional

import pytest

# ── Canonical module → stability (mirrors docs/STABILITY.md) ──────────────────
# Format:  "tributo.<module>": "<stability>"
# Omitted modules default to "developer" (internal).

STABILITY_MAP: dict[str, str] = {
    # Core — stable
    "tributo.config": "stable",
    "tributo.job": "stable",
    "tributo.exceptions": "stable",
    # Core — alpha
    "tributo.ray_jobs": "alpha",
    # Core — beta
    "tributo.cli": "beta",
    # Portable algorithm execution — alpha
    "tributo.algorithms.api.context": "alpha",
    "tributo.algorithms.api.artifacts": "alpha",
    "tributo.algorithms.api.descriptor": "alpha",
    "tributo.algorithms.api.distribution": "alpha",
    "tributo.algorithms.api.errors": "alpha",
    "tributo.algorithms.api.execution": "alpha",
    "tributo.algorithms.api.models": "alpha",
    "tributo.algorithms.builtin.multinomial_nb": "alpha",
    "tributo.algorithms.builtin.torch_collective": "alpha",
    "tributo.algorithms.builtin.xgboost_native": "alpha",
    "tributo.algorithms.composition": "alpha",
    "tributo.algorithms.core.builder": "alpha",
    "tributo.algorithms.core.runtime": "alpha",
    "tributo.algorithms.spi.execution": "alpha",
    "tributo.algorithms.spi.input": "alpha",
    # Util — stable
    "tributo.util.annotations": "stable",
    # Training — beta
    "tributo.training.base": "beta",
    "tributo.training.results": "beta",
    "tributo.training.checkpoint": "beta",
    "tributo.training.config": "beta",
    "tributo.training.xgboost_trainer": "beta",
    "tributo.training.dnn_trainer": "beta",
    "tributo.training.pu_trainer": "beta",
    "tributo.training.flavor": "beta",
    "tributo.training.algorithm_spec": "beta",
    "tributo.training.catalog": "beta",
    "tributo.training.data_loader": "beta",
    "tributo.training.registry": "beta",
    "tributo.training.job_submitter": "beta",
    "tributo.training.local_runner": "beta",
    "tributo.training.tune_config": "beta",
    "tributo.training.tune_runner": "beta",
    "tributo.training.tune_space": "beta",
    "tributo.training.priors": "beta",
    "tributo.training.causal_estimator": "beta",
    "tributo.training.losses": "beta",
    "tributo.training.models": "beta",
    "tributo.training.features": "beta",
    "tributo.training.exporters.artifact_protocol": "deprecated",
    # Training — alpha / deprecated
    "tributo.training.graph_trainer": "alpha",
    "tributo.training.onnx_exporter": "deprecated",
    "tributo.training.exporters": "deprecated",
    # Data — beta
    "tributo.data.source_config": "beta",
    "tributo.data.provider_registry": "beta",
    "tributo.data.refs": "beta",
    # Data — alpha candidate ingestion contract
    "tributo.data.scan_plan": "developer",
    "tributo.data.ingestion": "alpha",
    "tributo.data.handle_adapters": "alpha",
    "tributo.data.contracts.handles": "alpha",
    "tributo.data.contracts.modes": "beta",
    "tributo.data.contracts.storage": "beta",
    "tributo.data.transform_ir": "alpha",
    # Data — beta
    "tributo.data.provider": "beta",
    "tributo.data.graph": "beta",
    # Data — internal engine translation
    "tributo.data.engine_binding": "developer",
    "tributo.data.provider_plugins": "developer",
    "tributo.data.transform_compiler": "developer",
    # Data — alpha candidate write contract
    "tributo.data.writing.capabilities": "alpha",
    "tributo.data.writing.contracts": "alpha",
    "tributo.data.writing.gateway": "alpha",
    "tributo.data.writing": "alpha",
    # Distributed Lance vector indexing — alpha
    "tributo.vector_index.contracts": "alpha",
    "tributo.vector_index.index_job": "alpha",
    "tributo.vector_index.maintenance": "alpha",
    "tributo.vector_index.search": "alpha",
    # Exporting — beta
    "tributo.exporting.service": "beta",
    "tributo.exporting.models": "beta",
    "tributo.exporting.protocols": "beta",
    "tributo.exporting.manifest": "beta",
    "tributo.exporting.bundle_reader": "beta",
    "tributo.exporting.planner": "beta",
    "tributo.exporting.executor": "beta",
    "tributo.exporting.publisher": "beta",
    "tributo.exporting.validators": "beta",
    "tributo.exporting.registries": "beta",
    "tributo.exporting.options": "beta",
    "tributo.exporting.records": "beta",
    "tributo.exporting.gc": "beta",
    "tributo.exporting.hooks": "beta",
    "tributo.exporting.capabilities": "beta",
    "tributo.exporting.events": "beta",
    "tributo.exporting.dispatch": "beta",
    "tributo.exporting.repository": "beta",
    "tributo.exporting.runtime": "beta",
    "tributo.exporting.conftest": "beta",
    # Explainability — alpha
    "tributo.explainability.conformance": "alpha",
    "tributo.explainability.contracts": "alpha",
    "tributo.explainability.executor": "alpha",
    "tributo.explainability.export": "alpha",
    "tributo.explainability.job_runner": "alpha",
    "tributo.explainability.planner": "alpha",
    "tributo.explainability.protocols": "alpha",
    "tributo.explainability.reference": "alpha",
    "tributo.explainability.registry": "alpha",
    "tributo.explainability.shap": "alpha",
    # Integrations — beta
    "tributo.integrations.algorithm_inputs": "alpha",
    "tributo.integrations.algorithm_inputs.ingestion": "alpha",
    "tributo.integrations.exporters": "beta",
    "tributo.integrations.flavors": "beta",
    "tributo.integrations.flavors.onnx_runtime": "beta",
    "tributo.integrations.flavors.xgboost_native": "alpha",
    "tributo.integrations.model_importers.artifact": "alpha",
    "tributo.integrations.model_importers.mlflow": "alpha",
    "tributo.integrations.model_importers.registry": "alpha",
    "tributo.integrations.validators": "beta",
    "tributo.integrations.sources": "beta",
    "tributo.integrations.storage": "beta",
    "tributo.integrations.hooks": "beta",
    "tributo.integrations.sinks.parquet": "alpha",
    "tributo.integrations.sinks.lance": "alpha",
    "tributo.integrations.broker": "alpha",
    "tributo.integrations.broker_registry": "alpha",
    # Inference — beta
    "tributo.inference.base": "beta",
    "tributo.inference.batch_predictor": "beta",
    "tributo.inference.pipeline": "beta",
    "tributo.inference.job_runner": "beta",
    "tributo.inference.api": "alpha",
    "tributo.inference.bundle_predictor": "alpha",
    "tributo.inference.contracts": "alpha",
    "tributo.inference.executor": "alpha",
    "tributo.inference.importers": "alpha",
    "tributo.inference.input_resolver": "alpha",
    "tributo.inference.post_training": "alpha",
    "tributo.inference.resolver": "alpha",
    # Serving — beta
    "tributo.serving.serve_runner": "beta",
    "tributo.serving.model_deployment": "beta",
    "tributo.serving.schema": "beta",
    "tributo.serving.grpc_deployment": "beta",
    "tributo.serving.grpc_runner": "beta",
    "tributo.serving.streaming_runner": "beta",
    "tributo.serving.composition": "beta",
    # Serving — alpha
    "tributo.serving.streaming_deployment": "alpha",
    # Serving — developer (generated)
    "tributo.serving.proto": "developer",
    # Streaming — beta
    "tributo.streaming.protocol": "beta",
    # Streaming — alpha
    "tributo.streaming.kafka_source": "alpha",
    # Pipeline — alpha
    "tributo.pipeline.core": "alpha",
    # Registry — beta
    "tributo.registry.model_registry": "beta",
    "tributo.registry.schema": "beta",
    "tributo.registry.callback": "beta",
    "tributo.registry.mlflow_util": "developer",
    # Plugin — beta
    "tributo.plugin": "beta",
    # Common — developer
    "tributo._common": "developer",
    "tributo._common.storage_profiles": "beta",
    "tributo._common.dependencies": "beta",
}

#: Symbol-level stability overrides for modules with mixed stability.
#: When a module's STABILITY_MAP entry doesn't capture per-symbol granularity,
#: list the exceptions here.  Keys are ``module.symbol``, values are the
#: symbol-specific stability.
_SYMBOL_OVERRIDES: dict[str, str] = {
    "tributo.plugin.validate_distributed_algorithm_descriptor": "alpha",
    "tributo.config.AlgorithmExecutionConfig": "alpha",
    "tributo.config.AlgorithmInputConfig": "alpha",
    "tributo.config.AlgorithmWorkerResourcesConfig": "alpha",
    "tributo.config.LocalRayRuntimeConfig": "alpha",
    "tributo.exceptions.ResultMaterializationError": "alpha",
    "tributo.exceptions.BundleExportError": "beta",
    "tributo.exceptions.BundleCommitBusyError": "beta",
    "tributo.exceptions.AliasConflict": "beta",
    "tributo.exceptions.UnsupportedArtifactFormat": "beta",
    "tributo.exceptions.PostPublishCallbackError": "beta",
    "tributo.exceptions.PluginLoadIssue": "beta",
    "tributo.exceptions.StreamSourceError": "beta",
    "tributo.exceptions.KafkaCommitError": "beta",
    "tributo.exceptions.KafkaPoisonMessageError": "beta",
    "tributo.exceptions.ResourceBudgetExceededError": "beta",
    "tributo.exceptions.EngineNotAvailableError": "alpha",
    "tributo.inference.job_runner.InferenceJobAttempt": "alpha",
    "tributo.inference.job_runner.InferenceJobResult": "alpha",
    "tributo.inference.job_runner.map_ray_job_status": "alpha",
    "tributo.inference.job_runner.submit_inference_request": "alpha",
    "tributo.inference.job_runner.submit_inference_request_with_identity": "alpha",
    "tributo.inference.job_runner.submit_resolved_inference": "alpha",
    "tributo.inference.job_runner.submit_resolved_inference_with_identity": "alpha",
    "tributo.inference.job_runner.submit_inference_request_with_retry": "alpha",
    "tributo.inference.job_runner.wait_for_job": "alpha",
    "tributo.training.job_submitter.submit_training_job_with_identity": "alpha",
    "tributo.training.xgboost_trainer.run_training_result_with_config": "alpha",
}


def _get_expected_stability(module_name: str, symbol_name: str = "") -> str:
    """Return the expected stability for *symbol_name* in *module_name*.

    Checks symbol-level overrides first, then falls back to the module-level
    ``_get_module_stability``.
    """
    if symbol_name:
        key = f"{module_name}.{symbol_name}"
        if key in _SYMBOL_OVERRIDES:
            return _SYMBOL_OVERRIDES[key]
    return _get_module_stability(module_name)


#: Public API stability levels that signal "this is consumer-facing".
_CONSUMER_STABILITIES: frozenset[str] = frozenset({"beta", "stable"})


def _get_module_stability(module_name: str) -> str:
    """Return the expected stability for *module_name*."""
    if module_name in STABILITY_MAP:
        return STABILITY_MAP[module_name]
    # Parent match
    parts = module_name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        parent = ".".join(parts[:i])
        if parent in STABILITY_MAP:
            return STABILITY_MAP[parent]
    # Top-level tributo.xxx default
    if module_name.startswith("tributo.") and module_name.count(".") == 1:
        return "beta"
    return "developer"


def _resolve_source_path(module_name: str) -> str | None:
    """Resolve *module_name* to a source file path without importing."""
    parts = module_name.split(".")
    if parts[0] != "tributo":
        return None
    rel = os.path.join(*parts[1:])
    src_root = os.path.join(os.path.dirname(__file__), "..", "src", "tributo")
    src_root = os.path.abspath(src_root)
    # Try <module>.py first, then <module>/__init__.py
    py_path = os.path.join(src_root, rel + ".py")
    if os.path.isfile(py_path):
        return py_path
    init_path = os.path.join(src_root, rel, "__init__.py")
    if os.path.isfile(init_path):
        return init_path
    return None


def _parse_file_ast(file_path: str) -> list[tuple[str, str]]:
    """Parse *file_path* via AST and return [(symbol, stability), ...]."""
    try:
        with open(file_path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
    except (SyntaxError, OSError):
        return []
    results: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                stability = _parse_public_api_decorator(decorator)
                if stability:
                    results.append((node.name, stability))
                    break
    return results


def _extract_public_api_symbols(module_name: str) -> list[tuple[str, str]]:
    """Return ``@PublicAPI`` symbol annotations for *module_name*.

    Parse source directly so inventory collection never executes module import
    side effects or turns import-time deprecation warnings into test failures.
    """
    source_path = _resolve_source_path(module_name)
    if source_path is not None:
        return _parse_file_ast(source_path)

    return []


def _parse_public_api_decorator(node: ast.expr) -> Optional[str]:
    """Return stability string if *node* is ``@PublicAPI(stability="...")``."""
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "PublicAPI":
            for kw in node.keywords:
                if kw.arg == "stability" and isinstance(kw.value, ast.Constant):
                    val = kw.value.value
                    if isinstance(val, str):
                        return val
            return "beta"  # @PublicAPI without stability= defaults to beta
        elif isinstance(func, ast.Attribute) and func.attr == "PublicAPI":
            for kw in node.keywords:
                if kw.arg == "stability" and isinstance(kw.value, ast.Constant):
                    val = kw.value.value
                    if isinstance(val, str):
                        return val
            return "beta"
    return None


def _skip_reason(module_name: str) -> Optional[str]:
    """Return a skip reason, or ``None`` if the module should be tested."""
    if module_name.startswith("tributo.serving.proto"):
        return "generated protobuf code"
    return None


def _documented_stability_modules() -> set[str]:
    """Extract exact module entries from the canonical Markdown inventory."""
    docs_path = os.path.join(os.path.dirname(__file__), "..", "docs", "STABILITY.md")
    with open(docs_path, encoding="utf-8") as stream:
        text = stream.read()

    import re

    return set(re.findall(r"`(tributo\.[A-Za-z0-9_.*]+)`", text))


def test_stability_map_is_documented() -> None:
    """Every module-level stability declaration must appear in STABILITY.md."""
    documented = _documented_stability_modules()
    # A trailing ``.*`` entry (e.g. ``tributo.integrations.model_importers.*``)
    # documents every submodule below that prefix.
    covered: set[str] = set()
    for pattern in documented:
        if pattern.endswith(".*"):
            covered.update(
                module for module in STABILITY_MAP if module.startswith(pattern[:-1])
            )
        else:
            covered.add(pattern)
    missing = sorted(set(STABILITY_MAP) - covered)
    assert not missing, (
        "STABILITY_MAP modules missing from docs/STABILITY.md: " + ", ".join(missing)
    )


def _discover_tributo_modules() -> list[str]:
    """Walk ``src/tributo/`` and return fully-qualified module names."""
    root = os.path.join(os.path.dirname(__file__), "..", "src", "tributo")
    root = os.path.abspath(root)
    modules: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".py") and not fname.startswith("_"):
                rel = os.path.relpath(os.path.join(dirpath, fname[:-3]), root)
                mod = "tributo." + rel.replace(os.sep, ".")
                modules.append(mod)
        if "__init__.py" in filenames:
            rel = os.path.relpath(dirpath, root)
            if rel == ".":
                continue
            mod = "tributo." + rel.replace(os.sep, ".")
            modules.append(mod)
    return sorted(set(modules))


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("module_name", _discover_tributo_modules())
def test_no_beta_symbols_in_prototype_or_developer_module(
    module_name: str,
) -> None:
    """Modules declared ``prototype`` or ``developer`` must not carry
    ``@PublicAPI("beta")`` or ``@PublicAPI("stable")`` symbols.
    """
    reason = _skip_reason(module_name)
    if reason:
        pytest.skip(reason)

    expected = _get_module_stability(module_name)
    if expected not in ("prototype", "developer", "deprecated"):
        return

    symbols = _extract_public_api_symbols(module_name)
    if not symbols:
        return

    if expected == "deprecated":
        return  # Deprecated modules may retain @PublicAPI (legacy surface)

    pub_symbols = [(n, s) for n, s in symbols if s in _CONSUMER_STABILITIES]
    if pub_symbols:
        names = ", ".join(n for n, _ in pub_symbols)
        pytest.fail(
            f"{module_name} is declared '{expected}' in STABILITY.md, "
            f"but contains @PublicAPI('beta'/'stable') symbols: {names}. "
            f"Either remove @PublicAPI or update STABILITY.md."
        )


@pytest.mark.parametrize("module_name", _discover_tributo_modules())
def test_public_api_modules_in_inventory(module_name: str) -> None:
    """Modules with ``@PublicAPI`` symbols must have an explicit entry in
    STABILITY_MAP — not just the ``developer`` fallback default.
    """
    reason = _skip_reason(module_name)
    if reason:
        pytest.skip(reason)

    expected = _get_module_stability(module_name)
    if expected != "developer":
        return  # Has explicit entry or non-developer default

    symbols = _extract_public_api_symbols(module_name)
    if not symbols:
        return  # No @PublicAPI symbols → doesn't need explicit entry

    names = ", ".join(n for n, _ in symbols)
    has_explicit = module_name in STABILITY_MAP
    if not has_explicit:
        pytest.fail(
            f"{module_name} has @PublicAPI symbols ({names}) but is not "
            f"listed in STABILITY_MAP. Add it to the test and to "
            f"docs/STABILITY.md."
        )


@pytest.mark.parametrize("module_name", _discover_tributo_modules())
def test_per_symbol_stability_matches_module(module_name: str) -> None:
    """For non-deprecated/non-prototype/non-developer modules, every
    ``@PublicAPI`` symbol's stability must match the module's declared
    level — unless overridden in ``_SYMBOL_OVERRIDES``.

    This catches M1-type issues: source says ``beta`` but module says
    ``alpha`` (or vice versa).
    """
    reason = _skip_reason(module_name)
    if reason:
        pytest.skip(reason)

    expected = _get_module_stability(module_name)
    if expected in ("prototype", "developer", "deprecated"):
        return  # These modules have their own checks

    symbols = _extract_public_api_symbols(module_name)
    if not symbols:
        return

    mismatches: list[str] = []
    for sym_name, sym_stability in symbols:
        sym_expected = _get_expected_stability(module_name, sym_name)
        if sym_stability != sym_expected:
            mismatches.append(
                f"  {module_name}.{sym_name}: "
                f"source=@{sym_stability} vs STABILITY_MAP=@{sym_expected}"
            )

    if mismatches:
        pytest.fail(
            f"{module_name}: {len(mismatches)} symbol-level mismatch(es):\n"
            + "\n".join(mismatches)
            + "\n\nEither update the @PublicAPI annotation in source, "
            "the STABILITY_MAP entry, or add a _SYMBOL_OVERRIDES entry."
        )
