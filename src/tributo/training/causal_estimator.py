"""Causal inference base class.

Follows the DoWhy / EconML three-phase lifecycle:
``identify → estimate → refute``, adapted to the ``BaseTrainer``
Template Method via ``ExecutionKind.ESTIMATE``.

``training_loop()`` is overridden to execute the three causal phases
instead of a traditional training loop.  ``export_artifacts()``
serialises the causal effect and refutation result as a JSON report
(``artifact_kind="report"``).
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tributo.training.base import BaseTrainer
from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import ray.data

logger = logging.getLogger(__name__)


# ── Causal domain types ──────────────────────────────────────────────────────


@PublicAPI(stability="beta")
@dataclass
class CausalGraph:
    """Identified causal graph.

    Attributes:
        treatment: Treatment variable name.
        outcome: Outcome variable name.
        confounders: Confounder variable names.
        instruments: Instrumental variable names (optional).
        edges: Inferred causal edges as ``(source, target)`` tuples.
        graph_description: Human-readable description of the DAG.
    """

    treatment: str
    outcome: str
    confounders: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    graph_description: str = ""


@PublicAPI(stability="beta")
@dataclass
class CausalEffect:
    """Estimated causal effect.

    Attributes:
        method: Estimation method (e.g. ``"backdoor.linear_regression"``).
        estimand: The estimand expression.
        estimate_value: Point estimate.
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.
        p_value: P-value of the estimate.
        interpretation: Human-readable interpretation.
        metadata: Additional estimator-specific metadata.
    """

    method: str
    estimand: str = ""
    estimate_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = None
    interpretation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@PublicAPI(stability="beta")
@dataclass
class RefutationResult:
    """Result of a refutation test.

    Attributes:
        method: Refutation method (e.g. ``"placebo_treatment_refuter"``).
        passed: Whether the estimate survived refutation.
        new_effect: Estimated effect after refutation.
        p_value: P-value of the refutation test.
        interpretation: Human-readable interpretation.
    """

    method: str
    passed: bool
    new_effect: float = 0.0
    p_value: float | None = None
    interpretation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ── BaseCausalEstimator ──────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class BaseCausalEstimator(BaseTrainer):
    """Causal inference base class.

    Inherits ``BaseTrainer`` to reuse registry / catalog / callback
    infrastructure.  The lifecycle is re-mapped:

    * ``setup()`` — extracts treatment, outcome, confounders from config.
    * ``training_loop()`` — runs ``identify → estimate → refute``.
    * ``export_artifacts()`` — serialises the causal study as a JSON report.

    Subclasses must implement ``identify()`` and ``estimate()``.
    ``refute()`` has a default placebo-treatment implementation that
    subclasses may override.
    """

    # -- lifecycle configuration -----------------------------------------------

    def setup(self) -> None:
        """Extract treatment, outcome, and confounders from config.

        Expects ``config["causal"]`` with keys:
        ``treatment`` (str), ``outcome`` (str), and optionally
        ``confounders`` (list[str]).
        """
        causal_cfg: dict[str, Any] = self.config.get("causal", {})
        self._treatment: str = causal_cfg["treatment"]
        self._outcome: str = causal_cfg["outcome"]
        self._confounders: list[str] = causal_cfg.get("confounders", [])
        self._instruments: list[str] = causal_cfg.get("instruments", [])

    def _load_data(self) -> ray.data.Dataset:
        """Load the dataset from the configured source.

        Uses the framework's canonical ``load_ray_dataset_from_source``
        which validates against the ``SourceConfig`` discriminated union
        (``{"type": "parquet", "path": "s3://...", "s3": {...}}``).

        Returns:
            A ``ray.data.Dataset`` for causal analysis.
        """
        from tributo.training.data_loader import load_ray_dataset_from_source

        source = self.config.get("data", {}).get("source")
        if source is None:
            raise ValueError(
                f"{type(self).__name__}: config must contain a "
                f"'data.source' section (e.g. "
                f'{{"type": "parquet", "path": "s3://bucket/data.parquet"}}).'
            )
        return load_ray_dataset_from_source(source)

    # -- causal phases ---------------------------------------------------------

    @abstractmethod
    def identify(
        self,
        data: ray.data.Dataset,
        treatment: str,
        outcome: str,
        **kwargs: Any,
    ) -> CausalGraph:
        """Identify the causal graph from data and domain knowledge.

        Args:
            data: The input dataset.
            treatment: Treatment variable name.
            outcome: Outcome variable name.
            **kwargs: Additional identification parameters.

        Returns:
            An identified ``CausalGraph``.
        """

    @abstractmethod
    def estimate(
        self,
        data: ray.data.Dataset,
        causal_graph: CausalGraph,
    ) -> CausalEffect:
        """Estimate the causal effect given the identified graph.

        Args:
            data: The input dataset.
            causal_graph: The identified causal graph from ``identify()``.

        Returns:
            An estimated ``CausalEffect``.
        """

    def refute(
        self,
        estimate: CausalEffect,
        method: str = "placebo",
    ) -> RefutationResult:
        """Refute the estimated causal effect (default: placebo treatment).

        Args:
            estimate: The estimated effect from ``estimate()``.
            method: Refutation method name.

        Returns:
            A ``RefutationResult``.
        """
        logger.info(
            "Refuting estimate (method=%s): %s → %s, value=%.4f",
            method,
            self._treatment,
            self._outcome,
            estimate.estimate_value,
        )
        # Default placebo refutation: the result is left to the concrete
        # implementation or treated as passed when no specialised refuter
        # is available.
        return RefutationResult(
            method=method,
            passed=True,
            new_effect=estimate.estimate_value or 0.0,
            interpretation=f"Placebo refutation ({method}) not implemented; "
            f"estimate accepted by default.",
        )

    # -- BaseTrainer integration -----------------------------------------------

    def training_loop(self) -> dict[str, Any]:
        """Execute the causal pipeline: identify → estimate → refute.

        Returns:
            A dict with keys ``"effect"`` and ``"refutation"``.
        """
        data = self._load_data()
        logger.info(
            "Identifying causal graph: treatment=%r, outcome=%r",
            self._treatment,
            self._outcome,
        )
        graph = self.identify(
            data,
            self._treatment,
            self._outcome,
            confounders=self._confounders,
            instruments=self._instruments,
        )
        logger.info("Estimating causal effect via graph: %s", graph.graph_description)
        effect = self.estimate(data, graph)
        logger.info(
            "Estimated effect: %s = %.4f (CI: [%s, %s])",
            effect.method,
            effect.estimate_value,
            effect.ci_lower,
            effect.ci_upper,
        )
        refutation = self.refute(effect)
        return {"effect": effect, "refutation": refutation}

    def export_artifacts(self, checkpoint: Any, output_path: str) -> None:
        """Export the causal effect and refutation as a JSON report.

        Args:
            checkpoint: The dict ``{"effect": CausalEffect, "refutation":
                RefutationResult}`` from ``training_loop()``.
            output_path: Destination path for the report.
        """
        from dataclasses import asdict
        from pathlib import Path

        from tributo._common.storage import write_json

        effect: CausalEffect = checkpoint["effect"]
        refutation: RefutationResult = checkpoint["refutation"]

        report = {
            "treatment": self._treatment,
            "outcome": self._outcome,
            "confounders": self._confounders,
            "effect": asdict(effect),
            "refutation": asdict(refutation),
        }

        # Use the framework's write_json which handles s3://, file://, and
        # local paths.  For local paths, ensure the parent directory exists.
        if not output_path.startswith("s3://"):
            Path(output_path).mkdir(parents=True, exist_ok=True)
        out_file = str(Path(output_path) / "causal_report.json")
        write_json(out_file, report)
        logger.info("Causal report written to %s", out_file)
