"""CausalReportExporter — serialise causal effect as a JSON report.

Unlike model exporters, this produces ``artifact_kind="report"`` —
a structured JSON file containing the treatment, outcome, estimated
effect with confidence intervals, and refutation results.
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar, Mapping

from tributo.training.exporters.artifact_protocol import ARTIFACT_KIND_REPORT
from tributo.training.exporters.models import (
    ArtifactDraft,
    DraftFile,
    ExportContext,
    ExportSource,
    PlannedTarget,
    ProducerInfo,
    ResolvedArtifact,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class CausalReportExporter:
    """Export a causal study result as a JSON report.

    Expects ``source.metadata`` to contain the causal study dict
    (with ``effect``, ``refutation``, and optionally ``graph``).

    Class variables:
        artifact_kind: ``"report"`` — this is a statistics report, not
            a model.
    """

    api_version: ClassVar[int] = 1
    exporter_id: ClassVar[str] = "causal-report-v1"
    artifact_kind: ClassVar[str] = ARTIFACT_KIND_REPORT

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        """Export causal study as a JSON report.

        Args:
            context: Per-node export context (artifact_dir, execution_id).
            source: Read-only snapshot — must contain causal study data
                in ``source.metadata``.
            upstream: Resolved upstream artifacts (unused).
            target: Matched export target with typed options.

        Returns:
            An ``ArtifactDraft`` with ``artifact_kind="report"``.
        """
        causal_data = source.metadata.get("causal_study")
        if causal_data is None:
            raise ValueError(
                "CausalReportExporter requires source.metadata['causal_study'] "
                "to be set — it should contain the causal study result dict."
            )

        # Normalise dataclass objects to dicts for JSON serialisation.
        # We use manual field extraction instead of dataclasses.asdict()
        # because json.dump's default callback receives Any, which doesn't
        # match asdict's strict DataclassInstance overload signature.
        def _serialise(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {f: getattr(obj, f) for f in obj.__dataclass_fields__}
            raise TypeError(f"Cannot serialise {type(obj).__name__}")

        report = {
            "kind": "causal_report",
            "exporter_id": self.exporter_id,
            "study": causal_data,
        }

        output_path = context.artifact_dir / "causal_report.json"
        with open(str(output_path), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=_serialise)

        logger.info("Causal report written to %s", output_path)

        return ArtifactDraft(
            name=target.target.name,
            format="json",
            flavor_id="report",
            files=(DraftFile(relative_path="causal_report.json", role="aux"),),
            entrypoint="causal_report.json",
            producer=ProducerInfo(
                exporter_id=self.exporter_id,
            ),
            artifact_kind=self.artifact_kind,
        )
