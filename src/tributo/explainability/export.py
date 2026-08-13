"""Export-boundary helpers for explainability companion artifacts.

This module is the integration point between the generic Bundle exporter and
the explainability capability planner.  Format-specific knowledge stays here;
the core export lifecycle only delegates to this helper.
"""

from __future__ import annotations

from tributo.exporting.models import BundleOutputConfig, ExportSource, ExportTarget
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
def prepare_bundle_output_config(
    config: BundleOutputConfig, source: ExportSource
) -> BundleOutputConfig:
    """Add the XGBoost UBJ companion required by automatic Tree SHAP.

    The helper is deliberately conservative: only an XGBoost source can
    trigger this companion.  Other sources must explicitly select an
    approximate model-agnostic backend and provide its reference binding.
    """
    explainability = config.explainability
    if not explainability.enabled or config.targets is None:
        return config
    data = config.model_dump()
    data["targets"] = [target.model_dump() for target in config.targets]
    changed = False
    if (
        source.source_kind == "xgboost_result"
        and explainability.backend == "auto"
        and not any(
            target.format in {"ubj", "xgboost-json"}
            or target.exporter_id in {"xgboost-ubj-v1", "xgboost-json-v1"}
            for target in config.targets
        )
    ):
        data["targets"].append(
            ExportTarget(
                name="explainability-model",
                format="ubj",
                exporter_id="xgboost-ubj-v1",
                required=True,
            ).model_dump()
        )
        data["roles"] = {
            **config.roles,
            "explainability_model": "explainability-model",
        }
        changed = True
    if source.source_kind in {"dnn_result", "pu_result"}:
        for target in data["targets"]:
            if target["format"] == "onnx":
                target["options"] = {
                    **target.get("options", {}),
                    "include_feature_map": True,
                }
                changed = True
    if not changed:
        return config
    return BundleOutputConfig.model_validate(data)


__all__ = ["prepare_bundle_output_config"]
