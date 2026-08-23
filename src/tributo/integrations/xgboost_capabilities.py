"""Model-instance capability predicates shared by XGBoost integrations."""

from __future__ import annotations


def supports_native_tree_shap(*, booster_kind: str, objective: str) -> bool:
    """Return whether one loaded Booster satisfies the native TreeSHAP contract."""
    return booster_kind in {"gbtree", "dart"} and (
        objective.startswith(("binary:", "multi:"))
        or objective in {"reg:squarederror", "reg:logistic"}
    )
