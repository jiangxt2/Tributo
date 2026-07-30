"""StrictConfigModel — Pydantic base with extra="forbid".

All closed configuration models in Tributo inherit from this base to
reject unknown fields at validation time.  Only genuinely extensible
nodes (e.g. XGBoost ``ModelConfig`` for native parameters) override
with ``extra="allow"``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictConfigModel(BaseModel):
    """Pydantic model that rejects unknown fields.

    Inherit from this for every configuration model whose field set is
    fully known.  Unknown keys in the input dict will raise a
    ``ValidationError`` instead of being silently dropped.
    """

    model_config = ConfigDict(extra="forbid")
