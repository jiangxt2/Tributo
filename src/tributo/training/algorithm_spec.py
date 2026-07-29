"""Algorithm specification with ML metadata.

Extends the legacy ``TrainerSpec`` with three semantic groups of fields:

* **capabilities** — what the algorithm *can do* (lifecycle actions, resource hints).
* **problem_types** — what ML problems it *solves* (classification, ranking, etc.).
* **data_contract** — what data shape it *expects and produces*.

Backward-compatible: ``TrainerSpec`` is a pure type alias in ``base.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Problem type taxonomy
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
class ProblemType(str, Enum):
    """ML problem types an algorithm can address.

    These describe the *semantic* task, orthogonal to lifecycle capabilities
    (train / finetune / predict) declared in ``supported_tasks``.
    """

    BINARY_CLASSIFICATION = "binary_classification"
    MULTI_CLASS_CLASSIFICATION = "multi_class_classification"
    MULTI_LABEL_CLASSIFICATION = "multi_label_classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"
    PU_LEARNING = "pu_learning"


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class DataContract:
    """Describes the expected input / output data shape for an algorithm.

    Attributes:
        columns: Expected column names and their types.
        sparse: List of sparse feature column name patterns.
        dense: List of dense feature column name patterns.
        min_rows: Minimum recommended row count (``None`` = no constraint).
    """

    columns: dict[str, str] = field(default_factory=dict)
    sparse: list[str] = field(default_factory=list)
    dense: list[str] = field(default_factory=list)
    min_rows: int | None = None


# ---------------------------------------------------------------------------
# Resource hints
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class ResourceHints:
    """Runtime resource recommendations for an algorithm.

    These are *hints*, not hard requirements — the scheduler may override them.

    Attributes:
        gpu_required: Whether a GPU is necessary (vs. nice-to-have).
        min_memory_gb: Minimum recommended RAM in GiB.
        min_cpus: Minimum recommended CPU cores.
    """

    gpu_required: bool = False
    min_memory_gb: int = 2
    min_cpus: int = 1


# ---------------------------------------------------------------------------
# AlgorithmSpec
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class AlgorithmSpec:
    """Complete specification of a trainable algorithm.

    Field groups are named after their semantic domain to prevent confusion
    between ``supported_tasks`` (capabilities — "can it train?") and
    ``problem_types`` (ML semantics — "is it a classifier?").

    Attributes:
        name: Short algorithm name (e.g. ``"xgboost"``).
        trainer_cls: ``BaseTrainer`` subclass.
        default_config: Default configuration dict, overridable by user config.
        supported_tasks: Lifecycle actions the algorithm supports
            (``["train"]``, ``["train", "finetune", "predict"]``, etc.).
        version: Algorithm implementation version (semver string or commit).
        resource_hints: Runtime resource recommendations.
        extras_group: ``pip install`` extra needed (``"training"``, ``"identity"``).
        problem_types: ML problem types this algorithm can address.
        data_modality: Input data modalities (``["tabular"]``, ``["graph", "tabular"]``).
        tags: Free-form classification tags.
        input_schema: Expected input feature description.
        output_schema: Expected output format description.
        config_model: Pydantic model for config validation (Phase 1 onward).
    """

    # -- required fields (no defaults — must be positional args 1 & 2) ----------
    name: str
    trainer_cls: type

    # -- legacy TrainerSpec fields (kept for backward compat) --------------------
    default_config: dict[str, Any] = field(default_factory=dict)
    supported_tasks: list[str] = field(default_factory=lambda: ["train"])

    # -- version & provenance ---------------------------------------------------
    version: str = "0.1.0"

    # -- capabilities -----------------------------------------------------------
    resource_hints: ResourceHints = field(default_factory=ResourceHints)
    extras_group: str | None = None

    # -- problem types ----------------------------------------------------------
    problem_types: list[ProblemType] = field(default_factory=list)
    data_modality: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # -- data contract ----------------------------------------------------------
    input_schema: DataContract | None = None
    output_schema: DataContract | None = None
    config_model: type[BaseModel] | None = None
