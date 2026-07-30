"""Algorithm specification with ML metadata.

Extends the legacy ``TrainerSpec`` with three semantic groups of fields:

* **capabilities** — what the algorithm *can do* (lifecycle actions, resource hints).
* **problem_types** — what ML problems it *solves* (classification, ranking, etc.).
* **data_contract** — what data shape it *expects and produces*.

Phase 3 adds:

* **taxonomy** — ``AlgorithmStatus``, ``ProblemFamily``, ``DataLoadingMode``.
* **lifecycle** — ``status``, ``deprecated_since``, ``replacement`` with
  ``__post_init__`` invariants.
* **deep immutability** — all container fields are recursively frozen via
  ``deep_freeze`` so that a ``Catalog`` snapshot cannot be mutated back
  into the ``Registry``.

Backward-compatible: ``TrainerSpec`` is a pure type alias in ``base.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from tributo._common.immutable import deep_freeze
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
# Problem family (aggregation over ProblemType)
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
class ProblemFamily(str, Enum):
    """Logical groupings of ``ProblemType`` values for CLI / filtering.

    ``classification`` expands to BINARY + MULTI_CLASS + MULTI_LABEL.
    """

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    PU_LEARNING = "pu_learning"


PROBLEM_FAMILY_MAP: dict[ProblemFamily, tuple[ProblemType, ...]] = {
    ProblemFamily.CLASSIFICATION: (
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTI_CLASS_CLASSIFICATION,
        ProblemType.MULTI_LABEL_CLASSIFICATION,
    ),
    ProblemFamily.REGRESSION: (ProblemType.REGRESSION,),
    ProblemFamily.RANKING: (ProblemType.RANKING,),
    ProblemFamily.CLUSTERING: (ProblemType.CLUSTERING,),
    ProblemFamily.ANOMALY_DETECTION: (ProblemType.ANOMALY_DETECTION,),
    ProblemFamily.PU_LEARNING: (ProblemType.PU_LEARNING,),
}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
class AlgorithmStatus(str, Enum):
    """Algorithm lifecycle stage.

    ``READY`` — fully supported, appears in ``list()`` and recommendations.
    ``DEPRECATED`` — hidden by default in ``list()``, triggers
    ``FutureWarning`` on ``get_spec()``.
    """

    READY = "ready"
    DEPRECATED = "deprecated"


# ---------------------------------------------------------------------------
# Data loading ownership
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
class DataLoadingMode(str, Enum):
    """Who is responsible for loading data.

    ``LEGACY_DRIVER`` — old plugin contract; flat ``data`` dict is
    normalised by the legacy adapter and the Runner loads data.

    ``CANONICAL_DRIVER`` — Runner loads data from ``data.source``
    before constructing the trainer (XGBoost, DNN).

    ``CANONICAL_TRAINER`` — the trainer itself loads data inside the
    worker from the normalised ``data.source`` (PU).
    """

    LEGACY_DRIVER = "legacy_driver"
    CANONICAL_DRIVER = "canonical_driver"
    CANONICAL_TRAINER = "canonical_trainer"


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class DataContract:
    """Describes the expected input / output data shape for an algorithm.

    Attributes:
        columns: Expected column names and their types.
        sparse: Sparse feature column name patterns.
        dense: Dense feature column name patterns.
        min_rows: Minimum recommended row count (``None`` = no constraint).
    """

    columns: Mapping[str, str] = field(default_factory=dict)
    sparse: tuple[str, ...] = ()
    dense: tuple[str, ...] = ()
    min_rows: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", deep_freeze(self.columns))
        object.__setattr__(self, "sparse", tuple(self.sparse))
        object.__setattr__(self, "dense", tuple(self.dense))


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
        default_config: Default configuration dict (deep-frozen at init).
        supported_tasks: Lifecycle actions the algorithm supports.
        version: Algorithm implementation version.
        resource_hints: Runtime resource recommendations.
        extras_group: ``pip install`` extra needed (``"training"``, ``"identity"``).
        problem_types: ML problem types this algorithm can address.
        data_modality: Input data modalities (``["tabular"]``, ``["graph"]``).
        tags: Free-form classification tags.
        input_schema: Expected input feature description.
        output_schema: Expected output format description.
        config_model: Pydantic model for config validation.
        status: Lifecycle stage.
        deprecated_since: Version when deprecated (e.g. ``"0.4.0"``).
        replacement: Name of the replacement algorithm.
        data_loading: Who owns data loading.
    """

    # -- required fields -------------------------------------------------------
    name: str
    trainer_cls: type

    # -- legacy TrainerSpec fields ---------------------------------------------
    default_config: Mapping[str, Any] = field(default_factory=dict)
    supported_tasks: tuple[str, ...] = ("train",)

    # -- version & provenance --------------------------------------------------
    version: str = "0.1.0"

    # -- capabilities ----------------------------------------------------------
    resource_hints: ResourceHints = field(default_factory=ResourceHints)
    extras_group: str | None = None

    # -- problem types ---------------------------------------------------------
    problem_types: tuple[ProblemType, ...] = ()
    data_modality: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    # -- data contract ---------------------------------------------------------
    input_schema: DataContract | None = None
    output_schema: DataContract | None = None
    config_model: type[BaseModel] | None = None

    # -- lifecycle -------------------------------------------------------------
    status: AlgorithmStatus = AlgorithmStatus.READY
    deprecated_since: str | None = None
    replacement: str | None = None

    # -- data loading ----------------------------------------------------------
    data_loading: DataLoadingMode = DataLoadingMode.LEGACY_DRIVER

    def __post_init__(self) -> None:
        # ── deep-freeze mutable containers ──
        object.__setattr__(self, "default_config", deep_freeze(self.default_config))
        object.__setattr__(self, "supported_tasks", tuple(self.supported_tasks))
        object.__setattr__(self, "problem_types", tuple(self.problem_types))
        object.__setattr__(self, "data_modality", tuple(self.data_modality))
        object.__setattr__(self, "tags", tuple(self.tags))

        # ── lifecycle invariants ──
        if self.status == AlgorithmStatus.DEPRECATED:
            if self.deprecated_since is None:
                raise ValueError(
                    f"Algorithm '{self.name}': status=DEPRECATED requires "
                    f"deprecated_since (e.g. '0.4.0')"
                )
            if self.replacement is None:
                raise ValueError(
                    f"Algorithm '{self.name}': status=DEPRECATED requires "
                    f"replacement (name of the replacement algorithm)"
                )
            if self.replacement == self.name:
                raise ValueError(
                    f"Algorithm '{self.name}': replacement must not be self"
                )
        else:  # READY
            if self.deprecated_since is not None:
                raise ValueError(
                    f"Algorithm '{self.name}': deprecated_since must be None "
                    f"when status={self.status.value}"
                )
            if self.replacement is not None:
                raise ValueError(
                    f"Algorithm '{self.name}': replacement must be None "
                    f"when status={self.status.value}"
                )
