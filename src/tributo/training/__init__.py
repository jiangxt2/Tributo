"""Tributo training public API.

Framework-specific first-party trainers are loaded on demand so importing this
package does not bootstrap Trainer implementations or optional dependencies.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING, Any

from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    AlgorithmStatus,
    Capability,
    DataContract,
    DataLoadingMode,
    ExecutionKind,
    ProblemFamily,
    ProblemType,
    ResourceHints,
)
from tributo.training.base import BaseTrainer, TrainerSpec
from tributo.training.graph_trainer import BaseGraphTrainer
from tributo.training.onnx_exporter import export_to_onnx
from tributo.training.priors import estimate_class_prior
from tributo.training.results import (
    BundleStatus,
    TrainingHookStatus,
    TrainingResult,
    TrainingStatus,
)
from tributo.training.tune_config import TuneSearchConfig
from tributo.training.tune_space import (
    SearchSpaceSpec,
    parse_search_space,
    resolve_local_overrides,
    to_ray_param_space,
    validate_search_targets,
    warn_search_space_conflicts,
)

if TYPE_CHECKING:
    from tributo.training.causal_estimator import (
        BaseCausalEstimator,
        CausalEffect,
        CausalGraph,
        RefutationResult,
    )
    from tributo.training.dnn_trainer import (
        DNNTrainerImpl,
        run_dnn_training_from_json,
        run_dnn_training_with_config,
    )
    from tributo.training.job_submitter import (
        JobAttempt,
        TrainingJobResult,
        submit_training_job,
        submit_training_job_with_retry,
        wait_for_job,
    )
    from tributo.training.pu_trainer import (
        PUTrainerImpl,
        run_pu_training_from_json,
        run_pu_training_with_config,
    )
    from tributo.training.registry import get_trainer, list_trainers, register
    from tributo.training.tune_runner import TuneRunner, extract_best_params
    from tributo.training.xgboost_trainer import build_trainer, run_training_from_json

_LAZY_EXPORTS = {
    "get_trainer": ("tributo.training.registry", "get_trainer"),
    "list_trainers": ("tributo.training.registry", "list_trainers"),
    "register": ("tributo.training.registry", "register"),
    "JobAttempt": ("tributo.training.job_submitter", "JobAttempt"),
    "TrainingJobResult": (
        "tributo.training.job_submitter",
        "TrainingJobResult",
    ),
    "submit_training_job": (
        "tributo.training.job_submitter",
        "submit_training_job",
    ),
    "submit_training_job_with_retry": (
        "tributo.training.job_submitter",
        "submit_training_job_with_retry",
    ),
    "wait_for_job": ("tributo.training.job_submitter", "wait_for_job"),
    "TuneRunner": ("tributo.training.tune_runner", "TuneRunner"),
    "extract_best_params": (
        "tributo.training.tune_runner",
        "extract_best_params",
    ),
    "build_trainer": ("tributo.training.xgboost_trainer", "build_trainer"),
    "run_training_from_json": (
        "tributo.training.xgboost_trainer",
        "run_training_from_json",
    ),
    "DNNTrainerImpl": ("tributo.training.dnn_trainer", "DNNTrainerImpl"),
    "run_dnn_training_from_json": (
        "tributo.training.dnn_trainer",
        "run_dnn_training_from_json",
    ),
    "run_dnn_training_with_config": (
        "tributo.training.dnn_trainer",
        "run_dnn_training_with_config",
    ),
    "PUTrainerImpl": ("tributo.training.pu_trainer", "PUTrainerImpl"),
    "run_pu_training_from_json": (
        "tributo.training.pu_trainer",
        "run_pu_training_from_json",
    ),
    "run_pu_training_with_config": (
        "tributo.training.pu_trainer",
        "run_pu_training_with_config",
    ),
    "BaseCausalEstimator": (
        "tributo.training.causal_estimator",
        "BaseCausalEstimator",
    ),
    "CausalEffect": ("tributo.training.causal_estimator", "CausalEffect"),
    "CausalGraph": ("tributo.training.causal_estimator", "CausalGraph"),
    "RefutationResult": (
        "tributo.training.causal_estimator",
        "RefutationResult",
    ),
}


def __getattr__(name: str) -> Any:
    """Load optional or framework-specific public symbols on first access."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "BundleStatus",
    "TrainingHookStatus",
    "TrainingResult",
    "TrainingStatus",
    "AlgorithmSpec",
    "AlgorithmStatus",
    "BaseTrainer",
    "Capability",
    "DataContract",
    "DataLoadingMode",
    "ExecutionKind",
    "ProblemFamily",
    "ProblemType",
    "ResourceHints",
    "TrainerSpec",
    "get_trainer",
    "list_trainers",
    "register",
    "BaseGraphTrainer",
    "JobAttempt",
    "TrainingJobResult",
    "submit_training_job",
    "submit_training_job_with_retry",
    "wait_for_job",
    "export_to_onnx",
    "estimate_class_prior",
    "TuneSearchConfig",
    "TuneRunner",
    "extract_best_params",
    "SearchSpaceSpec",
    "parse_search_space",
    "resolve_local_overrides",
    "to_ray_param_space",
    "validate_search_targets",
    "warn_search_space_conflicts",
    "build_trainer",
    "run_training_from_json",
]

if importlib.util.find_spec("torch") is not None:
    __all__ += [
        "DNNTrainerImpl",
        "PUTrainerImpl",
        "run_dnn_training_from_json",
        "run_dnn_training_with_config",
        "run_pu_training_from_json",
        "run_pu_training_with_config",
    ]

if (
    importlib.util.find_spec("dowhy") is not None
    and importlib.util.find_spec("econml") is not None
):
    __all__ += [
        "BaseCausalEstimator",
        "CausalEffect",
        "CausalGraph",
        "RefutationResult",
    ]
