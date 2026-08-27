"""Tributo training public API.

Framework-specific first-party trainers are loaded on demand so importing this
package does not bootstrap Trainer implementations or optional dependencies.
"""

from __future__ import annotations

import importlib
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
    from tributo.training.job_submitter import (
        JobAttempt,
        TrainingJobResult,
        submit_training_job,
        submit_training_job_with_identity,
        submit_training_job_with_retry,
        wait_for_job,
    )
    from tributo.training.portable_tune import PortableTuneRunner
    from tributo.training.registry import get_trainer, list_trainers, register
    from tributo.training.tune_runner import TuneRunner, extract_best_params

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
    "submit_training_job_with_identity": (
        "tributo.training.job_submitter",
        "submit_training_job_with_identity",
    ),
    "wait_for_job": ("tributo.training.job_submitter", "wait_for_job"),
    "TuneRunner": ("tributo.training.tune_runner", "TuneRunner"),
    "PortableTuneRunner": (
        "tributo.training.portable_tune",
        "PortableTuneRunner",
    ),
    "extract_best_params": (
        "tributo.training.tune_runner",
        "extract_best_params",
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
    "submit_training_job_with_identity",
    "submit_training_job_with_retry",
    "wait_for_job",
    "TuneSearchConfig",
    "TuneRunner",
    "PortableTuneRunner",
    "extract_best_params",
    "SearchSpaceSpec",
    "parse_search_space",
    "resolve_local_overrides",
    "to_ray_param_space",
    "validate_search_targets",
    "warn_search_space_conflicts",
]

__all__ += [
    "BaseCausalEstimator",
    "CausalEffect",
    "CausalGraph",
    "RefutationResult",
]
