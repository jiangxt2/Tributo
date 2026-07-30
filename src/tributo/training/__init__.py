"""tributo.training — Distributed training and ONNX export public API.

JSON-driven training::

    from tributo.training import run_training_from_json
    run_training_from_json("config.json")

BaseTrainer subclass integration::

    from tributo.training import BaseTrainer, TrainerSpec, register

    class MyTrainer(BaseTrainer): ...

    register(TrainerSpec(name="my_algo", trainer_cls=MyTrainer))

Hyperparameter tuning::

    from tributo.training import TuneRunner, TuneSearchConfig, parse_search_space

    runner = TuneRunner(trainer_spec, tune_config, search_space)
    result_grid = runner.run(datasets={"train": ds}, output_path="/tmp/out")
"""

from __future__ import annotations

import logging

from tributo.exceptions import JobConfigurationError
from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    AlgorithmStatus,
    DataContract,
    DataLoadingMode,
    ProblemFamily,
    ProblemType,
    ResourceHints,
)
from tributo.training.base import BaseTrainer, TrainerSpec
from tributo.training.job_submitter import (
    submit_training_job,
    wait_for_job,
)
from tributo.training.onnx_exporter import export_to_onnx
from tributo.training.registry import get_trainer, list_trainers, register
from tributo.training.tune_config import TuneSearchConfig
from tributo.training.tune_runner import TuneRunner, extract_best_params
from tributo.training.tune_space import (
    SearchSpaceSpec,
    parse_search_space,
    resolve_local_overrides,
    to_ray_param_space,
    validate_search_targets,
    warn_search_space_conflicts,
)
from tributo.training.xgboost_trainer import (
    build_trainer,
    run_training_from_json,
)

logger = logging.getLogger(__name__)

# DNN trainer (lazy import, avoids requiring torch unconditionally)
try:
    from tributo.training.dnn_trainer import (
        DNNTrainerImpl,  # noqa: F401
        run_dnn_training_from_json,  # noqa: F401
        run_dnn_training_with_config,  # noqa: F401
    )

    _has_dnn = True
except ImportError:
    _has_dnn = False

# PU trainer (lazy import, requires torch)
try:
    from tributo.training.pu_trainer import (
        PUTrainerImpl,  # noqa: F401
        run_pu_training_from_json,  # noqa: F401
        run_pu_training_with_config,  # noqa: F401
    )

    _has_pu = True
except ImportError:
    _has_pu = False

# Class prior estimation (pure numpy, always available)
from tributo.training.priors import estimate_class_prior  # noqa: E402

__all__ = [
    # BaseTrainer abstraction
    "BaseTrainer",
    # AlgorithmSpec & supporting types
    "AlgorithmSpec",
    "AlgorithmStatus",
    "TrainerSpec",
    "DataContract",
    "DataLoadingMode",
    "ProblemFamily",
    "ProblemType",
    "ResourceHints",
    # Registry
    "register",
    "get_trainer",
    "list_trainers",
    # XGBoost training
    "build_trainer",
    "run_training_from_json",
    # Hyperparameter tuning
    "TuneSearchConfig",
    "TuneRunner",
    "parse_search_space",
    "SearchSpaceSpec",
    "extract_best_params",
    "resolve_local_overrides",
    "to_ray_param_space",
    "validate_search_targets",
    "warn_search_space_conflicts",
    # Utilities
    "export_to_onnx",
    "submit_training_job",
    "wait_for_job",
    # Class prior estimation
    "estimate_class_prior",
]

# Dynamically export DNN-related symbols (if torch is available)
if _has_dnn:
    __all__.extend(
        [
            "DNNTrainerImpl",
            "run_dnn_training_from_json",
            "run_dnn_training_with_config",
        ]
    )

# Dynamically export PU-related symbols (if torch is available)
if _has_pu:
    __all__.extend(
        [
            "PUTrainerImpl",
            "run_pu_training_from_json",
            "run_pu_training_with_config",
        ]
    )

# Auto-discover third-party trainer plugins via entry_points
from tributo.plugin import discover_trainer_plugins  # noqa: E402

for _ep_spec in discover_trainer_plugins():
    from tributo.training.registry import register as _reg

    try:
        _reg(_ep_spec)
    except JobConfigurationError:
        logger.debug(
            "Trainer %r from plugin already registered; skipping.",
            _ep_spec.name,
        )

# Phase 3 integrity gate: validate replacement graph after all plugins loaded.
from tributo.training.catalog import get_algorithm_catalog as _get_catalog  # noqa: E402

try:
    _get_catalog().validate_integrity()
except JobConfigurationError:
    logger.exception("Algorithm catalog integrity check failed")
    raise
