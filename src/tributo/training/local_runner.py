"""Local training runner — single trial without a Ray cluster.

Useful for development, debugging, and small-scale experiments where
setting up a Ray cluster would be overkill.  Ray is still required as a
library (``ray.data`` for local data loading), but no cluster or head
node is needed.

Canonical path::

    from tributo.training.local_runner import run_local_trial
    from tributo.training.config import build_effective_config

    spec = get_trainer("xgboost")
    effective = build_effective_config(spec, raw_user_config)
    summary = run_local_trial(spec, effective_config=effective, output_path="./out")

Legacy signature (still supported)::

    run_local_trial(spec, training_config=raw_dict, output_path="./out")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    DataLoadingMode,
    require_legacy_trainer_cls,
)
from tributo.training.config import (
    resolve_data_source,
    validate_and_normalize_config,
    validate_execution_config,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
def run_local_trial(
    trainer_spec: AlgorithmSpec,
    output_path: str,
    *,
    effective_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a single training trial locally without a Ray cluster.

    Accepts either:

    * ``effective_config`` — a fully-merged, validated config dict
      (preferred canonical path), or
    * ``training_config`` — raw user config (legacy path; the runner
      merges ``default_config`` internally).

    Args:
        trainer_spec: An ``AlgorithmSpec`` from the registry.
        output_path: Directory for exported model artifacts.
        effective_config: Pre-validated canonical config.
        training_config: Raw user config dict (legacy; must contain
            ``data`` section for driver-mode algorithms).

    Returns:
        A dict with ``status``, ``duration_sec``, and additional fields
        populated by the trainer's ``export_model`` via ``self._summary``.

    Raises:
        JobConfigurationError: Config is missing data source.
        FileNotFoundError: Data file not found.
    """
    # -- Resolve config -------------------------------------------------------
    if effective_config is not None:
        config = effective_config
    elif training_config is not None:
        # Legacy path: shallow merge (not recursive).  Only top-level keys from
        # *training_config* replace those in *default_config*; nested sections
        # are replaced wholesale, not deeply merged.  Prefer the canonical
        # ``effective_config`` path for recursive merge semantics.
        config = {**trainer_spec.default_config, **training_config}
    else:
        raise ValueError(
            "run_local_trial requires either effective_config or training_config"
        )

    # Re-validate idempotently (protects programmatic callers).
    config = validate_and_normalize_config(trainer_spec, config)
    validate_execution_config(trainer_spec, config, datasets_supplied=False)

    # -- Load data ------------------------------------------------------------
    datasets: dict[str, Any] = {}
    if trainer_spec.data_loading != DataLoadingMode.CANONICAL_TRAINER:
        source = resolve_data_source(trainer_spec, config)
        from tributo.training.data_loader import load_ray_dataset_from_source

        train_ds = load_ray_dataset_from_source(source)
        datasets["train"] = train_ds

        data_cfg = config.get("data", {})
        if isinstance(data_cfg, dict) and data_cfg.get("val_path"):
            # ``val_path``/``test_path`` are legacy local-runner fields whose
            # published behavior is CWD-relative. Canonical source objects use
            # the normal project-root policy instead.
            datasets["val"] = load_ray_dataset_from_source(
                {"type": "parquet", "path": data_cfg["val_path"]},
                project_root_path=Path.cwd(),
            )
        if isinstance(data_cfg, dict) and data_cfg.get("test_path"):
            datasets["test"] = load_ray_dataset_from_source(
                {"type": "parquet", "path": data_cfg["test_path"]},
                project_root_path=Path.cwd(),
            )

    # -- Run training ---------------------------------------------------------
    logger.info("Running local trial: trainer=%s", trainer_spec.name)
    t0 = time.monotonic()

    trainer_cls = require_legacy_trainer_cls(
        trainer_spec,
        consumer="local runner",
    )
    trainer = trainer_cls(datasets=datasets, config=config)
    summary = trainer.run(output_path=output_path)

    duration = time.monotonic() - t0
    summary["duration_sec"] = round(duration, 1)
    logger.info(
        "Local trial completed in %.1fs (status=%s)",
        duration,
        summary.get("status", "unknown"),
    )

    return summary
