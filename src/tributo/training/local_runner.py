"""Local training runner — single trial without a Ray cluster.

Useful for development, debugging, and small-scale experiments where
setting up a Ray cluster would be overkill.  Ray is still required as a
library (``ray.data`` for local data loading), but no cluster or head
node is needed.

Usage via CLI::

    tributo tune run --trainer xgboost --config train.json \
        --space search.json --output ./out --local

Programmatic::

    from tributo.training.local_runner import run_local_trial
    from tributo.training.registry import get_trainer

    trainer_spec = get_trainer("xgboost")
    summary = run_local_trial(
        trainer_spec=trainer_spec,
        training_config={"data": {"train_path": "data.parquet"}, ...},
        output_path="./output",
    )
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
def run_local_trial(
    trainer_spec: Any,  # TrainerSpec (lazy import to avoid hard dependency)
    training_config: dict[str, Any],
    output_path: str,
    search_space: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a single training trial locally without a Ray cluster.

    Loads data via ``ray.data.read_parquet`` (works locally with no
    cluster), instantiates the trainer, and calls the template method
    :meth:`BaseTrainer.run`.

    Args:
        trainer_spec: A ``TrainerSpec`` from ``tributo.training.registry``.
        training_config: Training configuration dict.  Must contain a
            ``data`` section with ``train_path`` pointing to a Parquet file.
        output_path: Directory for exported model artifacts.
        search_space: Optional parameter overrides merged into the config.

    Returns:
        A dict with ``status``, ``duration_sec``, and additional fields
        populated by the trainer's ``export_model`` via ``self._summary``.

    Raises:
        FileNotFoundError: If ``train_path`` does not exist.
        ValueError: If the training config is missing ``data.train_path``.
    """
    import ray.data

    search_space = search_space or {}
    trainer_cls = trainer_spec.trainer_cls

    # Build config from defaults + training_config + search_space overrides
    config = {
        **trainer_spec.default_config,
        **training_config,
        **search_space,
    }

    data_cfg = config.get("data", {})
    train_path = data_cfg.get("train_path") or data_cfg.get("path")
    if not train_path:
        raise ValueError(
            "Local mode requires 'data.train_path' or 'data.path' "
            "pointing to a Parquet file."
        )

    if not Path(train_path).exists() and "://" not in str(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")

    logger.info(
        "Running local trial: trainer=%s, data=%s", trainer_spec.name, train_path
    )
    t0 = time.monotonic()

    # Load data via ray.data (local, no cluster required)
    train_ds = ray.data.read_parquet(train_path)
    datasets: dict[str, ray.data.Dataset] = {"train": train_ds}
    if data_cfg.get("val_path"):
        datasets["val"] = ray.data.read_parquet(data_cfg["val_path"])
    if data_cfg.get("test_path"):
        datasets["test"] = ray.data.read_parquet(data_cfg["test_path"])

    # Instantiate trainer and run the template method
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
