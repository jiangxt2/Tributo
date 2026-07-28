"""Ray Tune hyperparameter search configuration.

TuneSearchConfig controls the search process behavior (algorithm, scheduler, sampling count, etc.),
which is separate from the training configuration (hyperparameters themselves).

Naming note: uses TuneSearchConfig instead of TuneConfig to avoid
naming conflicts with Ray's ray.tune.TuneConfig.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tributo.util.annotations import PublicAPI


@PublicAPI(stability="beta")
class TuneSearchConfig(BaseModel):
    """Ray Tune hyperparameter search configuration.

    Controls the search process behavior (algorithm, scheduler, sampling count, etc.),
    which is separate from the training configuration (hyperparameters themselves).

    Attributes:
        metric: Name of the optimization target metric.
        mode: Optimization direction, "min" or "max".
        num_samples: Number of samples (repeated experiments per config).
        max_concurrent_trials: Maximum concurrent trials, None means unlimited.
        time_budget_s: Global time budget in seconds, None means unlimited.
        search_alg: Search algorithm, supports "random" and "bayesopt".
        scheduler: Scheduler, supports "fifo", "asha" and "hyperband".
        fail_fast: Whether to abort the experiment immediately on first trial failure.
    """

    model_config = ConfigDict(extra="forbid")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate mode is 'min' or 'max'."""
        if v not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{v}'")
        return v

    @field_validator("search_alg")
    @classmethod
    def validate_search_alg(cls, v: str) -> str:
        """Validate search algorithm name."""
        supported = ("random", "bayesopt")
        if v not in supported:
            raise ValueError(f"search_alg must be one of {supported}, got '{v}'")
        return v

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler(cls, v: str) -> str:
        """Validate scheduler name."""
        supported = ("fifo", "asha", "hyperband")
        if v not in supported:
            raise ValueError(f"scheduler must be one of {supported}, got '{v}'")
        return v

    metric: str = Field(
        default="loss",
        description="Optimization target metric",
    )
    mode: str = Field(
        default="min",
        description="Optimization direction: min or max",
    )
    num_samples: int = Field(
        default=1,
        ge=1,
        description="Number of samples",
    )
    max_concurrent_trials: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of concurrent trials",
    )
    time_budget_s: float | None = Field(
        default=None,
        gt=0,
        description="Global time budget in seconds",
    )
    search_alg: str = Field(
        default="random",
        description="Search algorithm: random / bayesopt",
    )
    scheduler: str = Field(
        default="fifo",
        description="Scheduler: fifo / asha / hyperband",
    )
    fail_fast: bool = Field(
        default=False,
        description="Abort immediately on first trial failure",
    )
