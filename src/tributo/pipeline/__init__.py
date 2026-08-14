"""tributo.pipeline — Multi-step training workflow orchestration.

Provides a lightweight in-process DAG executor for composing training
steps (e.g. Encoder → Graph Features → Classifier for user profiling).

Usage::

    from tributo.pipeline import Pipeline, PipelineStep, ArtifactSpec, ArtifactRef, InputBinding
"""

from __future__ import annotations

from tributo.pipeline.core import (
    ArtifactRef,
    ArtifactSpec,
    InputBinding,
    Pipeline,
    PipelineStep,
)

__all__ = [
    "ArtifactRef",
    "ArtifactSpec",
    "InputBinding",
    "Pipeline",
    "PipelineStep",
]
