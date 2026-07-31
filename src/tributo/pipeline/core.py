"""Pipeline orchestration core.

Lightweight in-process DAG executor inspired by Kedro's Node/Pipeline
pattern.  Each ``PipelineStep`` references an algorithm from the trainer
registry and declares its input bindings and output specifications.
``Pipeline.validate()`` statically checks the DAG for cycles, dangling
references, and type compatibility before execution.

Current scope: process-local, fail-fast, unrecoverable — suitable for
internal multi-step workflows (e.g. user profiling).  Retry, timeout,
checkpoint, and caching are deferred to a future version.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from tributo._common.dag import topological_order
from tributo.exceptions import JobConfigurationError
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


# ── Pipeline data types ──────────────────────────────────────────────────────


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class ArtifactSpec:
    """Artifact type declaration — describes kind and schema.

    Type information is declared once at the *producer* step's
    ``outputs``.  Consumers declare their *expectations* via
    ``InputBinding.expected``, and ``Pipeline.validate()`` checks
    compatibility between the two.

    Attributes:
        artifact_kind: One of ``"model"``, ``"dataset"``, ``"features"``,
            ``"report"``.
        schema: Optional column / field schema for compatibility checks.
    """

    artifact_kind: str
    schema: dict[str, Any] | None = None


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class ArtifactRef:
    """Pure locator for a step output — no type information.

    ``ArtifactRef`` only identifies *where* an artifact comes from
    (step name + output port).  All type information lives on the
    producer's ``outputs`` declaration (single source of truth).

    Attributes:
        producer_step: Name of the step that produces the artifact.
        output_port: Output port name within the producer step.
    """

    producer_step: str
    output_port: str


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class InputBinding:
    """Consumer-side input binding — source reference + expected type.

    During validation, the framework resolves the producer's actual
    ``ArtifactSpec`` from ``outputs[output_port]`` and checks it
    against ``expected``.

    Attributes:
        source: Reference to the upstream step's output.
        expected: The consumer's expected artifact type (kind + optional
            schema).  Validation passes when the producer spec satisfies
            these constraints.
    """

    source: ArtifactRef
    expected: ArtifactSpec


# ── PipelineStep ─────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
@dataclass
class PipelineStep:
    """A single step in a training pipeline.

    Attributes:
        name: Unique step name within the pipeline.
        algorithm: Algorithm name from the trainer registry.
        config: Step-specific configuration dict.
        inputs: Input port → ``InputBinding`` (source + expected type).
        outputs: Output port → ``ArtifactSpec`` (producer declaration).
    """

    name: str
    algorithm: str
    config: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, InputBinding] = field(default_factory=dict)
    outputs: dict[str, ArtifactSpec] = field(default_factory=dict)


# ── Pipeline ─────────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class Pipeline:
    """In-process DAG executor for multi-step training workflows.

    Steps are executed in topological order.  ``validate()`` performs
    static checks:

    * No cycles (topological sort must succeed).
    * Referential integrity (every ``ArtifactRef`` points to an existing
      step + output port).
    * Type compatibility (every consumer's ``expected`` spec is satisfied
      by the producer's declared spec).
    """

    def __init__(self, steps: list[PipelineStep]) -> None:
        self.steps: list[PipelineStep] = steps
        self._step_map: dict[str, PipelineStep] = {s.name: s for s in steps}

    def validate(self) -> None:
        """Validate the pipeline DAG statically.

        Raises:
            JobConfigurationError: If the pipeline is invalid.
        """
        if not self.steps:
            raise JobConfigurationError("Pipeline must have at least one step.")

        # Check unique step names via Counter (O(n)).
        name_counts = Counter(s.name for s in self.steps)
        duplicates = [name for name, count in name_counts.items() if count > 1]
        if duplicates:
            raise JobConfigurationError(f"Duplicate step names: {sorted(duplicates)}")

        # Check cycles via topological sort (Kahn's algorithm).
        self._check_cycles()

        # Check referential integrity and type compatibility.
        for step in self.steps:
            for port_name, binding in step.inputs.items():
                producer_name = binding.source.producer_step
                output_port = binding.source.output_port

                # 1. Referential integrity: producer step exists.
                if producer_name not in self._step_map:
                    raise JobConfigurationError(
                        f"Step {step.name!r}: input {port_name!r} references "
                        f"unknown producer step {producer_name!r}."
                    )

                producer = self._step_map[producer_name]

                # 2. Referential integrity: output port exists on producer.
                if output_port not in producer.outputs:
                    raise JobConfigurationError(
                        f"Step {step.name!r}: input {port_name!r} references "
                        f"unknown output port {output_port!r} on step "
                        f"{producer_name!r}."
                    )

                # 3. Type compatibility: producer spec must satisfy consumer
                #    expected spec (same artifact_kind, compatible schema).
                actual = producer.outputs[output_port]
                expected = binding.expected
                self._check_type_compat(
                    step.name, port_name, producer_name, actual, expected
                )

    # -- DAG helpers (delegate to shared kernel) -------------------------------

    def _build_adjacency(self) -> dict[str, list[str]]:
        """Build adjacency dict ``{step_name: [upstream_step_names]}``."""
        return {
            s.name: [b.source.producer_step for b in s.inputs.values()]
            for s in self.steps
        }

    def _check_cycles(self) -> None:
        """Validate that the pipeline DAG is acyclic via shared kernel."""
        try:
            topological_order(self._build_adjacency())
        except ValueError as exc:
            raise JobConfigurationError(str(exc)) from exc

    @staticmethod
    def _check_type_compat(
        consumer_name: str,
        port_name: str,
        producer_name: str,
        actual: ArtifactSpec,
        expected: ArtifactSpec,
    ) -> None:
        """Validate that *actual* (producer) satisfies *expected* (consumer)."""
        if actual.artifact_kind != expected.artifact_kind:
            raise JobConfigurationError(
                f"Type mismatch on {consumer_name!r}.{port_name!r}: "
                f"producer {producer_name!r} declares artifact_kind="
                f"{actual.artifact_kind!r} but consumer expects "
                f"{expected.artifact_kind!r}."
            )
        # Schema compatibility check: consumer's expected schema keys must
        # each be present in the producer's actual schema with matching types.
        if expected.schema is not None:
            if actual.schema is None:
                raise JobConfigurationError(
                    f"Schema mismatch on {consumer_name!r}.{port_name!r}: "
                    f"producer {producer_name!r} does not declare a schema "
                    f"but consumer expects schema keys "
                    f"{sorted(expected.schema)}."
                )
            for key, expected_type in expected.schema.items():
                actual_type = actual.schema.get(key)
                if actual_type is None:
                    raise JobConfigurationError(
                        f"Schema mismatch on {consumer_name!r}.{port_name!r}: "
                        f"producer {producer_name!r} does not provide field "
                        f"{key!r} (expected type {expected_type!r})."
                    )
                if actual_type != expected_type:
                    raise JobConfigurationError(
                        f"Schema type mismatch on {consumer_name!r}.{port_name!r}."
                        f"{key!r}: producer {producer_name!r} declares "
                        f"{actual_type!r} but consumer expects "
                        f"{expected_type!r}."
                    )

    def run(self, initial_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the pipeline in topological order.

        Each step's algorithm is loaded from the trainer registry and
        executed with the upstream step outputs as inputs.  All steps
        run in-process (no distributed execution).

        Pipeline-step trainers receive upstream data via
        ``self.config["_pipeline_inputs"]``, a dict mapping input port
        names to the resolved upstream outputs (or ``initial_data`` for
        root steps).  Subclasses should read this key in ``setup()``
        or ``training_loop()`` rather than relying on ``self.datasets``,
        which the pipeline always passes as an empty dict.

        Args:
            initial_data: Initial data dict passed to root steps
                (steps with no upstream inputs).

        Returns:
            A dict mapping ``{step_name: step_output}`` for every step.

        Raises:
            JobConfigurationError: If validation fails.
            JobExecutionError: If a step fails during execution.
        """
        self.validate()

        # Topological order via shared DAG kernel.
        order = topological_order(self._build_adjacency())

        # Execute in topological order.
        outputs: dict[str, dict[str, Any]] = {}
        for step_name in order:
            step = self._step_map[step_name]
            logger.info(
                "Running pipeline step %r (algorithm=%r)", step_name, step.algorithm
            )

            # Resolve inputs: root steps get initial_data, others get
            # upstream outputs via InputBinding references.
            resolved_inputs: dict[str, Any] = {}
            if step.inputs:
                for port_name, binding in step.inputs.items():
                    upstream_outputs = outputs.get(binding.source.producer_step, {})
                    resolved_inputs[port_name] = upstream_outputs.get(
                        binding.source.output_port
                    )
            else:
                resolved_inputs = dict(initial_data)

            # Run the algorithm.  Inject resolved_inputs into a copy of
            # the step config BEFORE constructing the trainer so that
            # config validation / dataset loading sees the injected key.
            from tributo.training.registry import get_trainer

            spec = get_trainer(step.algorithm)
            step_config = dict(step.config)
            step_config["_pipeline_inputs"] = resolved_inputs
            trainer_cls = spec.trainer_cls
            trainer = trainer_cls(
                datasets={},  # pipeline steps receive resolved inputs
                config=step_config,
            )

            trainer.setup()
            result = trainer.training_loop()

            # Publish step outputs.
            step_outputs: dict[str, Any] = {}
            for output_port in step.outputs:
                # Extract the named output from the result dict.
                step_outputs[output_port] = (
                    result[output_port]
                    if isinstance(result, dict) and output_port in result
                    else result
                )
            outputs[step_name] = step_outputs

        return outputs
