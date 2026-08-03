"""Bundle model runtime — shared bridge for Serving / Batch Inference.

The stable consumption entry point is a ``bundle_uri`` plus an explicit
``role``; raw model paths are not part of this module — they remain a
compat adapter owned by individual consumers.  Loading always routes
through the flavor registry keyed by ``artifact.flavor_id``:

.. code-block:: text

    BundleReader
        → role
        → LogicalArtifact
        → artifact.flavor_id
        → FlavorRegistry
        → BundleModelRuntime
        → Model Signature validation
        → predict

``BundleModelLoader`` validates role, flavor, dependencies, and the
manifest signature *before* any model file is opened; ``BundleModelRuntime``
holds the BundleReader's artifact context (ExitStack) so S3 temp files
are never removed before the model finished loading, and closes them
idempotently.
"""

from __future__ import annotations

import importlib.util
import logging
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from tributo.exceptions import (
    JobConfigurationError,
    ModelLoadError,
    ModelSchemaMismatchError,
    UnsupportedArtifactFormat,
)
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.manifest import ExportManifest, SignatureField
from tributo.exporting.models import LogicalArtifact, ResolvedArtifact
from tributo.exporting.registries import FlavorRegistry
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

#: Security modes a flavor may declare.  Anything other than ``safe``
#: requires ``unsafe=True`` to load — the framework refuses to execute
#: pickle payloads or unknown code paths by default.
SECURITY_MODE_SAFE = "safe"
SECURITY_MODE_PICKLE = "pickle"
SECURITY_MODE_REMOTE_CODE = "remote-code"
SECURITY_MODE_UNKNOWN = "unknown-executable"

#: The only role default allowed to be implicit — serving entry points
#: must otherwise pass an explicit role.
DEFAULT_ROLE = "inference"


# ── Protocols ─────────────────────────────────────────────────────────────────


@runtime_checkable
@PublicAPI(stability="beta")
class BundleReaderLike(Protocol):
    """Structural reader contract — satisfied by ``BundleReader``.

    Keeps the loader decoupled from the concrete reader so tests can
    inject recording or in-memory readers.
    """

    def read_manifest(
        self, manifest_or_bundle_uri: str, *, storage_profile: str | None = None
    ) -> ExportManifest: ...

    def open_artifact(
        self,
        manifest_or_bundle_uri: str,
        *,
        role: str | None = None,
        artifact_name: str | None = None,
        storage_profile: str | None = None,
    ) -> Any: ...


@runtime_checkable
@PublicAPI(stability="beta")
class BundleModel(Protocol):
    """A loaded, in-memory model ready for prediction.

    After loading, the model must not depend on the bundle's temporary
    files — the runtime closes the artifact context as soon as loading
    completes, so ``predict`` must work purely in memory.
    """

    @property
    def input_names(self) -> tuple[str, ...]:
        """Model input names, in the order the runtime expects them."""
        ...

    @property
    def output_names(self) -> tuple[str, ...]:
        """Model output names, in execution order (unnamed outputs fall back
        to ``output_0``-style placeholders)."""
        ...

    @property
    def input_dtypes(self) -> tuple[str, ...]:
        """Framework-neutral dtypes of the inputs (e.g. ``"float32"``)."""
        ...

    @property
    def output_dtypes(self) -> tuple[str, ...]:
        """Framework-neutral dtypes of the outputs."""
        ...

    @property
    def input_shapes(self) -> tuple[tuple[int | None, ...], ...]:
        """Input shapes; ``None`` marks a dynamic dimension."""
        ...

    @property
    def output_shapes(self) -> tuple[tuple[int | None, ...], ...]:
        """Output shapes; ``None`` marks a dynamic dimension."""
        ...

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference on named inputs and return named outputs."""
        ...


@runtime_checkable
@PublicAPI(stability="beta")
class BundleModelFlavor(Protocol):
    """Loads a ``BundleModel`` from a verified bundle artifact.

    Class variables declare the flavor contract consumed by the loader:

    - ``api_version``: 1 for the first-generation protocol.
    - ``flavor_id``: Stable routing key (e.g. ``"onnx-runtime-v1"``).
    - ``security_mode``: One of the ``SECURITY_MODE_*`` constants.
      Anything other than ``safe`` requires ``unsafe=True``.
    - ``signature_required``: ``True`` when the flavor needs a non-empty
      typed manifest signature before it may serve.
    - ``required_dependencies``: Import names pre-checked by the loader
      so missing dependencies fail fast with an install hint.
    """

    api_version: ClassVar[int]
    flavor_id: ClassVar[str]
    security_mode: ClassVar[str]
    signature_required: ClassVar[bool]
    required_dependencies: ClassVar[tuple[str, ...]]

    def load(
        self,
        artifact: ResolvedArtifact,
        *,
        role: str,
        unsafe: bool = False,
        architecture_id: str | None = None,
    ) -> BundleModel:
        """Build a model from *artifact* (its files are locally verified).

        *architecture_id* comes from ``manifest.source_info`` — flavors
        that need to rebuild the model skeleton use it to resolve the
        ``ModelFactoryRegistry``.  Must complete before the artifact
        context is closed.
        """
        ...


# ── Serveable flavor support matrix ───────────────────────────────────────────


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class FlavorSupportEntry:
    """One row of the serveable flavor support matrix.

    Every entry declares what is required to serve a flavor so that
    consumers get a deterministic answer instead of a runtime surprise.
    ``loader`` is the import path of the loader class (same format as
    entry points), resolved lazily — the actual routing key is always
    the registry.
    """

    flavor_id: str
    artifact_role: str
    loader: str
    dependencies: tuple[str, ...]
    signature_required: bool
    security_mode: str


#: Frozen serveable flavor support matrix (E3).  Covers the primary
#: artifacts of the walking skeleton (XGBoost → ONNX) and of the
#: DNN/PU/XGBoost bundle paths, all of which publish ``onnx-runtime-v1``.
#: Other flavor IDs produced by exporters (``safetensors-v1``,
#: ``torch-export-v1``, ``xgboost-native-v1``, ``hf-onnx-v1``,
#: ``onnx-int8-v1``) are explicitly unsupported until their loaders are
#: registered here — they are never loaded by guessing.
SERVEABLE_FLAVOR_MATRIX: tuple[FlavorSupportEntry, ...] = (
    FlavorSupportEntry(
        flavor_id="onnx-runtime-v1",
        artifact_role="model",
        loader="tributo.integrations.flavors.onnx_runtime:ONNXRuntimeFlavor",
        dependencies=("onnxruntime",),
        signature_required=True,
        security_mode=SECURITY_MODE_SAFE,
    ),
)


# ── Loader ────────────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class BundleModelLoader:
    """Opens a published bundle as a serveable model runtime.

    Args:
        bundle_reader: Reader for manifest + artifacts; defaults to a
            fresh ``BundleReader``.
        flavor_registry: Registry keyed by ``flavor_id``; defaults to a
            registry populated with built-in flavors and entry-point
            plugins.
    """

    def __init__(
        self,
        *,
        bundle_reader: BundleReaderLike | None = None,
        flavor_registry: FlavorRegistry | None = None,
    ) -> None:
        self._reader = bundle_reader or BundleReader()
        self._flavors = flavor_registry or _build_flavor_registry()

    def open(
        self,
        bundle_uri: str,
        *,
        role: str = DEFAULT_ROLE,
        storage_profile: str | None = None,
        unsafe: bool = False,
    ) -> "BundleModelRuntime":
        """Open *bundle_uri* and load the model for *role*.

        Args:
            bundle_uri: Bundle manifest URI (local path or ``s3://``).
            role: Artifact role to serve; defaults to ``"inference"``
                (the only allowed implicit default).
            storage_profile: Storage profile name for S3 credentials.
            unsafe: Permit flavors whose security mode is not ``safe``
                and manifests without a typed signature.

        Returns:
            A ``BundleModelRuntime`` holding the loaded model.

        Raises:
            JobConfigurationError: Unknown role, unknown flavor, or
                invalid bundle URI.
            UnsupportedArtifactFormat: Flavor not in the serveable
                matrix, or the manifest lacks a typed signature.
            ModelLoadError: A required dependency is missing or the
                model file could not be loaded.
        """
        manifest = self._reader.read_manifest(
            bundle_uri, storage_profile=storage_profile
        )

        # Explicit role → artifact.
        target_name = manifest.roles.get(role)
        if target_name is None:
            raise JobConfigurationError(
                f"Role {role!r} not found in bundle. Available roles: "
                f"{sorted(manifest.roles)}"
            )
        artifact = _find_artifact(manifest, target_name)

        # flavor_id → loader (registry lookup is the single routing key).
        try:
            flavor_cls: type[BundleModelFlavor] = self._flavors.get(artifact.flavor_id)
        except JobConfigurationError as exc:
            raise JobConfigurationError(
                f"Artifact {target_name!r} has flavor {artifact.flavor_id!r} "
                f"but no loader is registered. Available flavors: "
                f"{self._flavors.list_all()}"
            ) from exc

        # Serveable matrix gate — a registered flavor may still be
        # explicitly unsupported for serving.  The matrix row also
        # declares which artifact kind it serves (e.g. ``model``) —
        # a report or auxiliary artifact must never reach a model
        # loader, regardless of its flavor_id.
        entry = _matrix_entry(artifact.flavor_id)
        if artifact.artifact_kind != entry.artifact_role:
            raise UnsupportedArtifactFormat(
                f"Artifact {artifact.name!r} has kind {artifact.artifact_kind!r} "
                f"but flavor {entry.flavor_id!r} serves role {entry.artifact_role!r}"
            )

        # Security gate: the loader's declared mode is what actually
        # executes; non-safe modes are refused unless explicitly enabled.
        if flavor_cls.security_mode != SECURITY_MODE_SAFE and not unsafe:
            raise UnsupportedArtifactFormat(
                f"Flavor {artifact.flavor_id!r} runs {flavor_cls.security_mode!r} "
                "code — refusing to load without unsafe=True"
            )

        # Dependency pre-check with an install hint.
        _check_dependencies(artifact.flavor_id, flavor_cls.required_dependencies)

        # Signature gate: serveable roles need non-empty typed signatures.
        if entry.signature_required:
            _require_typed_signature(manifest, artifact, unsafe)

        # Load the model inside the artifact context; the runtime keeps
        # the context open (ExitStack) so temp files survive lazy-loaders.
        # Signature validation happens inside the try so a mismatch still
        # closes the artifact context (no resource leak on rejection).
        stack = ExitStack()
        try:
            resolved = stack.enter_context(
                self._reader.open_artifact(
                    bundle_uri, role=role, storage_profile=storage_profile
                )
            )
            flavor = flavor_cls()
            model = flavor.load(
                resolved,
                role=role,
                unsafe=unsafe,
                architecture_id=manifest.source_info.architecture_id,
            )
            # Model signature validation against the manifest.
            _validate_loaded_signature(manifest, model)
        except BaseException:
            stack.close()
            raise

        return BundleModelRuntime(
            reader=self._reader,
            manifest=manifest,
            artifact=artifact,
            resolved_artifact=resolved,
            model=model,
            exit_stack=stack,
            bundle_uri=bundle_uri,
            role=role,
        )


# ── Runtime ───────────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class BundleModelRuntime:
    """A loaded model plus the reader resources it was loaded from.

    The runtime owns the BundleReader artifact context (an ``ExitStack``)
    — S3 temp files stay alive for the lifetime of the runtime and are
    released by :meth:`close`, which is idempotent.  After loading, the
    model itself is in memory, so prediction keeps working after close
    for flavors that do not touch bundle files lazily.
    """

    def __init__(
        self,
        *,
        reader: BundleReaderLike,
        manifest: ExportManifest,
        artifact: LogicalArtifact,
        resolved_artifact: ResolvedArtifact,
        model: BundleModel,
        exit_stack: ExitStack,
        bundle_uri: str,
        role: str,
    ) -> None:
        self._reader = reader
        self._manifest = manifest
        self._artifact = artifact
        self._resolved = resolved_artifact
        self._model = model
        self._exit_stack = exit_stack
        self.bundle_uri = bundle_uri
        self.role = role
        self._closed = False

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "BundleModelRuntime":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close reader resources (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._exit_stack.close()
        logger.debug(
            "Closed bundle runtime for %r (role=%r)", self.bundle_uri, self.role
        )

    def __del__(self) -> None:
        """Best-effort release when garbage-collected without an explicit close.

        Serving consumers (Ray Serve replicas, batch actors) do not get a
        lifecycle hook from Ray, so the runtime releases its temp files on
        GC as a fallback.  ``close`` is idempotent and prediction keeps
        working after it (in-memory model contract).
        """
        try:
            self.close()
        except Exception:
            # Interpreter shutdown may have torn down modules already.
            pass

    # -- read-only views ----------------------------------------------------

    @property
    def manifest(self) -> ExportManifest:
        """The verified bundle manifest."""
        return self._manifest

    @property
    def artifact(self) -> LogicalArtifact:
        """The logical artifact selected by the role."""
        return self._artifact

    @property
    def model(self) -> BundleModel:
        """The loaded model."""
        return self._model

    @property
    def resolved_artifact(self) -> ResolvedArtifact:
        """The verified local artifact view (files valid until :meth:`close`).

        Consumers that need auxiliary files (preprocessors, configs) read
        them through :meth:`ResolvedArtifact.path_for` while the runtime
        is open.
        """
        return self._resolved

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
        return self._closed

    # -- prediction ---------------------------------------------------------

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference on named inputs.

        Models are in-memory after loading (``BundleModel`` contract), so
        prediction keeps working after :meth:`close` — close only releases
        the bundle's temp files.
        """
        return self._model.predict(inputs)


# ── Registry assembly ─────────────────────────────────────────────────────────


def _build_flavor_registry() -> FlavorRegistry:
    """Build a FlavorRegistry with built-in flavors + entry-point plugins.

    Built-in flavors are registered first and own their ``flavor_id`` —
    an entry-point exposing the same id (from an editable install) is
    skipped instead of tripping the registry's duplicate-is-conflict rule.
    """
    from tributo.integrations.flavors.onnx_runtime import ONNXRuntimeFlavor

    builtin_flavors = (ONNXRuntimeFlavor,)
    registry = FlavorRegistry()
    for cls in builtin_flavors:
        registry.register(cls)

    from tributo.plugin import discover_flavor_plugins

    builtin_ids = {cls.flavor_id for cls in builtin_flavors}
    for cls in discover_flavor_plugins():
        if getattr(cls, "flavor_id", None) in builtin_ids:
            continue
        registry.register(cls)
    return registry


def _matrix_entry(flavor_id: str) -> FlavorSupportEntry:
    for entry in SERVEABLE_FLAVOR_MATRIX:
        if entry.flavor_id == flavor_id:
            return entry
    known = [e.flavor_id for e in SERVEABLE_FLAVOR_MATRIX]
    raise UnsupportedArtifactFormat(
        f"Flavor {flavor_id!r} is not in the serveable flavor support "
        f"matrix. Supported flavors: {known}. Use a bundle whose primary "
        "artifact is serveable, or register a loader in SERVEABLE_FLAVOR_MATRIX."
    )


def _find_artifact(manifest: ExportManifest, artifact_name: str) -> LogicalArtifact:
    for artifact in manifest.artifacts:
        if artifact.name == artifact_name:
            return artifact
    available = [a.name for a in manifest.artifacts]
    raise JobConfigurationError(
        f"Artifact {artifact_name!r} not found in bundle. Available: {available}"
    )


def _check_dependencies(flavor_id: str, dependencies: tuple[str, ...]) -> None:
    missing = [dep for dep in dependencies if importlib.util.find_spec(dep) is None]
    if missing:
        raise ModelLoadError(
            f"Flavor {flavor_id!r} requires missing dependencies: {missing}. "
            "Install them first (e.g. 'uv sync') before serving this bundle."
        )


def _require_typed_signature(
    manifest: ExportManifest, artifact: LogicalArtifact, unsafe: bool
) -> None:
    """Serveable roles require non-empty typed input/output fields.

    Legacy bundles with an empty signature can still be inspected via
    ``BundleReader`` but must not enter Serving unless ``unsafe=True``.
    """
    inputs = manifest.input_signature.input_fields
    outputs = manifest.output_signature.output_fields
    if inputs and outputs:
        return
    missing = []
    if not inputs:
        missing.append("input")
    if not outputs:
        missing.append("output")
    if unsafe:
        logger.warning(
            "Loading bundle without typed signature for artifact %r "
            "(missing: %s) with unsafe=True — signature validation skipped",
            artifact.name,
            ", ".join(missing),
        )
        return
    raise UnsupportedArtifactFormat(
        f"Artifact {artifact.name!r} has no typed {', '.join(missing)} "
        "signature. Bundles without a typed signature cannot be served; "
        "re-export the model, or pass unsafe=True to load it anyway "
        "(compat-only, signature validation skipped)."
    )


def _validate_loaded_signature(manifest: ExportManifest, model: BundleModel) -> None:
    """Cross-check the manifest signature against the loaded model.

    Enforced only when the manifest declares typed fields — an unsafe
    load without a signature is compat-only and skips this check.
    Validation covers, for both inputs and outputs: field names, dtypes
    (framework-neutral), and declared shape dimensions (rank plus fixed
    axes; dynamic axes and unset shapes are skipped).
    """
    # ── Inputs ──────────────────────────────────────────────────────────
    declared_in = manifest.input_signature.input_fields
    if declared_in:
        actual_names = tuple(model.input_names)
        declared_names = tuple(f.name for f in declared_in)
        if declared_names != actual_names:
            raise ModelSchemaMismatchError(
                f"Manifest input signature {declared_names!r} does not match "
                f"the loaded model inputs {actual_names!r}"
            )
        actual_dtypes = tuple(model.input_dtypes)
        mismatched_dtypes = [
            (f.name, f.dtype, actual)
            for f, actual in zip(declared_in, actual_dtypes)
            if f.dtype != actual
        ]
        if mismatched_dtypes:
            detail = ", ".join(
                f"{name}: manifest {declared!r} vs model {actual!r}"
                for name, declared, actual in mismatched_dtypes
            )
            raise ModelSchemaMismatchError(
                f"Manifest input dtypes do not match the loaded model: {detail}"
            )
        _validate_declared_shapes(declared_in, model.input_shapes, "input")

    # ── Outputs ─────────────────────────────────────────────────────────
    declared_out = manifest.output_signature.output_fields
    if declared_out:
        actual_out_names = tuple(model.output_names)
        declared_out_names = tuple(f.name for f in declared_out)
        if declared_out_names != actual_out_names:
            raise ModelSchemaMismatchError(
                f"Manifest output signature {declared_out_names!r} does not "
                f"match the loaded model outputs {actual_out_names!r}"
            )
        actual_out_dtypes = tuple(model.output_dtypes)
        mismatched_out = [
            (f.name, f.dtype, actual)
            for f, actual in zip(declared_out, actual_out_dtypes)
            if f.dtype != actual
        ]
        if mismatched_out:
            detail = ", ".join(
                f"{name}: manifest {declared!r} vs model {actual!r}"
                for name, declared, actual in mismatched_out
            )
            raise ModelSchemaMismatchError(
                f"Manifest output dtypes do not match the loaded model: {detail}"
            )
        _validate_declared_shapes(declared_out, model.output_shapes, "output")


def _validate_declared_shapes(
    fields: tuple[SignatureField, ...],
    actual_shapes: tuple[tuple[int | None, ...], ...],
    side: str,
) -> None:
    """Compare declared signature shapes against the loaded model.

    Only fields that declare a shape are checked — an empty declared
    shape means "unspecified".  For declared shapes the rank must match
    exactly (a missing rank check would let ``(2,)`` pass against a
    model expecting ``(2, 3)`` via zip truncation), then each fixed
    (int) dimension must equal the model's.  Dynamic axes — declared as
    strings, or model dims that are ``None`` — are skipped.
    """
    for field, actual_shape in zip(fields, actual_shapes):
        declared = field.shape
        if not declared:
            continue
        if len(declared) != len(actual_shape):
            raise ModelSchemaMismatchError(
                f"Manifest {side} {field.name!r} declares rank "
                f"{len(declared)} (shape {declared!r}) but the loaded model "
                f"expects rank {len(actual_shape)} (shape {actual_shape!r})"
            )
        for dim_declared, dim_actual in zip(declared, actual_shape):
            if (
                isinstance(dim_declared, int)
                and dim_actual is not None
                and dim_declared != dim_actual
            ):
                raise ModelSchemaMismatchError(
                    f"Manifest {side} {field.name!r} declares shape "
                    f"{declared!r} but the loaded model expects {actual_shape!r}"
                )
