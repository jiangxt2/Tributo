"""E3 contract tests for the shared BundleModelLoader / BundleModelRuntime.

Covers: serveable flavor support matrix consistency, explicit role
selection, flavor routing, unsupported flavors, dependency and factory
fail-fast, security gates (pickle / remote code), empty-signature
rejection with unsafe compat, model signature validation, and the
runtime's idempotent close contract (close-after-load, exception close).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.serving.bundle_fixtures import build_test_bundle
from tributo.exceptions import (
    JobConfigurationError,
    ModelLoadError,
    ModelSchemaMismatchError,
    UnsupportedArtifactFormat,
)
from tributo.exporting.registries import FlavorRegistry
from tributo.exporting.runtime import (
    DEFAULT_ROLE,
    SECURITY_MODE_PICKLE,
    SECURITY_MODE_SAFE,
    SERVEABLE_FLAVOR_MATRIX,
    BundleModel,
    BundleModelLoader,
)

# ── Fakes ──────────────────────────────────────────────────────────────────────


class _EchoModel:
    """In-memory model: label + probabilities outputs, like the ONNX
    fixture (label:int64, probabilities:float32).  Never touches bundle
    files."""

    input_names = ("float_input",)
    output_names = ("label", "probabilities")
    input_dtypes = ("float32",)
    output_dtypes = ("int64", "float32")
    input_shapes: tuple[tuple[int | None, ...], ...] = ((None, 2),)
    output_shapes: tuple[tuple[int | None, ...], ...] = ((None,), (None, 2))

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        x = inputs["float_input"]
        return {
            "label": (x.sum(axis=1) > 0).astype(np.int64),
            "probabilities": x + 1,
        }


class _EchoFlavor:
    """Fake flavor registered under onnx-runtime-v1 for runtime-logic tests."""

    api_version = 1
    flavor_id = "onnx-runtime-v1"
    security_mode = SECURITY_MODE_SAFE
    signature_required = True
    required_dependencies: tuple[str, ...] = ()

    def load(
        self,
        artifact: Any,
        *,
        role: str,
        unsafe: bool = False,
        architecture_id: str | None = None,
    ) -> BundleModel:
        del artifact, role, unsafe, architecture_id
        return _EchoModel()


class _RebuildFlavor(_EchoFlavor):
    """Flavor that rebuilds via ModelFactoryRegistry using the manifest's
    ``architecture_id`` (passed by the runtime, not guessed)."""

    def load(
        self,
        artifact: Any,
        *,
        role: str,
        unsafe: bool = False,
        architecture_id: str | None = None,
    ) -> BundleModel:
        from tributo.exporting.registries import ModelFactoryRegistry

        # The runtime passes the manifest's architecture_id; rebuilding
        # through the registry must fail fast when no factory exists.
        assert architecture_id is not None
        factory_cls = ModelFactoryRegistry().get(architecture_id)
        factory_cls()
        return _EchoModel()


def _loader(flavor: type[Any] = _EchoFlavor) -> BundleModelLoader:
    registry = FlavorRegistry()
    registry.register(flavor)
    return BundleModelLoader(flavor_registry=registry)


# ── Serveable flavor support matrix ───────────────────────────────────────────


class TestFlavorSupportMatrix:
    """Matrix rows must resolve to real loaders with matching ids."""

    def test_matrix_loader_paths_resolve(self) -> None:
        """每行 loader import 路径可解析，且 flavor_id 与类声明一致。"""
        import importlib

        assert SERVEABLE_FLAVOR_MATRIX, "matrix must not be empty"
        for entry in SERVEABLE_FLAVOR_MATRIX:
            module_name, _, attr = entry.loader.partition(":")
            module = importlib.import_module(module_name)
            cls = getattr(module, attr)
            assert cls.flavor_id == entry.flavor_id

    def test_matrix_flavors_are_registered(self) -> None:
        """矩阵中的 flavor 必须能在默认 FlavorRegistry 中路由。"""
        from tributo.exporting.runtime import _build_flavor_registry

        registry = _build_flavor_registry()
        for entry in SERVEABLE_FLAVOR_MATRIX:
            assert entry.flavor_id in registry.list_all()

    def test_matrix_entry_point_consistency(self) -> None:
        """pyproject entry-point 与矩阵一致（onnx-runtime-v1）。"""
        from importlib.metadata import entry_points

        flavors = entry_points(group="tributo.model_flavors")
        ep_names = {ep.name for ep in flavors}
        matrix_ids = {e.flavor_id for e in SERVEABLE_FLAVOR_MATRIX}
        assert matrix_ids.issubset(ep_names), (
            f"matrix flavors {matrix_ids - ep_names} missing from entry points"
        )

    def test_onnx_runtime_is_safe_and_signature_required(self) -> None:
        """onnx-runtime-v1 行声明 safe 安全模式与 typed signature 要求。"""
        (entry,) = [
            e for e in SERVEABLE_FLAVOR_MATRIX if e.flavor_id == "onnx-runtime-v1"
        ]
        assert entry.security_mode == SECURITY_MODE_SAFE
        assert entry.signature_required is True
        assert "onnxruntime" in entry.dependencies


# ── Role selection ─────────────────────────────────────────────────────────────


class TestRoleSelection:
    """BundleModelLoader 显式 role 契约。"""

    def test_default_role_is_frozen_inference(self) -> None:
        """便捷默认值只能冻结为 inference。"""
        assert DEFAULT_ROLE == "inference"

    def test_explicit_role_loads(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path)
        runtime = _loader().open(str(bundle), role="inference")
        assert runtime.role == "inference"
        runtime.close()

    def test_unknown_role_fails_fast(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path)
        with pytest.raises(JobConfigurationError, match="Role 'serve' not found"):
            _loader().open(str(bundle), role="serve")


# ── Flavor routing & gates ─────────────────────────────────────────────────────


class TestFlavorRouting:
    """flavor_id 是唯一路由键；format/猜测不作为加载依据。"""

    def test_flavor_id_routes_to_loader(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path, flavor_id="onnx-runtime-v1")
        runtime = _loader().open(str(bundle), role="inference")
        try:
            result = runtime.predict(
                {"float_input": np.array([[1.0, 2.0]], dtype=np.float32)}
            )
            np.testing.assert_allclose(result["probabilities"], [[2.0, 3.0]])
        finally:
            runtime.close()

    def test_unsupported_flavor_fails_fast(self, tmp_path: Path) -> None:
        """注册但不在 serveable matrix 的 flavor 必须显式拒绝。"""
        bundle = build_test_bundle(tmp_path, flavor_id="safetensors-v1")

        class _SafetensorsFlavor(_EchoFlavor):
            flavor_id = "safetensors-v1"

        with pytest.raises(UnsupportedArtifactFormat, match="not in the serveable"):
            _loader(_SafetensorsFlavor).open(str(bundle), role="inference")

    def test_unregistered_flavor_fails_fast(self, tmp_path: Path) -> None:
        """无 loader 注册的 flavor：JobConfigurationError 列出可用项。"""
        bundle = build_test_bundle(tmp_path, flavor_id="torch-export-v1")
        with pytest.raises(JobConfigurationError, match="no loader is registered"):
            _loader().open(str(bundle), role="inference")

    def test_missing_dependency_fails_fast(self, tmp_path: Path) -> None:
        """声明了缺失依赖的 flavor：ModelLoadError + 安装提示。"""
        bundle = build_test_bundle(tmp_path)

        class _MissingDepFlavor(_EchoFlavor):
            required_dependencies = ("definitely-not-a-real-module-xyz",)

        with pytest.raises(ModelLoadError, match="missing dependencies"):
            _loader(_MissingDepFlavor).open(str(bundle), role="inference")

    def test_missing_model_factory_fails_fast(self, tmp_path: Path) -> None:
        """重建型 flavor 收到 manifest 的 architecture_id，缺 factory 时结构化失败。"""
        bundle = build_test_bundle(tmp_path, architecture_id="dnn")
        with pytest.raises(JobConfigurationError, match="Unknown architecture"):
            _loader(_RebuildFlavor).open(str(bundle), role="inference")


class TestSecurityGate:
    """pickle / remote-code 默认拒绝，unsafe 显式开启。"""

    def test_unsafe_flavor_refused_by_default(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path)

        class _PickleFlavor(_EchoFlavor):
            security_mode = SECURITY_MODE_PICKLE

        with pytest.raises(UnsupportedArtifactFormat, match="refusing to load"):
            _loader(_PickleFlavor).open(str(bundle), role="inference")

    def test_unsafe_flavor_loads_with_unsafe_flag(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path)

        class _PickleFlavor(_EchoFlavor):
            security_mode = SECURITY_MODE_PICKLE

        runtime = _loader(_PickleFlavor).open(
            str(bundle), role="inference", unsafe=True
        )
        runtime.close()


# ── Signature validation ───────────────────────────────────────────────────────


class TestSignatureValidation:
    """可服务 role 必须非空 typed signature；空签名仅 unsafe compat。"""

    def test_empty_signature_refused_by_default(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path, with_signature=False)
        with pytest.raises(UnsupportedArtifactFormat, match="no typed"):
            _loader().open(str(bundle), role="inference")

    def test_empty_signature_loads_with_unsafe_flag(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path, with_signature=False)
        runtime = _loader().open(str(bundle), role="inference", unsafe=True)
        try:
            result = runtime.predict(
                {"float_input": np.array([[1.0]], dtype=np.float32)}
            )
            assert "probabilities" in result
        finally:
            runtime.close()

    def test_signature_mismatch_fails_fast(self, tmp_path: Path) -> None:
        """manifest 签名与实际模型输入不一致 → ModelSchemaMismatchError。"""
        bundle = build_test_bundle(tmp_path)

        class _MismatchedModel(_EchoModel):
            input_names = ("other_input",)

        class _MismatchedFlavor(_EchoFlavor):
            def load(
                self,
                artifact: Any,
                *,
                role: str,
                unsafe: bool = False,
                architecture_id: str | None = None,
            ) -> BundleModel:
                return _MismatchedModel()

        with pytest.raises(ModelSchemaMismatchError, match="does not match"):
            _loader(_MismatchedFlavor).open(str(bundle), role="inference")

    def test_declared_rank_mismatch_rejected(self, tmp_path: Path) -> None:
        """manifest 声明 rank 1 (2,)，模型实际 rank 2 (2,3) → 拒绝。

        A naive zip-over-dims comparison would truncate and let this
        pass — the rank must be checked before comparing dimensions.
        """
        bundle = build_test_bundle(tmp_path, input_field_shape=(2,))

        class _RankMismatchModel(_EchoModel):
            input_shapes: tuple[tuple[int | None, ...], ...] = ((2, 3),)

        class _RankMismatchFlavor(_EchoFlavor):
            def load(
                self,
                artifact: Any,
                *,
                role: str,
                unsafe: bool = False,
                architecture_id: str | None = None,
            ) -> BundleModel:
                return _RankMismatchModel()

        with pytest.raises(ModelSchemaMismatchError, match="declares rank"):
            _loader(_RankMismatchFlavor).open(str(bundle), role="inference")

    def test_declared_output_shape_mismatch_rejected(self, tmp_path: Path) -> None:
        """manifest 声明的输出固定维与实际不一致 → 拒绝。"""
        bundle = build_test_bundle(tmp_path, output_field_shapes={"label": (2,)})

        class _OutputMismatchModel(_EchoModel):
            output_shapes: tuple[tuple[int | None, ...], ...] = ((3,), (None, 2))

        class _OutputMismatchFlavor(_EchoFlavor):
            def load(
                self,
                artifact: Any,
                *,
                role: str,
                unsafe: bool = False,
                architecture_id: str | None = None,
            ) -> BundleModel:
                return _OutputMismatchModel()

        with pytest.raises(ModelSchemaMismatchError, match="output"):
            _loader(_OutputMismatchFlavor).open(str(bundle), role="inference")

    def test_declared_shapes_matching_pass(self, tmp_path: Path) -> None:
        """动态轴（"batch" ↔ None）与固定维一致 → 通过。"""
        bundle = build_test_bundle(tmp_path, input_field_shape=("batch", 2))
        runtime = _loader().open(str(bundle), role="inference")
        runtime.close()


# ── Runtime lifecycle ──────────────────────────────────────────────────────────


class TestRuntimeLifecycle:
    """Runtime 持有 reader context；close 幂等；异常关闭不泄漏。"""

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path)
        runtime = _loader().open(str(bundle), role="inference")
        runtime.close()
        runtime.close()  # 第二次 close 是 no-op

    def test_close_after_load_keeps_model_usable(self, tmp_path: Path) -> None:
        """加载完成后关闭资源，predict 仍工作（模型已入内存）。"""
        bundle = build_test_bundle(tmp_path)
        runtime = _loader().open(str(bundle), role="inference")
        runtime.close()
        result = runtime.predict({"float_input": np.array([[5.0]], dtype=np.float32)})
        np.testing.assert_allclose(result["probabilities"], [[6.0]])

    def test_failed_load_closes_artifact_context(self, tmp_path: Path) -> None:
        """flavor.load 抛错时 reader context 必须被正确关闭。"""
        bundle = build_test_bundle(tmp_path)

        class _ExplodingFlavor(_EchoFlavor):
            def load(
                self,
                artifact: Any,
                *,
                role: str,
                unsafe: bool = False,
                architecture_id: str | None = None,
            ) -> BundleModel:
                raise RuntimeError("boom")

        class _RecordingReader:
            def __init__(self) -> None:
                self.exits: list[bool] = []

            def read_manifest(
                self, manifest_or_bundle_uri: str, *, storage_profile: str | None = None
            ) -> Any:
                import json

                from tributo.exporting.manifest import _read_manifest_v1

                raw = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
                return _read_manifest_v1(raw, b"")

            def open_artifact(
                self,
                manifest_or_bundle_uri: str,
                *,
                role: str | None = None,
                artifact_name: str | None = None,
                storage_profile: str | None = None,
            ) -> Any:
                from contextlib import contextmanager

                @contextmanager
                def _cm() -> Any:
                    self.exits.append(False)
                    try:
                        yield None
                    finally:
                        self.exits[-1] = True

                return _cm()

        reader = _RecordingReader()
        registry = FlavorRegistry()
        registry.register(_ExplodingFlavor)
        loader = BundleModelLoader(bundle_reader=reader, flavor_registry=registry)
        with pytest.raises(RuntimeError, match="boom"):
            loader.open(str(bundle), role="inference")
        assert reader.exits and reader.exits[-1] is True, (
            "artifact context was not closed"
        )

    def test_manifest_view_available(self, tmp_path: Path) -> None:
        bundle = build_test_bundle(tmp_path)
        runtime = _loader().open(str(bundle), role="inference")
        try:
            assert runtime.manifest.bundle_id == "bundle-e3-test"
            assert runtime.artifact.name == "model"
            assert runtime.resolved_artifact.entrypoint_path.name == "model.onnx"
        finally:
            runtime.close()


# ── End-to-end ONNX Runtime path ───────────────────────────────────────────────


class TestOnnxRuntimeEndToEnd:
    """真实 ONNX 模型经 BundleModelLoader 加载并推理。"""

    def test_load_and_predict_real_onnx(self, tmp_path: Path) -> None:
        import pytest as _pytest

        onnx_path = None
        try:
            from tests.serving.bundle_fixtures import make_dummy_onnx

            onnx_path = make_dummy_onnx(tmp_path)
        except Exception as exc:
            _pytest.skip(f"skl2onnx or sklearn not installed: {exc}")

        bundle = build_test_bundle(tmp_path, onnx_path=onnx_path)
        loader = BundleModelLoader()  # 默认 registry（含内置 onnx-runtime-v1）
        runtime = loader.open(str(bundle), role="inference")
        try:
            result = runtime.predict(
                {"float_input": np.array([[0.5, 0.5]], dtype=np.float32)}
            )
            assert "label" in result
            assert "probabilities" in result
        finally:
            runtime.close()

    def test_default_loader_requires_typed_signature(self, tmp_path: Path) -> None:
        """默认 loader（真实 ONNXRuntimeFlavor）拒绝空签名 bundle。"""
        bundle = build_test_bundle(tmp_path, with_signature=False)
        loader = BundleModelLoader()
        with pytest.raises(UnsupportedArtifactFormat, match="no typed"):
            loader.open(str(bundle), role="inference")


class TestFileUriAndE2E:
    """file:// URI 与真实 ONNX 路径的补充契约。"""

    def test_file_uri_scheme_accepted(self, tmp_path: Path) -> None:
        """file:// 前缀的 bundle URI 应可用（与 BundleOutputConfig 一致）。"""
        bundle = build_test_bundle(tmp_path)
        runtime = _loader().open("file://" + str(bundle), role="inference")
        try:
            assert runtime.manifest.bundle_id == "bundle-e3-test"
        finally:
            runtime.close()

    def test_file_uri_real_onnx_predict(self, tmp_path: Path) -> None:
        """file:// URI + 真实 ONNX 模型端到端推理。"""
        from tests.serving.bundle_fixtures import make_dummy_onnx

        onnx_path = make_dummy_onnx(tmp_path)
        bundle = build_test_bundle(tmp_path, onnx_path=onnx_path)
        loader = BundleModelLoader()
        with loader.open("file://" + str(bundle), role="inference") as runtime:
            result = runtime.predict(
                {"float_input": np.array([[0.5, 0.5]], dtype=np.float32)}
            )
            assert result["label"].shape == (1,)
            assert result["probabilities"].shape == (1, 2)
