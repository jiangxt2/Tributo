"""Post-publish hook protocol — downstream actions after bundle commit.

A ``PostPublishHook`` is invoked after ``BundleRepository.commit()``
succeeds.  Hooks are independent of the commit lifecycle: a failed hook
does not invalidate the bundle.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from tributo.util.annotations import PublicAPI

# ── Data models ────────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class HookReceipt(BaseModel):
    """Result of a single post-publish hook execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hook_id: str
    status: str = Field(pattern=r"^(success|failed|skipped)$")
    idempotency_key: str = ""
    error: str | None = None
    retryable: bool = False


# ── Protocol ────────────────────────────────────────────────────────────────────


@runtime_checkable
@PublicAPI(stability="beta")
class PostPublishHook(Protocol):
    """A downstream action that runs after a bundle has been committed.

    Hooks are best-effort by default.  Set ``required=True`` in the
    hook configuration to make failure abort the publication flow.
    """

    hook_id: ClassVar[str]

    def execute(
        self,
        canonical_uri: str,
        manifest: dict[str, Any],
        options: dict[str, Any] | None = None,
        local_bundle_dir: str | None = None,
    ) -> HookReceipt:
        """Execute the hook.

        Args:
            canonical_uri: The bundle's canonical URI.
            manifest: The committed manifest as a JSON dict.
            options: Hook-specific configuration.
            local_bundle_dir: Local bundle directory, valid only during
                the staging window (hooks run before staging cleanup).
                Hooks must verify the directory actually holds a bundle
                (``manifest.json``) before relying on it — for S3-only
                publishes the staging layout is ``nodes/<id>/artifact/...``,
                not a bundle layout.

        Returns:
            A ``HookReceipt`` describing the outcome.
        """
        ...

    def idempotency_key(
        self,
        canonical_uri: str,
        manifest_sha256: str,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Return a deterministic idempotency key for this hook invocation."""
        ...


# ── Runner ──────────────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class PublicationRunner:
    """Execute a list of post-publish hooks sequentially.

    Each hook receives the committed bundle's metadata.  Failures are
    collected as receipts; required-hook failures propagate immediately.
    """

    def __init__(
        self, hooks: list[tuple[PostPublishHook, dict[str, Any], bool]]
    ) -> None:
        self._hooks = hooks

    def run(
        self,
        canonical_uri: str,
        manifest: dict[str, Any],
        manifest_sha256: str,
        local_bundle_dir: str | None = None,
    ) -> list[HookReceipt]:
        """Execute all hooks.

        Args:
            canonical_uri: The bundle URI.
            manifest: The manifest dict.
            manifest_sha256: SHA-256 of the manifest — the canonical JSON
                digest computed by the Publisher (the same value stored
                in ``BundleResult`` and S3 metadata).
            local_bundle_dir: Local bundle directory, valid only during
                the staging window (see ``PostPublishHook.execute``).

        Returns:
            Receipts for every hook that ran.
        """
        # The manifest JSON on disk carries no ``_manifest_sha256`` key —
        # surface the publisher's canonical digest so hooks that read it
        # (e.g. the MLflow hook) report the value consumers verify
        # against, not a re-computed fallback.
        manifest["_manifest_sha256"] = manifest_sha256

        receipts: list[HookReceipt] = []
        for hook, options, required in self._hooks:
            receipt = hook.execute(canonical_uri, manifest, options, local_bundle_dir)
            receipts.append(receipt)
            if receipt.status == "failed" and required:
                raise RuntimeError(
                    f"Required hook {hook.hook_id!r} failed: {receipt.error}"
                )
        return receipts
