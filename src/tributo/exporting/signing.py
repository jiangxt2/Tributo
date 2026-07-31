"""Bundle manifest signing — Ed25519 signatures for tamper detection.

Provides:

- ``BundleSigner``: Sign a bundle manifest with an Ed25519 private key.
- ``BundleVerifier``: Verify a manifest signature against a public key.
- ``BundleSignature``: Self-describing signature model (algorithm + key id + sig).
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tributo.util.annotations import PublicAPI

# ── Signature model ──────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class BundleSignature(BaseModel):
    """Self-describing Ed25519 signature for a bundle manifest.

    Embedded in the bundle's ``metadata/signature.json`` (per the plan's
    bundle directory layout) so verifiers can validate authenticity
    without external state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm: str = Field(default="ed25519", pattern=r"^ed25519$")
    key_id: str = Field(
        ...,
        min_length=8,
        description="Short identifier for the signing key (e.g. 'prod-2024').",
    )
    signature: str = Field(
        ...,
        min_length=64,
        description="Base64-encoded Ed25519 signature over canonical JSON.",
    )
    signed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    manifest_sha256: str = Field(min_length=64, max_length=64)


# ── Signer ───────────────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class BundleSigner:
    """Sign bundle manifests with an Ed25519 private key.

    Usage::

        signer = BundleSigner(private_key_bytes, key_id="prod-2024")
        sig = signer.sign(manifest_canonical_json_bytes)
        # Write sig.model_dump_json() to metadata/signature.json
    """

    def __init__(self, private_key_bytes: bytes, key_id: str) -> None:
        """Initialise with a 32-byte Ed25519 seed or private key.

        Args:
            private_key_bytes: 32-byte Ed25519 seed or loaded private key.
            key_id: Human-readable key identifier.
        """
        from cryptography.hazmat.primitives.asymmetric import ed25519

        if len(private_key_bytes) != 32:
            raise ValueError(
                f"Ed25519 private key must be 32 bytes, got {len(private_key_bytes)}"
            )
        self._private_key = ed25519.Ed25519PrivateKey.from_private_bytes(
            private_key_bytes
        )
        self._key_id = key_id

    @property
    def public_key_bytes(self) -> bytes:
        """Return the raw 32-byte public key (for distribution)."""
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            PublicFormat,
        )

        raw: bytes = self._private_key.public_key().public_bytes(
            encoding=Encoding.Raw,
            format=PublicFormat.Raw,
        )
        return raw

    def sign(self, manifest_bytes: bytes) -> BundleSignature:
        """Sign *manifest_bytes* (canonical JSON) and return a ``BundleSignature``.

        The signature is computed over SHA-256(manifest_bytes), NOT the
        raw bytes, so the signed payload is always 32 bytes regardless of
        manifest size.
        """
        digest = hashlib.sha256(manifest_bytes).digest()
        raw_sig = self._private_key.sign(digest)
        sig_b64 = base64.b64encode(raw_sig).decode("ascii")
        manifest_sha256_hex = hashlib.sha256(manifest_bytes).hexdigest()

        return BundleSignature(
            algorithm="ed25519",
            key_id=self._key_id,
            signature=sig_b64,
            manifest_sha256=manifest_sha256_hex,
        )


# ── Verifier ─────────────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class BundleVerifier:
    """Verify bundle manifest signatures.

    Usage::

        verifier = BundleVerifier(trusted_keys={"prod-2024": public_key_bytes})
        verifier.verify(signature, manifest_bytes)
    """

    def __init__(self, trusted_keys: dict[str, bytes]) -> None:
        """Initialise with a map of key_id → 32-byte Ed25519 public key.

        Args:
            trusted_keys: Mapping from key_id to raw 32-byte public key.
        """
        from cryptography.hazmat.primitives.asymmetric import ed25519

        self._trusted: dict[str, ed25519.Ed25519PublicKey] = {}
        for kid, raw in trusted_keys.items():
            if len(raw) != 32:
                raise ValueError(
                    f"Ed25519 public key for {kid!r} must be 32 bytes, got {len(raw)}"
                )
            self._trusted[kid] = ed25519.Ed25519PublicKey.from_public_bytes(raw)

    def verify(
        self,
        signature: BundleSignature,
        manifest_bytes: bytes,
    ) -> bool:
        """Verify that *signature* is valid for *manifest_bytes*.

        Returns ``True`` if the signature is valid and the key is trusted.
        """
        if signature.key_id not in self._trusted:
            return False

        public_key = self._trusted[signature.key_id]
        digest = hashlib.sha256(manifest_bytes).digest()

        try:
            sig_bytes = base64.b64decode(signature.signature)
            public_key.verify(sig_bytes, digest)
            return True
        except Exception:
            return False

    def verify_manifest(
        self,
        signature: BundleSignature,
        manifest: dict[str, Any],
    ) -> bool:
        """Verify *signature* against a JSON manifest dict.

        The manifest is canonicalised (sorted keys, no indent) before hashing,
        matching the behaviour of ``ExportManifest.canonical_json()``.
        """
        canonical = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return self.verify(signature, canonical)


# ── Key generation helper ────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
def generate_signing_key() -> tuple[bytes, bytes]:
    """Generate a fresh Ed25519 key pair.

    Returns:
        ``(private_key_bytes, public_key_bytes)`` — both 32 bytes.
    """
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    private_key = ed25519.Ed25519PrivateKey.generate()
    priv_bytes = private_key.private_bytes(
        encoding=Encoding.Raw,
        format=PrivateFormat.Raw,
        encryption_algorithm=NoEncryption(),
    )
    pub_bytes = private_key.public_key().public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw,
    )
    return priv_bytes, pub_bytes
