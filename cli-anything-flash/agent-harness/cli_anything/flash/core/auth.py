"""Authentication guard for connecting the FLASH harness to the server tree."""

from __future__ import annotations

import hashlib
import hmac
import os


DEFAULT_PASSWORD_HASH = (
    "pbkdf2_sha256$200000$4967dd7b91f7262e0a7bfcbded36bab4"
    "$170d28b58998ab2f9ab3d4d925c2b2d355ba3741ab21e88df412afaa45223871"
)


class FlashAuthError(PermissionError):
    """Raised when a server connection is attempted without valid auth."""


def _configured_hash() -> str:
    return os.environ.get("FLASH_SKILL_PASSWORD_HASH", DEFAULT_PASSWORD_HASH).strip()


def verify_password(password: str | None) -> bool:
    """Verify a password against the configured PBKDF2 hash."""

    if password is None:
        return False
    parts = _configured_hash().split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        raise FlashAuthError("Invalid FLASH_SKILL_PASSWORD_HASH configuration.")
    _, iterations_text, salt_hex, digest_hex = parts
    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        str(password).encode("utf-8"),
        bytes.fromhex(salt_hex),
        int(iterations_text),
    ).hex()
    return hmac.compare_digest(candidate, digest_hex)


def require_password(password: str | None) -> None:
    """Raise unless the password is correct."""

    if not verify_password(password):
        raise FlashAuthError("FLASH server connection denied: password is missing or incorrect.")
