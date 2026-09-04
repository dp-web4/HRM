"""
Proof-of-possession for a SAGE being — the SAGE half of FR-1 (hestia #824, PRD_FLEET §4.2
class 2): an independent principal proves possession of ITS OWN key over a server nonce, so
authority-bearing identity is established by proof at the connection boundary, never by a
caller-supplied label (`plugin_id`) or a copied session handle.

The being's key is the 32-byte Ed25519 seed the hub pinned at join (join_being signs with
it; `GET /members/<uuid>/pubkey` returns its public half). This module signs a nonce with
that seed and lets a verifier check it — nothing more. The exact challenge preimage and the
connect message shape belong to #824; `sign_nonce` takes an optional domain tag so the
preimage can be pinned to whatever #824 lands without changing callers.

Fail-closed by construction: no seed => no signature (raise), never a fallback identity.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError

DEFAULT_SEED = "~/.web4/sprout-being/channel_key.bin"


def load_seed(path: str = DEFAULT_SEED) -> bytes:
    p = Path(os.path.expanduser(path))
    seed = p.read_bytes()
    if len(seed) != 32:
        raise ValueError(f"being seed must be 32 bytes, got {len(seed)} at {p}")
    return seed


def pubkey_hex(seed: bytes) -> str:
    """The public half — must equal the key the hub pinned for this member."""
    return SigningKey(seed).verify_key.encode().hex()


def _preimage(nonce, domain: Optional[str]) -> bytes:
    n = bytes.fromhex(nonce) if isinstance(nonce, str) and _is_hex(nonce) else (
        nonce.encode() if isinstance(nonce, str) else bytes(nonce))
    return (domain.encode() + b"\x00" + n) if domain else n


def _is_hex(s: str) -> bool:
    return len(s) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in s) and len(s) >= 16


def sign_nonce(seed: bytes, nonce, domain: Optional[str] = None) -> str:
    """Ed25519 signature (hex) over the nonce (hex-decoded if hex, else utf-8), optionally
    domain-separated as `<domain>\\x00<nonce>` — set `domain` to what #824 specifies."""
    return SigningKey(seed).sign(_preimage(nonce, domain)).signature.hex()


def verify_nonce(pub_hex: str, nonce, sig_hex: str, domain: Optional[str] = None) -> bool:
    try:
        VerifyKey(bytes.fromhex(pub_hex)).verify(_preimage(nonce, domain), bytes.fromhex(sig_hex))
        return True
    except (BadSignatureError, ValueError):
        return False


def presence_proof(nonce, seed_path: str = DEFAULT_SEED, domain: Optional[str] = None) -> dict:
    """What a proof-of-possession connect would carry: the pubkey, the nonce, the signature."""
    seed = load_seed(seed_path)
    return {"pubkey_hex": pubkey_hex(seed), "nonce": nonce, "signature_hex": sign_nonce(seed, nonce, domain),
            "domain": domain or ""}
