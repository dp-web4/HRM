"""Proof-of-possession signer tests. Hermetic (fresh seed); plus a live self-check that the
being's real seed derives the key the hub pinned, when that seed is present on this host."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sage.gateway import being_presence as bp  # noqa: E402

SEED = bytes(range(32))
PINNED = "daf57b8980c93755a3d6b8891dbe90439d87b97d046db379543dd6a17021165f"  # hub GET members/<uuid>/pubkey


def test_sign_verify_roundtrip_hex_nonce():
    nonce = "7732d638404f86af36718b8a2be12b5b"
    sig = bp.sign_nonce(SEED, nonce)
    assert bp.verify_nonce(bp.pubkey_hex(SEED), nonce, sig)


def test_domain_separation_changes_signature_and_verifies_only_with_same_domain():
    n = "abcdef0123456789abcdef0123456789"
    s1 = bp.sign_nonce(SEED, n, domain="hestia.connect.v1")
    assert s1 != bp.sign_nonce(SEED, n)
    assert bp.verify_nonce(bp.pubkey_hex(SEED), n, s1, domain="hestia.connect.v1")
    assert not bp.verify_nonce(bp.pubkey_hex(SEED), n, s1)


def test_tampered_nonce_or_wrong_key_fails():
    n = "00112233445566778899aabbccddeeff"
    sig = bp.sign_nonce(SEED, n)
    assert not bp.verify_nonce(bp.pubkey_hex(SEED), "00112233445566778899aabbccddeef0", sig)
    assert not bp.verify_nonce(bp.pubkey_hex(bytes(reversed(SEED))), n, sig)


def test_missing_or_short_seed_raises_never_falls_back():
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "k.bin"); open(p, "wb").write(b"short")
    try:
        bp.load_seed(p); assert False, "must raise"
    except ValueError:
        pass


def test_live_seed_matches_the_hub_pinned_key_if_present():
    path = os.path.expanduser(bp.DEFAULT_SEED)
    if not os.path.exists(path):
        return  # not this host
    assert bp.pubkey_hex(bp.load_seed(path)) == PINNED
    proof = bp.presence_proof("7732d638404f86af36718b8a2be12b5b1d237e3184b8dbcffae56043079fe807")
    assert bp.verify_nonce(proof["pubkey_hex"], proof["nonce"], proof["signature_hex"])


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"PASS {name}")
    print(f"\n{n} passed")
