"""
SAGE Identity — Three-layer hardware-gated identity system.

Layer A: Public manifest (identity.json) — who I am, readable by anyone
Layer B: Sealed secret (identity.sealed) — hardware-gated root material
Layer C: Attestation cache (identity.attest.json) — recent hardware proofs

The manifest stays readable. The secret requires hardware authorization
to unseal. The attestation proves legitimacy to peers and governance.

Usage:
    from sage.identity import IdentityProvider

    provider = IdentityProvider(instance_dir)
    manifest = provider.initialize(anchor_type='software')  # First-time setup

    # No hardware anchor is implemented yet. Requesting one ('tpm2', 'fido2',
    # 'secure_enclave') seals in software anyway and is RECORDED as 'software',
    # so manifest.anchor_type / trust_ceiling describe how the secret is really
    # held. Read them back rather than assuming the request was honoured.

    # On startup — requires hardware
    context = provider.authorize()
    if context:
        # Identity authorized — signing context available
        signature = context.sign(data)

    # On shutdown
    provider.lock()  # Clear in-memory secrets
"""

from sage.identity.provider import IdentityProvider

__all__ = ['IdentityProvider']
