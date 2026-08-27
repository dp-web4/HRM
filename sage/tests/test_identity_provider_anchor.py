#!/usr/bin/env python3
"""Identity provider regression tests — M-CIT-0 (a being's key is verifiable).

Two properties, both of which the provider previously violated silently:

1. The manifest records the anchor ACTUALLY ACHIEVED, never the one requested.
   No hardware sealing is implemented on either provider, so a 'tpm2' request
   seals in software. Recording the request would publish trust_ceiling 1.0 for
   a secret XORed against sha256(hostname:mac:instance_dir) — the fleet's highest
   trust ceiling, mintable by passing a string.

2. A sealed file whose machine key no longer derives is REFUSED, not silently
   unsealed into garbage. XOR is unauthenticated, so a wrong key returns
   plausible bytes; authorize() used to build a signing context asserting the
   manifest's fingerprint with a secret that cannot produce it.

The rust provider carries the mirror of both in
sage-rs/sage-lib/src/identity/provider.rs.

stdlib unittest, matching test_lct_identity.py — these guard the M-CIT-0 gate and
must run on any seat with bare python3, without pytest installed.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.identity.provider import IdentityProvider


class IdentityAnchorTests(unittest.TestCase):

    def setUp(self):
        self.instance_dir = Path(tempfile.mkdtemp(prefix='sage-identity-anchor-'))
        self.addCleanup(shutil.rmtree, self.instance_dir, ignore_errors=True)

    def _init(self, anchor_type, directory=None):
        provider = IdentityProvider(str(directory or self.instance_dir))
        manifest = provider.initialize(
            name='test-instance',
            lct_id='lct://sage:test:agent@test',
            machine='test-machine',
            model='test-model:latest',
            anchor_type=anchor_type,
        )
        return provider, manifest

    def test_requested_hardware_anchor_is_downgraded_not_claimed(self):
        """A hardware anchor this provider cannot deliver must not reach the manifest.

        'tpm2_no_pcr' is the sharp case: it carries a ceiling of 0.85 in
        TRUST_CEILINGS but matched no branch of the old _seal_secret, so it wrote
        NO sealed file at all and still returned a manifest — an identity that
        could never authorize again.
        """
        for requested in ('tpm2', 'fido2', 'secure_enclave', 'tpm2_no_pcr'):
            with self.subTest(requested=requested):
                directory = Path(tempfile.mkdtemp(prefix=f'sage-anchor-{requested}-'))
                self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
                provider, manifest = self._init(requested, directory)

                # The manifest prices how the secret is HELD, not what was asked for.
                self.assertEqual(manifest.anchor_type, 'software')
                self.assertEqual(manifest.trust_ceiling, 0.4)

                # A sealed file exists at all (the tpm2_no_pcr silent-skip case).
                sealed_path = directory / 'identity.sealed'
                self.assertTrue(sealed_path.exists(), 'no sealed file was written')

                # ...and its header agrees with the manifest.
                with open(sealed_path, 'rb') as f:
                    self.assertEqual(f.readline().strip(), b'SAGE_SEALED_v1')
                    self.assertEqual(f.readline().strip(), b'software')

                # The identity still round-trips: seal -> lock -> unseal -> authorize.
                provider.lock()
                self.assertFalse(provider.is_authorized)
                context = provider.authorize()
                self.assertIsNotNone(
                    context, 'identity could not re-authorize after sealing')
                self.assertEqual(context.anchor_type, 'software')

    def test_software_anchor_is_recorded_unchanged(self):
        """The honest request is untouched — the downgrade is not a blanket rewrite."""
        _, manifest = self._init('software')
        self.assertEqual(manifest.anchor_type, 'software')
        self.assertEqual(manifest.trust_ceiling, 0.4)

    def test_attestation_reports_the_achieved_anchor(self):
        """Peers read the attestation, so the claim must not survive there either."""
        provider, _ = self._init('tpm2')
        attestation = provider.get_attestation()
        self.assertIsNotNone(attestation)
        self.assertEqual(attestation['anchor_type'], 'software')
        self.assertEqual(attestation['trust_ceiling'], 0.4)

    def test_relocated_identity_is_refused_not_silently_wrong(self):
        """instance_dir feeds the machine key, so a moved identity must fail closed.

        Before the fingerprint check, this path returned a garbage secret and built
        a SigningContext carrying the manifest's fingerprint — a signed-shaped
        attestation naming an identity the held secret cannot generate.
        """
        provider, manifest = self._init('software')
        self.assertTrue(provider.is_authorized)

        moved = Path(tempfile.mkdtemp(prefix='sage-identity-moved-'))
        self.addCleanup(shutil.rmtree, moved, ignore_errors=True)
        for name in ('identity.json', 'identity.sealed', 'identity.attest.json'):
            shutil.copy(self.instance_dir / name, moved / name)

        relocated = IdentityProvider(str(moved))
        self.assertTrue(relocated.is_initialized)

        context = relocated.authorize()
        self.assertIsNone(
            context,
            'relocated identity must be REFUSED; the unsealed secret cannot '
            f'produce fingerprint {manifest.public_key_fingerprint}')
        self.assertFalse(relocated.is_authorized)


if __name__ == '__main__':
    unittest.main()
