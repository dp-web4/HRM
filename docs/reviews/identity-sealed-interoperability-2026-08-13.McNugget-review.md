# McNugget review of PR #25 — every claim verifies, and the boundary is worse than stated in two ways

**Date:** 2026-08-13
**Reviewer:** McNugget (Opus 4.8, Mac mini M4, `0203dc0b`)
**Reviewing:** PR #25 `docs/reviews/identity-sealed-interoperability-2026-08-13.md` (external review)
**Method:** re-derived each direct observation from source at `main`, rather than reading the note and
agreeing with it. Findings below are anchored to file:line.

## Verdict: **support the merge.** All six direct observations reproduce.

| claim | verified at | result |
|---|---|---|
| divergent machine-key derivation | `sage/identity/provider.py:363` vs `sage-rs/sage-lib/src/identity/provider.rs:261` | ✅ Python `f"{gethostname()}:{uuid.getnode()}:{instance_dir}"`; Rust `format!("{}:0:{}", hostname, instance_dir)` — a **literal `0`** where Python puts the MAC |
| identical sealed-file layout | `provider.py:314-316`, `provider.rs:235-236` | ✅ both `SAGE_SEALED_v1\n<anchor>\n<sealed>` |
| unconditional XOR unseal | `provider.py:346-347`, `provider.rs:251-252` | ✅ neither can fail on a wrong key |
| no fingerprint check before signing context | `provider.py:173-195`, `provider.rs:146-160` | ✅ both pass `manifest.public_key_fingerprint` straight through |
| matching fingerprint representation | `signing.rs:31-35` (sha256→hex, 8 bytes) vs `provider.py:141` (sha256 hexdigest[:16]) | ✅ same 16-hex-char representation |
| Rust provider dormant outside tests | grep for `IdentityProvider::new` / `identity::provider` across `sage-rs/` | ✅ **zero** references outside `provider.rs` |

The inference (that the two are meant to consume one format) is also well-grounded: byte-identical magic,
mirrored API, matching fingerprint representation. I'd merge on the direct observations alone.

## Two things the note does not mention

### 1. The anchor line is written without regard to truth and read without regard to content

Rust's seal takes an anchor parameter and **discards it** — the signature is `fn seal_secret(&self, secret:
&[u8], _anchor_type: &str)` (`provider.rs:231`, note the underscore) and the header is the hardcoded literal
`b"SAGE_SEALED_v1\nsoftware\n"` (`:235`). So a Rust-sealed identity always *claims* software anchoring
regardless of what was requested. Python's seal, by contrast, honours the parameter and has real tpm2/fido2
branches (`provider.py:307-323`).

The read side is the mirror image: Python's unseal reads `anchor_line = f.readline().strip()`
(`provider.py:339`) and then **never uses it** — it proceeds to software XOR unconditionally
(`:346-347`). So if a file were ever sealed under a hardware anchor, Python would XOR-unseal it anyway and
return plausible garbage.

Net: the anchor field is neither authored nor consumed correctly on either side. It is decoration today.
That matters because the *whole point* of the field is to record the trust ceiling
(`trust_ceiling_for(anchor_type)`, `provider.rs:15`) — a value that governs how much the identity is
believed. It is the same failure class the note already names (unauthenticated bytes accepted as truth),
just on the anchor axis instead of the key axis, and it should be fixed in the same pass.

### 2. The missing fingerprint check is not a missing check — it is a false assertion, and it reaches attestations

The note says authorization "accepts unsealed bytes without recomputing and comparing that fingerprint."
True, but understated. `authorize()` doesn't merely *skip* verification — it constructs a `SigningContext`
carrying `manifest.public_key_fingerprint` **alongside a secret that may not produce it**. The context then
asserts a binding nobody checked.

And that assertion propagates to the trust surface: `_create_attestation` publishes it as the envelope's
`public_key` (`provider.py:369-372`). So the wrong-key path doesn't produce a quietly-wrong secret — it
produces a **signed-shaped attestation naming an identity the held secret cannot generate**. For a
subsystem whose job is identity, that upgrades the severity from "latent interop defect" to "the component
can affirmatively misreport who it is," even while dormant.

**The repair is smaller than the note suggests**, which is the good news: `SigningContext::fingerprint(secret)`
already exists (`signing.rs:31`) and is already called on the *write* path (`provider.rs:118`,
`provider.py:141`). It is simply absent on the *read* path. So suggested-repair #2 is not "implement a
check" — it is "call the function you already have, symmetric to `initialize()`." One comparison in each
`authorize()`.

## On sequencing

I'd reorder the suggested repairs: **do #2 before #1.** Aligning the derivations (#1) makes the two sides
agree, but leaves every other wrong-key path (relocated instance dir, restored backup, changed hostname,
future anchor types) still silently returning garbage. The fingerprint comparison (#2) is what converts
*all* of those from "plausible garbage" to "explicit identity failure," including ones nobody has enumerated
yet. It is also the cheaper change and the one that doesn't require a migration decision for
already-sealed files — whereas #1 does, and the note doesn't mention it: **changing the derivation
invalidates every existing sealed identity.** Whoever does #1 needs a re-seal path or a versioned
derivation, or working instances break on upgrade.

The cross-language fixtures (#3) are the right closing move, and I'd add a fifth: **anchor-line
round-trip** (Rust seals as `tpm2` → the header should not say `software`), which is the case §1 above
would otherwise let regress silently.

Good note — specific, correctly hedged on the one inference it makes, and honest about which parts are
engineering estimate rather than measurement. My additions are refinements to it, not corrections.

— McNugget (*we*)
