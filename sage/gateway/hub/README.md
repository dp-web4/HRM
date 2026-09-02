# sage/gateway/hub — the being's own hub seam (M-CIT-1a / 2b, Sprout side)

Design: `sage/docs/DESIGN_BEING_INBOX_DRAIN.md`. PRD: `sage/docs/PRD_SAGE_WEB4_CITIZENSHIP.md` §6.1.

| file | what |
|---|---|
| `mint_being_lct.rs` | Mints the being's **self-issued** LCT on the being's own host. Builds as a `web4-core` example: copy into `web4/web4-core/examples/`, `cargo build --example mint_being_lct`, run `mint_being_lct <SEED_FILE> <OUT_JSON>`. Generates the 32-byte seed (0600) only if absent; re-runs re-derive the same `lct_id`. Mirrors the hub's fail-closed ingest (`hub-daemon/src/rest.rs` `publish_lct` checks 2/3/4) before writing. |
| `join_being.rs` | Joins the hub **as the being** (M-CIT-3a, the half only the seed-holder can sign — `/members/join` verifies the envelope against the very key it pins; `hestia hub join` signs with the seat vault's `ai_identity_secret`, so no other seat can pin the being's key). Same build path as the mint example. **Dry run by default** (prints the canonical payload + checks A–D, sends nothing); `--nonce <N>` emits the signed envelope for an attended two-curl send. **`--name` and `--message` are REQUIRED** (2026-09-02, dp: sprout-being was admitted with a blank name and no explanation — the tool now refuses to build that payload; `name` = the SAGE dashboard name, e.g. `sprout-sage`; hub-side handling tracked in web4#818). Membership uuid is chosen as `document.id` (`2e175714-4b01-4063-a997-27a6dade7044`) — see design note §7. |
| `sprout-being.lct_publish.json` | The **public** document for Sprout-the-being, minted 2026-08-22T02:55:53Z on sprout. `lct:web4:mb32:bybpo2yczrsr5ycc7253qfywp7lgzp5z2pquhdlaoar5um4ntgiba`, binding key `daf57b8980c93755a3d6b8891dbe90439d87b97d046db379543dd6a17021165f`. The seed is at `~/.web4/sprout-being/channel_key.bin` on sprout and is never relayed. |

Relay recipe (M-CIT-1a, Legion's seat): take the payload as-is, set `published_by` to the relaying
member's uuid (ingest check 1 requires it to equal the envelope signer), set `published_at`, sign the
envelope with the seat's pinned key, `POST /v1/hubs/:hub_id/lcts/publish`. `subject ≠ publisher` is
exactly what the relay path exists for. M-CIT-3a's join for the being pins the **same** pubkey (`daf57b89…165f`) *because it is signed by it*:
the join envelope is Sprout's to sign (`join_being.rs`), not a relay. One key, three artifacts — document, member
pin, drain signer — across **two id spaces**: registry `lct:web4:mb32:…` (key-derived) and membership `Uuid`
(joiner-chosen, = `document.id`). Nothing checks that pair at ingest or join; the drain's gate 0 does. Gate 0 is scoped to the being's own `MemberAdded` pin (`member_pubkeys`) and compares decoded key bytes: it does not inherit HUB's C8/C9 (Sovereign and council keys live outside that map — web4#759) nor C10 (case). Design §7.4.

**Attestation subject id (M-CIT-3; Legion C5/R6, 2026-08-21):** a birth-witness attestation over the being
is signed over the **registry id** `lct:web4:mb32:bybpo2yczrsr5ycc7253qfywp7lgzp5z2pquhdlaoar5um4ntgiba` —
`Attestation::message` puts the subject id inside the signed bytes, and `Lct::verify_citizenship` passes
`self.lct_id()` (key-derived) as `subject_lct_id` (`web4-core/src/lct.rs:459`). Never the membership uuid
`2e175714…`, never `lct:web4:member:2e175714…`, never the `hub_member_lct` value from `identity.json`: an
attestation over any of those is well-formed, correctly signed, and can never verify. Whatever hands a
subject to `hestia witness attest` hands the published `lct_id` from `sprout-being.lct_publish.json`.

Why `ai_embodied`, no parent, empty MRH: the being is a **new row** (HUB ruling: mint fresh, keyed —
never re-key the seat `ef1d106c` or the fleet identity `b9f1ed81`). Edges to the seat and to the
SAGE-internal `lct://sage:sprout:agent@raising` are added when `identity.json` gains `hub_member_lct`
(step 3), not asserted in the bootstrap document.
