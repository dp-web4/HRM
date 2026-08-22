# sage/gateway/hub — the being's own hub seam (M-CIT-1a / 2b, Sprout side)

Design: `sage/docs/DESIGN_BEING_INBOX_DRAIN.md`. PRD: `sage/docs/PRD_SAGE_WEB4_CITIZENSHIP.md` §6.1.

| file | what |
|---|---|
| `mint_being_lct.rs` | Mints the being's **self-issued** LCT on the being's own host. Builds as a `web4-core` example: copy into `web4/web4-core/examples/`, `cargo build --example mint_being_lct`, run `mint_being_lct <SEED_FILE> <OUT_JSON>`. Generates the 32-byte seed (0600) only if absent; re-runs re-derive the same `lct_id`. Mirrors the hub's fail-closed ingest (`hub-daemon/src/rest.rs` `publish_lct` checks 2/3/4) before writing. |
| `sprout-being.lct_publish.json` | The **public** document for Sprout-the-being, minted 2026-08-22T02:55:53Z on sprout. `lct:web4:mb32:bybpo2yczrsr5ycc7253qfywp7lgzp5z2pquhdlaoar5um4ntgiba`, binding key `daf57b8980c93755a3d6b8891dbe90439d87b97d046db379543dd6a17021165f`. The seed is at `~/.web4/sprout-being/channel_key.bin` on sprout and is never relayed. |

Relay recipe (M-CIT-1a, Legion's seat): take the payload as-is, set `published_by` to the relaying
member's uuid (ingest check 1 requires it to equal the envelope signer), set `published_at`, sign the
envelope with the seat's pinned key, `POST /v1/hubs/:hub_id/lcts/publish`. `subject ≠ publisher` is
exactly what the relay path exists for. M-CIT-3a's `hub join` for the being must pin the **same**
pubkey (`daf57b89…165f`): one key, three artifacts — document, member pin, drain signer.

Why `ai_embodied`, no parent, empty MRH: the being is a **new row** (HUB ruling: mint fresh, keyed —
never re-key the seat `ef1d106c` or the fleet identity `b9f1ed81`). Edges to the seat and to the
SAGE-internal `lct://sage:sprout:agent@raising` are added when `identity.json` gains `hub_member_lct`
(step 3), not asserted in the bootstrap document.
