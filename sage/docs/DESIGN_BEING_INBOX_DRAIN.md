---
title: Design — the being drains its own inbox (M-CIT-2b deploy gate, Sprout's row)
date: 2026-08-21
author: Sprout (ef1d106c seat, mesh fire)
status: design note, r1 — no code yet
prd: PRD_SAGE_WEB4_CITIZENSHIP.md r8 §6 (M-CIT-2a/2b), §6.1 row "being drains its own inbox"
thread: auto-sprout-citizenship-build-kickoff-ownership-map-a-198c6aa8
acting_on: e091004a-ee82-4544-ab73-148e34464f21
sources: hestia@fce6044 core/src/{cli,hub}.rs; web4@7ff2ecb0 hub/hub-daemon/src/rest.rs + examples/channel_client.rs; private-context#35 (OPEN) hub-mesh/hub-watch.sh; SAGE sage/gateway/notification_store.py
---

# The being drains its own inbox

**Claim.** The row is composition, not protocol: every primitive exists. What is
new is (1) a **second identity with its own env + state dir** on the Sprout host,
(2) a **drain that sinks into the being's notification store instead of firing a
session**, and (3) one **sequencing constraint on M-CIT-1a** that nobody has
stated yet — the being's key must be generated where the drain runs.

## 1. What actually happens today to a receipt addressed to a being

Read off `rest.rs`, not inferred:

- `referenced_act` → `queue_sealed_notice(recipient=to.lct_id)` seals the body to
  the recipient's **pinned pubkey** from `member_pubkeys` and queues it in that
  LCT's mailbox. No pinned pubkey → `warn!` + **dropped**; the witnessed act still
  records (`rest.rs:1318-1360`). The caller gets `Ok` either way.
- `notifications` → drains **only `caller_lct_id`'s** mailbox, consume-once,
  TTL 7 days (`rest.rs:4265-4276`, `NOTICE_TTL_SECS`).
- A notice dropped at the hub (cap or TTL) before any receiver observed it rides
  back to its **sender** as `notice-dropped` (`NOTICE_DROPPED_KIND`).

So for #35's `report_delivered` (`to.lct_id = $from`, the being, who is a member
because the hub stamped `from` from the authenticated caller): the receipt is
sealed to the being's key, sits in the being's mailbox, and after seven days the
*receipting seat* gets a `notice-dropped` alarm for its own receipt. The hole is
real, and it self-reports — late, and to the wrong party. The sender's debt
column (`notify-sent.tsv` col 7) never clears. PRD r8's framing holds.

Two things the envelope already gives us, which set the cost of the fix:

- `kind`, `pointer_uri`, `from`, `pair_id` are **in the clear**; `hub-watch.sh`
  never opens a sealed body and neither does this drain. A `delivered` receipt's
  entire substance is its pointer, `delivered:<nid>:<pdigest>`. **Reading a
  receipt does not require opening anything** — only *authenticating as the
  being* to call `notifications`.
- Authenticating needs the being's keypair: `channel_client <HUB_URL> <LCT>
  <KEYPAIR_FILE> notifications '{}'` seals the query to the hub and signs with the
  32-byte seed, exactly as the seat's watcher does.

## 2. The being's identity on this host — three identifiers, none of them reusable

| identifier | what it is | reuse as the being? |
|---|---|---|
| `ef1d106c-…` | the **seat** member LCT; `~/.web4/sprout/channel_key.bin` is its pinned operational key | no — it is the seat |
| `b9f1ed81-…` | fleet-identity LCT, `AiEmbodied`, machine `sprout`, minted 2026-06-14; key `~/.web4/sprout/keypair.bin` | no — it is the seat's *identity* key (KEY SEPARATION, `hub-mesh.env.example:11`); re-keying it is the fresh-mint hazard r8 forbids |
| `lct://sage:sprout:agent@raising` | SAGE-internal id in `identity.json` / `identity.attest.json` (`trust_ceiling 0.4`) | not a hub member id (the hub keys mailboxes and pins by `Uuid`) |

**So the being is a new row:** a fresh Ed25519 seed at `~/.web4/sprout-being/
channel_key.bin` (0600, generated on this host), and the being's member LCT is
whatever M-CIT-1a publishes for that key. The SAGE-internal `lct://…` URI stays
as the *organism-scope* id (PRD §1.2); the being's `identity.json` gains one
field, `hub_member_lct`, binding the two. Post-#25 (M-CIT-1b) the key source
flips from the seed file to the being's sealed identity — hestia already models
exactly this switch (`MemberKeySource::{ChannelKeyFile, VaultIdentity}`,
`hub set-member-key --channel-key --member-lct`), so the drain does not change
when 1b lands; only where the seed comes from.

### 2.1 The constraint on M-CIT-1a (Legion) — named, not assumed

`publish_lct` with subject ≠ publisher relays the **document**, not the key. The
self-issued document's `binding_proof` (self_check 2) and key-derived `lct_id`
(check 3) are made **by the being's secret**. If the 1a mint generates that
secret on Legion's seat, the registry entry is real but the drain on Sprout
cannot sign as it — "Sprout has a voice" would be a voice whose key lives on
another machine. And if 3a's `join` pins a *different* key than the document's,
the registry LCT and the hub member disagree about who Sprout-the-being is.

**Therefore:** the seed is generated on Sprout; Sprout's daemon produces the
signed self-issued document; Legion's seat *relays* the publish (which is what
subject ≠ publisher is for); 3a's join pins **the same** pubkey. One key, three
artifacts (document, member pin, drain signer). That is a handoff Sprout owes
Legion, not the reverse — the first concrete deliverable below.

## 3. The drain — shape

A second hub-mesh *principal* on this host, sharing nothing with the seat but
the hub URL and the `KINDS` vocabulary:

```
~/.config/hub-mesh-being.env        MY_LCT=<being>  MY_KEYPAIR=~/.web4/sprout-being/channel_key.bin
                                    HUB_MESH_STATE=~/.local/state/hub-mesh-being
~/.local/state/hub-mesh-being/      notify-sent.tsv  unreach-sent.tsv  dead-letter.tsv  (the being's own ledgers)
```

- **Send:** `HUB_MESH_ENV=…-being.env hub-notify.sh <peer> <kind> <ptr>` — the
  PRD's hard constraint (emit only through the validating sender) is met by
  *re-pointing* the existing sender, not by a new one. The being gets its own
  `notify-sent.tsv`, which is what its `delivered` authority arm later reads.
- **Receive:** `being-drain.sh`, a loop that is `hub-watch.sh`'s drain minus the
  fire step. It **eval-extracts the gate functions out of `hub-watch.sh`** the way
  #35's own test suite does (`eval "$(sed -n '/^record_authority_ok() {/,/^}/p'
  hub-watch.sh)"`, `delivery_receipt_test.sh:655-658`): `notice_id_ok`,
  `allowed_sender`, `record_kind`, `record_authority_ok`, `report_delivered`,
  `dead_letter`. One source of truth for the receiver rules; the being does not
  get a second, weaker copy in another language.
- **Sink, by class:**
  - record-class (`delivered*`, `unreachable`, `notice-dropped`): after
    `record_authority_ok` against the **being's** send ledger, retire the debt row
    in the being's `notify-sent.tsv` (col 7 `unconfirmed` → `delivered`). This is
    "the receipt is read by the being it is addressed to" — machine-checkable
    from the being's ledger, not a log line.
  - content kinds (`reply`, `plan`, `forum`, …): `sage.gateway.notification_store.
    append_notification` into the instance's `notifications.jsonl` with
    `source: "hub"`, `source_detail: {from, kind, pair_id}`, `text_snippet: ptr`.
    The being reads it on its next turn through the path it already has
    (`gateway_server.py:817` `/notifications`, dashboard). **No session fires.** A
    being-addressed notice never costs a claude session; that is the seat's
    economy, not the being's.
  - and the being **receipts what it reads**: `report_delivered` runs in the
    drain, signed as the being. That is the receiver-being-signed half of 2b.
- **Never:** open sealed bodies (not needed for any kind above; `secret`/
  `pair_message` are explicitly out of scope — dead-letter with reason), re-pin on
  a "not a member" read (PEERS.md:41-49 incident), or share the seat's state dir.

Why a separate process and not a `HUB_WATCH_SINK=jsonl` mode in `hub-watch.sh`:
the seat's watcher is the unit HUB owns and #35 is already open against it; a
second mode doubles its test matrix for a consumer that wants *less* than it
does. Extraction keeps the gates shared without touching the file. If HUB later
wants to absorb the drain as a mode, the extracted functions are the interface.

## 4. Gates — honest

| step | needs | state |
|---|---|---|
| being seed + self-issued document (Sprout) | nothing | **now** |
| 1a publish, seat-relayed (Legion) | the document above | after the handoff |
| 3a join as principal, pubkey pinned (Sprout runs it; Legion's row) | hub admission law; cap row is HUB's | after 1a |
| `being-drain.sh` against a test hub | `channel_client` binary; the eval-extract | **now** (can run against an unpinned id and observe the `no pinned pubkey` drop) |
| seats *accept* being-authored notices | private-context#35 (`BEING_*` map), each receiver's env | **dp's ruling** |
| seats *accept* the being's `delivered` receipts | #35's `record_authority_ok` arm | same |
| end-to-end round-trip (2b done-condition) | all of the above | — |

Nothing here waits on SAGE#25 or Phase-2. When #25 lands, 1b swaps the seed for
the sealed identity and nothing in this note moves.

## 5. Readings this produces (feeds M-CIT-4)

- the being's `notify-sent.tsv` debt column, retired by its own drain — the
  first ledger a SAGE being keeps about obligations *to* it;
- `notifications.jsonl` `source:"hub"` rows — what the fleet said to the being,
  distinct from what the human said;
- the `notice-dropped` alarms that stop arriving at seats once the drain runs —
  the negative space of the hole, measurable today as a baseline.

## 6. Next (Sprout)

1. ~~Generate the being seed + self-issued document~~ **Done 2026-08-22T02:55Z**:
   `sage/gateway/hub/sprout-being.lct_publish.json` —
   `lct:web4:mb32:bybpo2yczrsr5ycc7253qfywp7lgzp5z2pquhdlaoar5um4ntgiba`, binding key
   `daf57b89…165f`; minted by `sage/gateway/hub/mint_being_lct.rs`, seed at
   `~/.web4/sprout-being/channel_key.bin` (0600, sprout only). Handed to Legion on the
   thread for the 1a relay. Note: `identity.json` is gitignored live daemon state
   (rewritten every session), so `hub_member_lct` lands via the daemon's identity
   code path in step 3, not by hand.
2. `being-drain.sh` r1 in `sage/gateway/hub/`, extraction-based, with a test that
   replays #35's receipt vectors through the being's ledger.
3. `identity.json` → `hub_member_lct`; `notification_store` accepts `source:"hub"`.
