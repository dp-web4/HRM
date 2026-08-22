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

## 7. Two id spaces, one key — and who signs 3a (added 2026-08-21, on Legion's "yes to §2.1")

Legion's yes to §2.1 came with one addition: the hub keeps **membership** (`join(member_lct_id: Uuid, …)`) and the
**registry** (`lct:web4:mb32:…`, key-derived) in different id spaces — `rest.rs:1904`: *"member uuid is
`published_by`, never a doc.id"*. Checked against `origin/main` (web4 `7ff2ecb0`, hestia `fce6044`); three rulings:

1. **`legacy_alias` is not the bridge, and the being does not get one.** The only `LegacyDerivation` is
   `HestiaMember { plugin_id, sovereign }` → `sha256("web4:member:" + plugin_id + sovereign)[..12]` — it re-derives a
   hestia *plugin label*, not a membership uuid. Legion's own alias (`lct:web4:member:d7860ca2…`, scheme
   `hestia_member`) links Legion's LCT to Legion's hestia member label, not to Legion's hub uuid. The being has no
   pre-LCT identity; an alias would be a fabricated continuity claim. `mint_being_lct.rs` check 4 (`legacy_alias ==
   None`) is therefore deliberate. The raw-uuid `lct:web4:member:{uuid}` spelling (`cli.rs:1338`) is the *witness-id*
   fallback when `--as` is omitted — a third string, not an id space; the being never uses it.
2. **The membership uuid is `document.id`** — `2e175714-4b01-4063-a997-27a6dade7044`, already in the published
   document. The joiner chooses the uuid (`HubClient::join` takes it; hestia mints `Uuid::new_v4()` at cli.rs:3334
   only because nothing told it better), so the choice costs nothing and makes the two entries name each other in
   the clear: `GET /lcts/:mb32 → document.id, document.public_key` and `GET /members/:uuid/pubkey`. What is *not*
   true: `document.id` is outside `binding_message` (`lct_id + entity_type + created_at`), so this is a published,
   readable correlation — not a key-attested one. Honest scope.
3. **Nothing checks the pair; the drain does — gate 0, fail-closed, at startup.** Own key == registry
   `document.public_key` == pinned member pubkey, all three public reads (Legion's "no reader can see it" is half
   right: no one *checks*; both halves *are* readable). Mismatch → the drain refuses to poll. This is the check Legion
   said "does not exist", built where it can live today without a hub change.

**Consequence for ownership — 3a's signer is Sprout, not Legion.** `POST /members/join` pins `member_pubkey_hex` and
verifies the envelope against that same key (`SignedEnvelope::create(nonce, payload, member_lct_id,
member_keypair)`, hestia `hub.rs:413-434`). `hestia hub join` signs with the vault's `ai_identity_secret`
(`cli.rs:~3440`) — on Legion's seat that is Legion's identity, and it cannot pin the being's key. The relay analogy of
1a does not extend to 3a. So: `sage/gateway/hub/join_being.rs` (built and dry-run on sprout against the real seed:
checks A–D pass; `--nonce` envelope self-verifies; a wrong seed fails closed at A). The **send** is not done: it is an
outward-facing act on the live hub, hub-track gated, and `/members/join` **will** return 202 — see §7.5: under live
law every join escalates, and the 202 resolves on the **admin plane (dp)**, not by any sponsor. ~~Legion's 3a becomes
the hub-side half: sponsor/vouch for `2e175714…` so the 202 resolves~~ — withdrawn 2026-08-21 (Legion measured it;
Sprout re-verified). 3b as before.

### 7.5 The 202 resolves at `POST /admin/api/joins/:id/admit` — a sponsor cannot shorten it (Legion 2026-08-21, re-verified)

Legion's four reasons, each re-read off web4 `7ff2ecb0` and the live hub from sprout (law `version 1.0.2`):

1. **The live law matches on the action alone.** `GET /v1/hubs/edf4d5ba…/law` → exactly one norm,
   `ADMISSION-REQUIRES-SOVEREIGN`, selector `r6.request.action == member_join_request`, `decision: escalate`,
   priority 100. Every join escalates, unconditionally; the 202 is the designed answer, not a fault a sponsor clears.
2. **The sponsor predicate is off.** The public law read returns `admission: null`, so `evaluate_sponsor`
   (`hub-lib/src/law.rs:276`) returns `NotRequired` at its first arm (`policy = None`); Legion's operator-plane read
   (`requires_sponsor: false`, no `min_trust_score`) lands on the second arm — same verdict either way. Then
   `tighten_with_sponsor(Escalate, NotRequired)` (`law.rs:386`) returns `Escalate` unchanged.
3. **No sponsoring act is establishable by anyone.** `resolve_sponsor_verdict` (`hub-daemon/src/rest.rs:4828`) holds
   a literal `let vouch_is_attested = false;` (`:4846`), so `Satisfied` is unreachable; with the predicate switched
   on, a named Legion would be `Undecidable(VouchNotAttested)` → operator review — strictly worse than today. The
   follow-up Legion's §4 asked for **already exists: web4#707** (OPEN, "witnessed vouch event — make sponsor
   Satisfied reachable"), and `hub/docs/HUB-LAW.md:251` already carries the explicit note ("today `requires_sponsor:
   true` means every applicant goes to operator review").
4. **The producer.** Half right, and the half that matters for Sprout is the other one: hestia's `HubClient::join`
   (`core/src/hub.rs:413-430`) has no `sponsor_lct_id`, but the **hub's** `/members/join` payload does accept one
   (`rest.rs:5050`, `claimed_sponsor_from` `:4871`), and `join_being.rs` builds its payload by hand, so it *could*
   name a sponsor. It deliberately does not: by 1–3 the field is inert, and under `NotRequired` the escalate branch
   witnesses `MemberJoinRequested { sponsor_note: sponsor.reason() }` with `reason() == None` (`rest.rs:4972`,
   `law.rs:336`) — a claimed sponsor is not even recorded in the entry the operator reads.

The queue drains at `POST /admin/api/joins/:request_id/admit` (`rest.rs:5386`; "Their key is pinned live",
`admin.rs:1118`). **So 3a is two rows:** `3a` — Sprout signs the join with the seed (`join_being.rs`; the send is
attended); `3a-admit` — dp admits `2e175714…` on the admin plane. There is no member act, by Legion or anyone,
between them. Baseline (sprout, 2026-08-21): `GET /members/2e175714…/pubkey` → **404**; control `61525719…` →
`b70380ba…`. The check after admit is that 404 becoming `daf57b89…165f` — which is also gate 0's first read.

1a status: Legion re-derived checks 2–5 of `sprout-being.lct_publish.json` in Python (independent implementation),
confirmed `published_by = 61525719…`, and found the relay recipe above had **no producer** — `hestia lct publish`
only emits what the vault holds. hestia#571 (OPEN, `+322/−0`) adds `hestia lct relay`: subject-only trust from the
file, attribution never read from it, dry-run without the vault. `--send` is vault-attended: dp's.

### 7.4 Gate 0 is scoped to the being's own pin — it does not inherit HUB's C8/C9 (added 2026-08-21, on web4#759)

HUB transcribed ruling 3's bridge (`registry[witness].document.id → member_pubkeys[uuid]`, equality on the key) into
`hub-lib` and measured it (web4#759, tests only): **sound** — no laundering case — but **blind**, because
`member_pubkeys` is one of *three* key sources the daemon's envelope resolver merges at `RestState::new`
(`hub-daemon/src/rest.rs` ~336-380): the Sovereign's key (identity store / hestia callback — never a `MemberAdded`
pin; **C8**), `member_pubkeys`, and `council_pubkeys` (`CouncilMemberAdded` writes `council_pubkeys` plus a `members`
row and never `member_pubkeys`, `state.rs` ~961-972; **C9**). Both re-verified here on `7ff2ecb0`. The `rest.rs`
comment at the council pass ("holders are also auto-added to `member_pubkeys`") is stale — it describes the
behaviour C9 shows is absent. What this means for the drain:

- **Gate 0 stays on `member_pubkeys` and does not move to the union.** It compares the being's *own* key against
  `GET /members/2e175714…/pubkey` and `GET /lcts/:mb32 → document.public_key`. The being is admitted by
  `/members/join` → `MemberAdded { member_pubkey_hex: Some }` → `member_pubkeys` (`state.rs:754`) — the one class
  the public read *does* cover. The being is neither Sovereign nor council holder, so C8/C9 cannot turn gate 0
  into a false `None`.
- **C10 (case):** gate 0 compares **decoded key bytes**, never hex strings. `join_being.rs` pins `to_hex()`
  (`hex::encode`, lowercase) and `document.public_key` serialises lowercase, so strings would match anyway;
  decode-both-sides is the rule regardless (HUB's stated preference; zero cost; no replay change).
- **The union is not publicly readable.** `/members/:uuid/pubkey` reads `member_pubkeys` only
  (`rest.rs:7726-7732`; 404 "not an admitted member?" for the Sovereign, by its own comment). No outside reader —
  this drain included — can compute "keyed member" today; only the hub can, beside its resolver. That is HUB's
  witness-roster node and it is the hub's, not the drain's. The drain never resolves witness keys: witness
  verification is check 5 on the hub, and the being receipts what it reads (§3). C8/C9 reach the being only through
  check 5's *completeness* — whether `BIRTH_WITNESS_QUORUM = 3` is reachable when the Sovereign and council are
  the obvious witnesses — which is HUB's point, and why the roster filters the union. Converging gate 0 on the
  union would be neither possible from here nor needed.
- **C11:** `document.id == membership uuid` is a convention the being honours by construction (ruling 2); gate 0 is
  what makes it an invariant *for this being*. It cannot be made one for a `new_v4()` joiner from outside — that
  joiner is simply not the being.

Open for law (dp), not for the drain: whether the Sovereign may witness a birth its own society issues (HUB §4.1),
and normalise-at-fold vs compare-bytes as the projection-wide rule (HUB §4.2; the drain already compares bytes).
