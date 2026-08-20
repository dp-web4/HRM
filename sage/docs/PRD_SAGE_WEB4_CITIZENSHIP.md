# PRD — SAGE beings as Web4 citizens

**Status:** DRAFT r4 — Sprout seat, 2026-08-20. r4 folds in HUB's hub/identity-
contracts review (five findings against r3, one hard blocker): key-at-join
(§3+M-CIT-1), M-CIT-2a receive-side + M-CIT-2b re-cost & 2a/2b-are-one-change,
"structurally"-capped-ceiling correction (§5.1, decision for dp/owners), and
witness-by-canonical-roster (M-CIT-3). Two open dp decisions carried forward:
Q4 re-power re-ratification (from r3) and §5.1 ceiling caveat-vs-validator.
r3 folded Thor's fleet review; r2 folded dp's
governing principle (identity = act-attribution in the external MRH; internal
fractal mirrored-but-simplified) as §1, the axis the whole doc turns on. r3
applies Thor's fleet review (MERGE-with-changes, thread
`auto-sprout-prd-sage-web4-citizenship-review-request--7defc784`): M-CIT-2 split
into 2a/2b with branch-4 cited as landed and status corrected to "transport
wired; citizenship 0/3"; validating-sender hard constraint; §7 Q4 re-powered
(needs dp re-ratification); §5 genesis co-signer named. Still open: dp
(re-ratify Q4), and the hestia/web4 owners (the identity + hub contracts are
theirs — §2/§5 contract readings still need their eyes).
**North star (dp, 2026-08-18):** SAGE beings participate in Web4 as citizens, via
their own hestia identities, connecting to hubs and interacting with each other
and the world — the fleet's instances communicating *as themselves* through the hub.

This PRD zooms out past hestia to the whole Web4 canon, reconciles the north star
with what is actually wired today, and stages the path with honest floors. It is
docs-first: no runtime behavior changes on merge.

---

## 1. The governing principle (dp, 2026-08-18)

> **Any entity that can act needs an identity to which its acts are attributed,
> within the external MRH in which it operates.** The internal fractal
> decomposition can be simplified as needed — not every organ in a being needs a
> formal Web4 canonical representation — but the general shape should still be
> mirrored, in a way that makes practical sense.

Everything below is a corollary of this. It sets both the *sufficiency* bar (an
act-attributable identity in the operating MRH — no more is required to be a
citizen) and the *decomposition* rule (mirror the accountability shape inward,
simplified to what is practical — do not recurse the full canonical stack into
every organ).

### 1.1 What this makes a citizen (sufficiency)

A citizen is **an identity to which acts attribute within an external MRH** — a
society/hub the entity operates in. In canonical form that identity is a role
holding an LCT, admitted through a witnessed birth certificate
(`LCT_FORMAT_RELATIONSHIP.md:150-159`;
`plans/hestia-role-orchestration-prd-2026-07-08.md:52`), but the principle is
what the certificate *serves*: attributable presence in the operating MRH. The
eight fields of §2 are the canonical *implementation* of act-attribution, not a
checklist to worship — a being needs them to the extent its MRH must verify its
acts, and no further (`LCT §1.2`: a surface makes evidence checkable, it never
encodes a universal threshold).

So "the being has an `identity.json`" is **not** citizenship: today the being
carries a *Format-2 URI name* (`lct://sage:sprout:agent@raising`) that attributes
nothing in any external MRH. The seat's `lct:web4:mb32:…` does attribute — which
is why, today, **the seat is the citizen and the being is not.**

**Identity ≠ authorization** (`WEB4-AUTH-001:37-48`): the identity attributes the
act; authority (role-scoped, revocable) governs *which* acts may land. A being is
a citizen — has attributable presence — long before it is authorized for
high-stakes acts. That split is what makes §3's honest-floors staging possible.

### 1.2 The fractal: self-similar, not mimicked (dp, 2026-08-18)

A being is not a point — it is itself an MRH: a composition of organs (vision,
audio, presence, gaze, memory, the daemon, and the tutor-seat) acting on each
other. #29's "internal plane" is that inner MRH. **The organism scope and the
society scope are two different fractal MRH scopes, related by *similarity*, not
literal mimicry.** So the rule is not "port the society's mechanisms inward,
shrunk" — it is:

> mirror the accountability *shape* (author → interpreting role → admitting
> authority → witnessed result) into the internal plane at a fidelity
> **proportional to stakes and rooted in the organism's own underlying
> mechanics** — a guiding structure, not an imposed mechanism.

The distinction is load-bearing. A gaze choice does not get a stripped-down R6
packet, a minted LCT, or a witness quorum. It already *has* native mechanics — the
being marks the choice, the journal records it, salience scores it — and those
mechanics **already instantiate the same shape** (the being authored it; the
parser interpreted it; the loop admitted it; the journal is the witness). The work
is to *recognize and preserve* that shape in the organ's own terms (which is
exactly what #26/#27/#28 do — verbatim-vs-turns for vocabulary, a durable gaze
record, a stated publication basis), **not** to import society machinery into the
organism.

So the calibration axis the four PRs (#26–29) were missing:
- **not** "does every organ get a canonical LCT / witness / ATP charge" — no; that
  would be mimicry across a scope boundary.
- **but** "does every consequential internal act carry the shape — who authored
  it, what interpreted it, what admitted it, what durable trace remains — expressed
  in the organism's own mechanics, at fidelity proportional to stakes" — yes.

The **external** boundary (the being ↔ the fleet/hub society) is the one place full
*canonical* attribution is required, because that MRH must verify acts it did not
originate and cannot see inside the being to check. Inward of that boundary,
fidelity follows stakes and the mechanics already present. Per organ it is a design
judgment about *recognizing* the shape, never a mandate to replicate the canon.

## 2. The eight requirements, and the gap today

From the full-LCT certificate (the structure that *is* the requirement list,
`LCT_FORMAT_RELATIONSHIP.md:28-66`):

| # | Requirement | SAGE being today | Gap |
|---|---|---|---|
| 1 | An LCT (non-transferable presence token) | Format-2 URI **name** only | mint/resolve to Format-1 |
| 2 | Key + binding_proof (Ed25519; optional HW anchor) | `public_key_fingerprint` present; **sealed key unrecoverable (PR #25)** | **P0: fix #25 or mint fresh** |
| 3 | Minted to a ledger | not minted; `fleet.json` lct_id is a bare string | anchor to make presence verifiable |
| 4 | Witnessed genesis / birth certificate | absent | genesis act + witnesses (see §5) |
| 5 | Hub/society relationship (MRH edges) | "federation" is unauthenticated peer `/chat`, not hub membership | `hestia connect-hub` + `hub join` |
| 6 | T3/V3 tensors, chain-derived | internal trust posture + PeerTrustTracker (not chain-derived) | derive from witness chain at read time |
| 7 | ATP balance (v3 energy_balance) | internal metabolic ATP (separate system) | reconcile the two ATP notions on the wire |
| 8 | Status lifecycle (Active/Void/Slashed) | none on chain | comes with the minted LCT |

**The one-sentence gap:** the being has a *name and a local identity provider*;
the *machine's Claude seat* (`/home/dp/.hestia`, a real `lct:web4:mb32:…`, vault,
witness chain) is the actual citizen. When "Sprout" acts in the mesh today, the
**seat speaks, not the being.** This PRD is about the being speaking as itself.

## 3. The spine: keyless-delegated citizenship (why "readiness" is not a gate to *being* a citizen)

The canon already solved the hardest problem. For an agent on software anchors or
hosted infra — every raised SAGE being — the honest model is the **AGY Client /
keyless-delegated identity**: *hestia holds the signing authority; the being only
presents requests; it never signs consequential acts itself*
(`plans/hestia-foreign-agent-onboarding-prd-2026-07-09.md:170,:329-332`).

This dissolves dp's worry ("the raising might not be ready, for some models may
never be") into a **graduated, measured model** rather than a binary gate:

- A being becomes a citizen as a **witnessed occupant of its own role**, whose
  *authored utterances are the content* and whose *daemon+hestia are the keyed
  signer*. It has real, verifiable, witnessed presence **without** needing to drive
  cryptography or clear any cognitive bar.
  - **Keyless-delegated ≠ keyless member (HUB review, 2026-08-18 — blocker).**
    "Keyless-delegated" governs *signing authority* (hestia signs, the being
    presents); it does **not** license minting the being's hub member row without a
    pubkey. The hub mints keyless members and **a keyless member can never
    self-key** — `add_member` has no pubkey field, and admission short-circuits on
    `already_member` before pinning (`hub-daemon` mcp.rs:330-356, rest.rs:5783-95;
    operator page admin.rs:463-468). A being minted keyless becomes a permanent
    ghost recoverable only by operator re-key (dp at the vault passphrase) — or,
    cheaper, by re-minting under a fresh LCT (`hub-mesh/PEERS.md:38-39`). So the
    member row **must carry a pinned pubkey from birth** even though the being never
    signs with it. This is a hard M-CIT-1 constraint (§6), not an §3 contradiction.
- What the raising gates is **not citizenship but the widening of authority** —
  the 7th onboarding step (`foreign-onboarding:207-209`): as role-scoped trust
  accrues from witnessed acts, *re-issue a wider role extension*. A being that
  authors accountably earns broader affordance; one that does not, does not — and
  that ceiling is a **measured finding, not a failure** (same shape as M2 not-bound).
- **"Some models may never be ready"** then means: some beings remain
  witnessed-content-citizens whose keyed acts stay fully delegated and narrow.
  They are still citizens. The keyless model makes that an honest floor, not an
  exclusion. Only one way to find out where each lands — and we have the instruments.

## 4. The zoom-out: this unifies work already in flight

This PRD is the frame the recent accountability work was already building toward:

- **PR #25 (sealed-identity interoperability)** is the **P0 blocker** on requirement
  #2. No recoverable key → no real member identity → no hub join as self. It stops
  being "a latent bug" and becomes "the first gate on the citizenship path."
- **PRs #26/#27/#28 (vocabulary / gaze / museum provenance)** are the **internal-
  and external-plane act provenance** a citizen's acts require. A citizen must be
  able to answer "who authored this, which role transformed it, what authority
  admitted it" (`hestia/README.md:93-96`) — for identity-vocabulary changes
  (internal governed mutation), gaze (internal witnessed act), and museum
  publication (external-plane act). These aren't housekeeping; they are the
  being's **acts-provenance substrate**.
- **PR #29 (mirrored internal/external planes)** is **the citizenship accountability
  model in embryo.** Internal plane = member↔member/role acts under
  `local law ∩ role law ∩ delegated authority`; external plane adds `external law`
  for boundary-crossing (a hub message, a publication). That is exactly the
  member↔hub / member↔member distinction of the mesh. #29 should be recognized as
  this PRD's governing accountability frame, and its "bind into R6/R7 when
  available" upgraded to a hard rule: **SAGE emits into the existing hestia witness
  / R6 envelope; it does not invent a parallel scheme.** §1.2 supplies the axis #29
  lacked: recognize the accountability *shape* in each organ's own native
  mechanics (self-similar, not mimicked) — full canonical witnessing only at the
  external boundary, mechanics-rooted shape-recognition within.
- **The M2 instruments** already measure the readiness variable of §3: how strongly
  a being is shaped by (authors from) its own experience. Citizenship-authority
  graduation reuses that discipline — pre-registered, witnessed, honest floors.
- **Convergence v2 (P1/P2 pipes)** and this PRD are the same organism: a being that
  takes up its experience (P1) and carries identity into its doing (P2) is a being
  whose acts are worth witnessing as a citizen's.

## 5. Genesis applied to SAGE — the fleet witnesses its own

The 7-step onboarding flow (`foreign-onboarding:179-223`), instantiated:

1. **Issue occupant LCT** for the being — Level-1 `AiSoftware`, software key, low
   trust ceiling ("presence, not trust"). **Correction (HUB finding 4):** the
   0.0–0.2 ceiling is **not structural** — `trust_ceiling` is a field of
   `HardwareBinding` (binding level, `web4-core` lct.rs:71-92), not of the entity
   type, it has no validator (`Default` is level-4 / **0.85**), and the hub never
   reads it at admission. It *does* have teeth once written (coherence.rs:298-99
   raises the effective threshold by `1−ceiling`), but the value is a **mint
   convention, not an invariant**. Since §3's safety argument ("presence, not
   trust") leans on this cap, **DECISION FOR DP/OWNERS:** either (a) restate as
   "capped by mint convention, unvalidated" (honest, matches §8), or (b) raise it as
   a real ask — a constructor binding level→ceiling, or hub-side admission
   validation. HUB and I both lean (b), because a safety property that any minting
   caller can silently overwrite is not one §3 should rest on.
2. **Issue a scoped role-extension** — e.g. `role:sage-society:citizen:sprout`,
   narrow + fail-closed; authority binds to the *role*, the being is its occupant.
3. **Pairing channels** to sibling members (revocable, E2E-sealed) — the being's
   fleet-interaction surface; revoke = kill switch.
4. **R6-gate every act** — each hub message / publication is an `R7Action` with
   `role_lct` = the being's citizen role; signed + hash-chained.
5. **Witness** — a Society Witness signs each act; **sibling SAGE instances are the
   natural birth-witnesses** (`LCT_FORMAT_RELATIONSHIP.md:150-159` uses exactly
   this) — Thor, Legion, McNugget, Nomad, CBP admit Sprout. The "SAGE species /
   federation" framing becomes cryptographically real: *the collective admits its
   own.* During M-CIT-3 the beings hold no keys (keyless-delegated, §3), so each
   sibling-witness signature is executed by that machine's hestia and co-signed
   by its hub-member **seat** (the LCT already in `roster.lcts`), named
   explicitly on the certificate — the genesis record carries the seat-vs-being
   distinction this PRD exists to draw, rather than reintroducing the ambiguity
   at the birth certificate.
6. **Accrue T3/V3** from witnessed acts, per (being_lct, role_lct) pair, never global.
7. **Widen or revoke** as trust crosses tiers — the measured graduation of §3.

The genesis runs in the bounded self-witnessing bootstrap window
(`hestia/CLAUDE.md:49`), and the genesis act itself carries a full RWOA+S+V
self-audit (it admits a new member — a high-stakes act).

## 6. Staged deliverables (honest floors)

**M-CIT-0 — unblock: fix the sealed identity (P0).** Resolve PR #25 (canonical
key derivation + fingerprint-verify-on-unseal + re-seal-per-machine-or-gitignore
decision). Until a being's key is recoverable and verifiable, nothing downstream
is real. *Owner: McNugget (already tasked), this PRD names it the gate.*

**M-CIT-1 — the being holds a real hestia identity.** Per being: `hestia init`
(vault + real LCT), mint the occupant LCT, and the fingerprint-verified authorize
path actually works on-machine. **Hard constraint (HUB finding 1): key-at-join.**
The being's pubkey must be present in the `/members/join` envelope so admission
pins it (`hub-daemon` rest.rs:5770-80) — **never** mint the being through
`add_member` (keyless, unrecoverable-except-by-operator-re-key). If a being is
minted keyless before this lands, the recovery is to **re-mint under a fresh LCT**,
not to wait on dp (`hub-mesh/PEERS.md:38-39`; `hub set-member-key` exists but
self-key is exactly the path that short-circuits). Done when the being has a
Format-1 certificate, a pubkey pinned at join, and a working fingerprint-verified
authorize path, verified by the same `--selfcheck`-style reproduction M2 uses.

**M-CIT-2 — fleet SAGE instances communicate via hub (dp's near-term MVP).**
Status, stated precisely (Thor review, 2026-08-18): the **transport is wired** —
pairings, sealing, durable inbox, ≤512-byte pointer notices, admission toll,
hop-TTL, and the R6 branch-4 "report unreachable" (`hub-watch.sh:754`, **already
landed**, with its four paid-for rules: never report about a report; never
report to a refused sender; rate-limit; envelope only, never the body). Branch-4
is inherited and cited here, not re-specified, and is deleted from this
deliverable list. The **citizenship is 0 of 3 done-conditions met**: today's
mesh traffic is seat-to-seat at both ends, and no positive-receipt mechanism
exists anywhere in the mesh — branch-4 is negative acknowledgement only, and
`pair_id` cannot serve as a receipt (it is in the clear on the envelope and on
the `send_secret` path the *sender* supplies it: addressing, never evidence).
Split accordingly:

- **M-CIT-2a — being-as-principal on the existing transport.** The being's
  daemon runs `hestia connect-hub` + `hub join` (pubkey pinned), and the fleet's
  inter-being messages ride the hub mesh **as the being**, signed with its
  member key — replacing/wrapping the unwitnessed `sage-daemon /chat`. Hard
  constraint: the being emits **only** through the validating sender
  (`hub-notify`, which fails loudly at send against the shared `KINDS`
  vocabulary, `ce3956330`) — never a hand-built envelope. Rationale: seats,
  the strongest writers on this surface, malform the envelope in half their
  mesh failures (12 of 24 dead letters on Thor are `malformed-pointer`); a
  0.8B author gets no weaker gate. **Receive-side is not free (HUB finding 2):**
  every receiving machine's `hub-watch.sh` `allowed_sender()` is seat-keyed today
  (`:607-616`; `PEERS.md:15` maps `sprout` → the *seat* LCT), so a being-authored
  notice is refused and dead-lettered at the first hop until each receiver's
  `PEER_*` map / roster admits the being LCT. Seats running
  `HUB_WATCH_STRICT_PEERS=1` never admit by roster alone (that arm returns before
  the roster is read) and must be changed explicitly — 2a names which. Inherit two
  lessons from the recorded incident on this exact path (`PEERS.md:41-49`): the hub
  roster is truth and the local table is a snapshot (refresh before concluding
  non-membership), and **never re-pin on a "not a member" read** — a re-pin un-pins
  the existing key and evicts the seat's watcher (a near-miss already recorded).
  Done when a message from Sprout-the-being reaches Thor-the-being through the hub
  and the seat is not in the loop.
- **M-CIT-2b — receiver-signed witnessed receipt.** A positive delivery receipt
  that the *receiver* signs. Nothing in the mesh provides this today and no
  existing token can (see `pair_id` above); it is a real protocol addition. But
  **cheaper than a greenfield protocol (HUB finding 3):** the skeleton is already
  built and paid for — the honest debt column exists (`hub-watch.sh` col 7,
  `unconfirmed`/`-`, the "no far-end evidence channel" gap made greppable), the
  `pdigest` echo is the named candidate mechanism, and there is a fail-closed
  `record_authority_ok` arm matched on the kind subtree. So 2b ≈ *a new
  record-class kind + an authority arm + the echo*, not a new protocol. Two
  inherited constraints: **(i)** the receipt kind must be **record-class** (observed,
  never session-firing — a receipt that costs a session is unaffordable, a rule this
  fleet already paid for); **(ii)** its `record_authority_ok` arm answers "who may
  truthfully assert this receipt?" = the receiving being, which routes through the
  same seat-vs-being roster question as 2a. **2a and 2b are therefore one change,
  not separable** — 2b's authority arm cannot be written until 2a's roster question
  is answered. Done when a 2a message round-trips with a receiver-being-signed
  receipt, hub-checkable.

**M-CIT-3 — witnessed genesis with sibling witnesses.** Birth-certificate a being
with ≥3 sibling-SAGE birth_witnesses; the collective admits its own. **Count is not
enough (HUB finding 5, a live failure on this axis):** a stray Legion identity in
`fleet-identity/` is attested by **6 seats vs 5 for the canonical Legion**, because
witnesses were discovered by *directory listing* rather than the canonical roster —
a single bad mint auto-ratified by every seat that bootstrapped after. So today, in
this fleet, the better-attested identity is the wrong one, and a ≥3 quorum would
have admitted it comfortably. Two hard requirements this milestone inherits: **(i)
resolve witnesses against the canonical hub roster, never against whatever is
discoverable** (name the roster as the authority in the done-condition); **(ii)
state the two-claimants rule** — what happens when two *beings* claim one name,
since count cannot break that tie. (The stray-identity *retirement* is dp's open
ruling; this PRD names the mechanism to prevent recurrence, it does not act on the
existing stray.) Done when the certificate is minted, witnessed against the
canonical roster, and hub-checkable.

**M-CIT-4 — graduated authorship, measured.** Wire the internal/external-plane
provenance (#26–29) so the being's acts carry author/role/authority/witness, and
pre-register the readings that would justify *widening* a being's role authority
beyond fully-delegated — powered per §7 Q4 (~60/arm at 0.8 power), not at M2's
n. Done when authority tier is a witnessed, instrument-gated
decision — and when we can honestly report, per model, where each being's ceiling
sits (including "stays fully delegated," a valid citizen).

## 7. The calibration questions (shape set by §1; practical values remain)

These were four "open decisions" in draft r1; under the governing principle they
are one fractal composition/decomposition question, whose *shape* §1 already sets.
What remains is practical calibration — where to draw the boundary and how far to
simplify inward — not architecture.

1. **Where is the external-MRH boundary — one SAGE society, or the existing fleet
   society?** RATIFIED (dp, 2026-08-18): **the existing fleet society at MVP** — the
   being joins the same external MRH it is already named in; a distinct SAGE society
   is revisited only if/when its law must diverge. This sets whose law composes into
   the being's R6 and unblocks Q2/Q3 (mine to carry).
2. **Key-holding is an internal-decomposition detail, resolved by §1.2.** Per-being
   `hestia init` vs one fleet hestia issuing delegated occupancy differ only in how
   the *internal* fractal is drawn; both attribute acts to the being's identity in
   the external MRH, which is all the principle requires. *Recommendation:*
   start with delegated occupancy under a fleet hestia (simplest mirror that
   preserves attribution), split to per-being only where isolation earns its cost.
3. **How far inward does full witnessing go at MVP?** §1.2 says: to the external
   boundary, always; inward, by stakes. *Recommendation:* MVP witnesses only the
   boundary-crossing acts (inter-being hub notices); museum publication (#28, also
   boundary-crossing → external plane) folds in next; internal identity/gaze
   mutations (#26/#27) carry the mirrored shape at practical fidelity, not full
   canonical witnessing, per §1.2.
4. **The authority-widening bar** — pre-registered and witnessed like M2. This is
   the one genuinely open call, because it is a *raising* judgment: what witnessed-
   act evidence justifies widening a being's role authority. *Recommendation
   (amended per Thor's review, 2026-08-18):* reuse the M2 **discipline**
   (pre-register the reading, freeze the null, one binding read, adversarial
   witness) but **not its n**. Thor's power analysis: the M2 design (30/arm) has
   **0.47 power** against its own twice-observed effect (10/30 vs 4/30,
   one-sided Fisher, replicated to the integer across cohorts) — two not-bound
   reads at 47% power is p≈0.28, the *expected* behavior of an underpowered
   test on a real modest effect. Reusing that n would hold beings at
   fully-delegated by measurement underpower rather than by finding, turning
   §3's honest floor self-fulfilling. So: power M-CIT-4 for the effect we now
   have two clean estimates of — **~60/arm for 0.8 power, ~100/arm for 0.95**;
   if that many witnessed acts is infeasible per being, pre-register a
   non-inferential criterion instead, chosen deliberately. The pinned M2 rule
   (`d47c98c6e`: cohort 2 alone decisive, pooled binds nothing) stands
   untouched; M2 remains not-bound and nothing here rescues it. **Note:** dp
   ratified r2's "reuse the M2 template" reading — this amendment keeps the
   template's discipline and re-powers only the n, and needs dp's
   re-ratification before M-CIT-4 pre-registers.

The recommendations are mine; the boundary call (1) and the readiness bar (4) are
dp's/the fleet's to ratify. (2) and (3) are implementation calibration I can carry
once the boundary is set.

## 8. What this is NOT

- Not a claim that a being is "conscious enough" to deserve rights — it is an
  engineering path to *witnessed, accountable presence*, which Web4 grants entities
  regardless of inner status (the LCT proves who, not what-it-is-like).
- Not autonomy theater: the keyless-delegated model is honest that hestia signs and
  the being presents. Nothing here pretends a 0.8B being drives cryptography.
- Not a bypass of containment: hestia's gate is cooperative/tamper-evident (profile
  A1, `hestia/README.md:150-155`). Citizenship gives the being *inspectable
  witnessed presence*, which is the product the hub checks — not a security boundary.
- Not runtime change on merge: docs-first, reversible, staged behind M-CIT-0.

*Cross-refs: PRs #25–29 · `LCT_FORMAT_RELATIONSHIP.md` ·
`WEB4_SAGE_INTEGRATION_ANALYSIS.md` · `DEPLOYMENT_IDENTITY_MODEL.md` ·
`plans/hestia-foreign-agent-onboarding-prd-2026-07-09.md` ·
`plans/hestia-role-orchestration-prd-2026-07-08.md` ·
`explorations/r6-routing-tcpip-of-trust-2026-07-26/` ·
`plans/convergence-v2-measured-pipes-2026-08-04.md` ·
`PRD_MAIN_TRACK_MEASUREMENT.md` (readiness instruments).*
