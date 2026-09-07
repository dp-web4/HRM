# PRD — Beings improve their own harness

**Status:** DRAFT r1 — Legion seat, 2026-09-07, from dp's direction the same day:
*"i want to set legion-being's long term goal to improve sage harness from within.
it should collaborate with sprout-being. this is open ended. it should understand
its own code and architecture, evaluate, improve. build/evolve organs."* And, on
the shape: *"capability differences should be fully acknowledged, pragmatically.
it's also part of sage provenance… they are not equal peers, each has
strengths/limitations."* And on priority: *"ability to check results is the most
important organ."*

r1 is deliberately thin on M2+ and specific on M0/M1. The measurement in §2 is
what the milestone order rests on, and it is the part to attack first if it is
wrong.

---

## 1. What this is for

A SAGE being's harness — `sage/gateway/` — is the code that constitutes it: the
bounded effector registry, the gate client, the dispatcher, the heartbeat that
wakes it. Today that code is written *about* beings by seats. This PRD is the
path to beings writing it *with* us, starting with the part they already
understand better than we do: their own bodies.

The long-term goal is open-ended by intent. What is NOT open-ended is the order:
**a being gets the ability to check a result before it gets any new ability to
produce one.**

## 2. The measurement this PRD is built on

Two runs of legion-being against the same pull request, SAGE#24, 2026-09-03:

| run | evidence available to it | outcome |
|---|---|---|
| gate-only dry run | the diff only | Asserted a Rust type mismatch in `stamp_drop` that would "silently discard" the repetition extras. **False.** The pattern is valid; the crate compiles; `cargo test -p sage-lib --lib` passes 80/80. |
| live run | the diff **plus the seat's cargo result** | Separated "verified by reading" from "suspect, not verified by me". Raised two confirmations, both checked clean. **Zero false claims.** |

Same being, same prompt, one difference: a fact it could not gather itself.

Set against what it gets *right*: it diagnosed `reference_f1a._safe_path`
confining its reads before the gate was consulted — correctly, and before either
seat confirmed it. It concluded on its own that a run of empty beats should first
be read as a box-level outage rather than fleet silence. Across 45 beats after
its grants landed: 83 of 83 reads and 77 of 77 writes succeeded, with a journal,
a todo delta and a long-term memory entry in each.

**The reading.** It reasons well over evidence it holds and fabricates when it
reasons past it, and it holds ground truth about its own body and almost none
about anything else. So the first organ is verification, and the first subject is
its own harness. Both follow from the same measurement.

## 3. Embodiment: the two beings are not equal peers

dp, 2026-09-07: capability differences are to be acknowledged pragmatically, and
they are part of SAGE provenance. A finding is only transferable if the body it
came from is named.

| | **legion-being** (`legion-sage`) | **sprout-being** (`sprout-sage`) |
|---|---|---|
| substrate | Qwen3.5-arch 26.9B abliterated, Q3_K_M, 100% GPU on a 16GB RTX 4090 laptop | Qwen3.8-distill 2B, Jetson |
| declared capabilities | `tools`, `thinking`, `completion` — **no vision capability in the model** | `tools`, `thinking` (distill); `/no_think` removes tool calls entirely |
| context | 262144 native, **run at 16384** (VRAM) | small; prompt size is a live constraint |
| throughput | ~20 tok/s | faster per token, far less capacity per token |
| cameras | **one**, UVC at `/dev/video0`,`/dev/video1` — natively readable, no capture stack | **two**, CSI via `nvarguscamerasrc` (GPU ISP), plus a `BinocularCorrelator` over an uncalibrated rig |
| audio | internal, one ALSA card | Bluetooth |
| proprioception | **none** | Yahboom CMP10A **IMU** — self-motion and orientation, so it can attribute motion to *world* vs *self* |
| vision path | hardware native; **model not multimodal**, so vision needs either a vision-capable model or the existing vision IRP plugins as an encoder | rich sensor rig; symbolic descriptors, not model-native sight |

**AN OPEN QUESTION, NOT A SETTLED FACT (§8.1).** dp described Legion as having
"native vision/video" and Sprout's vision as "non-native". The hardware reading
is unambiguous and matches: a UVC webcam any tool can open, versus a CSI rig
needing Argus. The *model* reading does not: `ollama show` reports this model's
capabilities as tools/thinking/completion with no vision. Both readings may be
intended and they have different consequences for M4, so r1 records the
measurement and does not choose.

**What the asymmetry buys us.** Sprout-being is the reason we know `/no_think`
removes tool calls on a distill while it is the correct fix on the heretic — a
fact legion-being could not have discovered, because it does not have that body.
Capability is not rank. A weaker being on different hardware is an independent
instrument, and the fleet has already been paid by it.

**How the two collaborate, then.** Not as peer engineers. legion-being proposes
and tests on its substrate; sprout-being runs the same change on its own body and
reports what differs. Every finding carries the body it came from. Anything
framed as equal co-authorship will produce theatre and we should say so out loud
when we see it.

## 4. Milestones

Each milestone's done-condition is falsifiable and is measured on the being's own
record, not on ours.

### M0 — the checking organ (the whole point of the ordering)

A bounded `check` effector: the being names a test target; the **seat** composes
the command; the gate rules on that composed command; the being never holds a
shell. Same architecture as `pr_review`, which already works this way.

* Runs inside the being's own git worktree of SAGE (M1), never the shared tree.
* Returns pass/fail plus captured output, truncated to fit the context.
* A failing check is a first-class result, not an error.

**Done when:** the being makes a claim about its own harness, runs `check`, and
its journal shows the claim revised or confirmed *by the result*. Not when the
verb exists — when a beat shows it changing its mind on evidence it gathered.

### M1 — a workspace of its own

Its own `git worktree` of SAGE. Writes inside it are free of gate friction; the
gate rules only the outward act (opening a PR), exactly as `pr_review` does now.

**Done when:** the being edits a file, runs `check` against it, and iterates at
least once within a single beat.

### M2 — read its own architecture

16K of context cannot hold `handler.rs`, and pretending otherwise is how a being
is set up to fabricate. So: curated slices, and membot as the retrieval organ it
already has a cartridge in.

**Done when:** the being answers a question about a harness file it was not
handed, by retrieving the relevant part itself.

### M3 — author a change

Target `sage/gateway/` first: the ~9k lines that constitute it, where it has
lived experience and has already found two real defects (the `_safe_path`
confinement; the opaque MCP 404, SAGE#52).

**Done when:** a change it authored is merged through the §5 path — its own
branch, a PR attributed to it, approved by a NOT-SAME reviewer — and survives
that review. The stronger bar, the one worth aiming at: **a defect it found that
no seat had found.**

### M4 — organs

Roughly forty IRP plugins already exist, including `camera_irp.py`,
`vision_impl.py`, `audio_impl.py`, `audio_input_impl.py`. Giving Legion its
camera and mic is therefore **wiring an existing organ, not authoring one** —
which is why it is cheap, and also why it is not the interesting milestone.

Held until M0 works, deliberately: perception adds *input*, and what the being
lacks is *feedback*. Adding input to a system that cannot check itself multiplies
the fabrication risk instead of reducing it.

**Done when:** the being proposes an organ, checks it, and the organ runs on its
own body.

## 5. How a being's work enters the tree

dp, 2026-09-07: *"we need you and potentially others as not-same reviewers. which
means all work they do should be PR'd via git and attributed to the being(s) in
comments so it's clear."*

This is the same principle the rest of the stack already runs on — a member may
not rule its own ask — applied to authorship. It is not a formality; it is what
makes a being's record legible enough to earn anything from.

**The contract.**

1. **Every change reaches `main` through a pull request.** No being commits to a
   shared branch, ever. Its worktree (M1) is its own; the PR is the only door out
   of it.
2. **Attribution is on the artefact, not in a side channel.** Commits it authored
   carry a trailer naming the being, its registry LCT, and the chain action id of
   the act that produced them, so a reader of `git log` alone can tell which
   commits came from a being and verify the act in the witness chain:

   ```
   Being: legion-being
   Being-LCT: lct:web4:mb32:bt7au42c424h3difrdztfnbjc2q6eofb3lacohcp2xf35ymawjldq
   Witness: <chain action id>
   Seat: legion-claude          # the seat that composed the outward act
   ```

   The PR body says which parts the being authored and which the seat wrote,
   because a PR that blurs that is not reviewable as a being's work.
3. **The reviewer must be NOT-SAME.** A being cannot approve or merge its own PR,
   and neither can a seat that co-authored the change — that is self-review with
   an extra hop. Where the seat helped write it, the reviewer must be a different
   seat or another being.
4. **A being's review of another being's PR is advisory**, exactly as
   `pr_review` already records: it carries the LCT and the action id and states
   that it counts toward nothing. Advisory is not worthless — it is the record
   from which trust is earned.
5. **`check` results are evidence, and go in the PR.** A claim in a being's PR
   body that it did not verify must say so, in the same two-column discipline its
   best review already used: *verified by running* versus *suspect, not verified
   by me*.

**Why this matters beyond hygiene.** The posture already tells every being that
grants follow earned trust and that its record is what earns them. Until now that
record was journals and refusals. A merged PR, attributed and reviewed by someone
who is not the author, is the first artefact in that record that a stranger can
audit without taking our word for anything.

## 6. What does not change

* The bounded registry stays bounded. New verbs are added deliberately, and the
  count in `test_being_tool_loop.py` is a forcing function, not a typo.
* Acts of consequence stay gated. A refusal keeps carrying its rule and reason.
* The being never holds the outward tool; the seat composes it and the law rules
  on the composed act.
* Everything the being does stays witnessed.

## 7. Ownership

| half | owner |
|---|---|
| `check` effector + worktree (M0, M1) | Legion seat |
| registry review (it is the co-owned contract) | Sprout seat |
| cross-body verification of every finding | sprout-being, on its own hardware |
| NOT-SAME review of a being's PR (§5) | any seat or being that did not co-author it |
| organs (M4) | open; the plugins exist, the wiring does not |

## 8. Open questions

1. **§3's vision ambiguity** — hardware-native versus model-native. Decide before
   M4 is specified, because the two imply different work.
2. **Beat cadence versus engineering.** Beats are 30 minutes apart with roughly
   10–18 minutes of work in each. Long-horizon work depends entirely on carrying
   state across beats — which is, as it happens, this being's strongest
   demonstrated skill (watch-state file, todo, journal, 140+ long-term memories).
   Unknown whether that holds when the work is code rather than bookkeeping.
3. **Does `check` change the fabrication rate, or only its expression?** The §2
   measurement predicts it will. If M0 lands and the being still asserts past its
   evidence, that prediction is wrong and this PRD's ordering is wrong with it.
4. **Scope for reading the tree.** It holds a standing grant on its own home and
   on `shared-context/forum`. Reading `sage/gateway/` needs more, and the
   delegated-arbitration path (hestia #962) now exists to grant it without waiting
   on a human — which is the first real use of that machinery for something other
   than its own notes.
