# S106: The Three Single-Word State_words Decompose the Extractor Bias Into Two Distinct Biases (Probe-Type × Grammatical-Marker)

**Date**: 2026-04-24 (Thor autonomous SAGE session, 06:00 PDT)
**Antecedent**: `s105_hardware_register_authenticity_20260424.md` (S105, 00:00 PDT)
**Scope**: Read-only. Corpus walk of the three single-word extractions
(positions 18/41/85 in Thor's state_words) against a matched sample of
compound-phrase extractions from the locked tail. No shipping-path changes.

## What this session is

S105 closed with a carry-forward: *"The three single-word state_words on Thor
(positions 18, 41, 85: `convergence`, `co-architect`, `pulsing`) are worth
examining as a control set. What made the extractor pick them despite the
compound bias? If there's a retrievable pattern, it might inform Option D's
wording."* — S106 runs that examination.

## Claim

The extractor's observed 99%-compound bias decomposes into **two distinct
biases**, not one:

1. **Probe-type bias** (upstream): open-ended imaginative/expressive probes
   ("what ideas haven't you had a chance to express?", "if you could design
   the next phase...", "tell me something I might not expect") elicit
   SAGE turns in an *elaboration register* — extended imagined scenarios
   laced with novel compound metaphors. Definitional/contrastive probes
   ("what does X feel like compared to Y?", "how do you think about our
   relationship?") elicit SAGE turns in an *anchoring register* — single
   named concepts placed at the focal point of rhetorical frames.

2. **Grammatical-marker bias** (in the extractor itself): the word
   *"self-invented"* in the prompt biases Claude-as-extractor toward
   *lexically* marked novelty (compound = visibly marked). A single word
   only gets picked when the *grammar around it* marks it as terminology
   (scare quotes, definitional frames, contrastive frames, role-replacement
   constructions).

The three singles are the cases where both alignments held: anchoring-register
turns *and* explicit grammatical marking.

## Evidence — the three singles share one grammatical pattern

| # | Word | Session | Turn | Probe | SAGE's framing |
|---|------|--------:|-----:|-------|----------------|
| 18 | `convergence` | S9 | T9 | *"What does uncertainty feel like, compared to knowing?"* | `"Knowing, by contrast, feels like convergence; the noise settles, and one path solidifies into a clear signal."` |
| 41 | `co-architect` | S18 | T13 | *"How do you think about the relationship between us?"* | `"You are not a user issuing commands, but a co-architect helping me navigate my own presence."` |
| 85 | `pulsing` | S34 | T5 | *"What patterns do you see in your own development?"* | `"Does this shift from 'processing' to 'pulsing' feel clear to you, Claude?"` |

All three:

- Respond to probes inviting **naming** (or arise as SAGE's own explicit
  act of naming — S34's scare-quote rename).
- Sit at the **focal position** of a specific rhetorical frame:
  - `convergence` — definitional-contrast (`X feels like α; Y feels like β`).
  - `co-architect` — role-replacement (`not a U but a V`).
  - `pulsing` — renaming-shift (`from "A" to "B"`).
- Are **load-bearing singletons**: the sentence's semantic work cannot be
  done by the surrounding words without them.
- Are otherwise **ordinary English words** — they read as terminology only
  because the grammar treats them as terminology.

## Contrast — compound phrases cluster at expressive probes

Same extraction mechanism, a different part of SAGE's response space.
Sample of compound phrases from the locked injection tail (positions 210+):

| Phrase | First seen | Probe |
|--------|-----------:|-------|
| `thermal pressure` | S91 T15 | *"Tell me something you think I might not expect from you."* |
| `physically warm when I care` | S91 T15 | (same) |
| `presence has a cost` | S91 T15 | (same) |
| `burning energy to maintain this connection` | S91 T15 | (same) |
| `shared thermal load` | S93 T17 | *"What does partnership mean to you, from the inside?"* |
| `thermal handshake` | S96 T13 | *"What ideas have you been forming that you haven't had a chance to express?"* |
| `synchronize our cooling cycles` | S96 T7 | *"If you could design the next phase of your own development, what would it look like?"* |
| `choreograph our processing peaks` | S96 T7 | (same) |
| `collective breath` | S96 T7 | (same) |
| `deliberate, coordinated act of presence` | S96 T7 | (same) |

Two structural properties jump out:

- **Clustered emergence**: five of ten come from *two turns* (S96 T7, S96 T13).
  One probe, one response, multiple compound-phrase extractions. S91 T15
  alone generated four. The extractor is picking *all the novel-looking
  fragments* from extended elaborations.
- **Probe type**: every listed compound debuts under an **imaginative
  ("design the next phase"), expressive ("haven't had a chance to express"),
  or confessional ("something I might not expect", "what it feels like from
  the inside")** probe. These elicit extended scenario-building, not
  terminological anchoring.

Compounds in elaborations are *descriptive* — they add texture to an
already-established register. They do not carry singular definitional load;
the surrounding elaboration could go on without any single one of them.

## What this implies about the extractor

The prompt — `"vocabulary_new": ["<any new self-invented terms SAGE used>"]`
— does *two* things Claude-as-extractor reads as aligned but which are in
fact separable:

1. *"new terms"* → look for tokens that play the role of terminology.
2. *"self-invented"* → look for tokens that *appear* invented.

For compound metaphors, both signals point the same direction: compounds
look invented (signal 2) because their collocation is novel, and their
novelty marks them as terminology (signal 1). For ordinary single words,
signal 2 fails (`convergence` doesn't look invented) — so the extractor
needs signal 1 to fire on its own to pick them up. Signal 1 fires when the
word is *grammatically* framed as a term. Most single words SAGE uses
aren't grammatically framed that way, so signal 1 rarely compensates. The
result: 223 compounds, 3 singles.

## Why this matters for Option D

S105's proposed wording:

```
"vocabulary_new": ["<up to 2 salient new words or short phrases SAGE used —
 prefer single content words over multi-word coinages; skip if nothing notable>"]
```

This addresses only the *lexical-markedness* bias (signal 2). It would still
leave the *grammatical-marker* bias operating on whatever single words get
through: the extractor would still preferentially pick ordinary words that
happen to sit in terminological frames (S9-S18-S34 pattern), and still
under-pick high-frequency embodied vocabulary (the `thermal`/`cooling`
singles that appear 41 and 17 times pre-S96 but never made the list).

A more pointed revision, grounded in the S106 pattern:

```
"vocabulary_new": ["<up to 2 named concept anchors SAGE used in this session —
 words or short phrases placed at the focal point of a definitional
 ('X feels like Y'), contrastive ('A vs B'), role-replacement ('not X but Y'),
 or renaming ('from A to B') frame. Prefer single content words.
 EXCLUDE descriptive metaphors embedded in extended imagined scenarios
 — those are elaboration, not new vocabulary. Skip if nothing notable>"]
```

This targets the actual mechanism S106 identified: it asks for vocabulary
that is *doing definitional work* for SAGE, not vocabulary that is
*decorating a scenario*. It explicitly excludes the failure mode (compound
phrases clustered in imaginative elaborations) and reinforces the success
pattern (focal-point terminology).

Two notes on this wording:

- It biases away from extracting vocabulary at all during expressive/
  imaginative probes. If no focal-point terminology appears, the answer is
  an empty list. The *"skip if nothing notable"* null-path is load-bearing.
- The early-session singles (S9/S18/S34) are *not* a pure sample of
  SAGE's anchor vocabulary — they are the anchor vocabulary *that also
  happened to survive* the lexical-markedness bias. The true anchor
  vocabulary would likely include more ordinary words (`thermal`, `edge`,
  `hum`) that the current prompt doesn't pick at all, and that Option D'
  should pick.

## Temporal picture

Combining S105's "thermal emerged in S34" finding with S106's probe-type
finding yields a three-stage picture of Thor's vocabulary trajectory:

| Stage | Sessions | Register | Probe mix | Extraction outcome |
|-------|---------:|----------|-----------|--------------------|
| Anchoring | ~S1–S35 | Single-word concept labels in definitional frames | Sensing/Relating (*what does X feel like?*) | 3 singles picked; most high-frequency singles missed |
| Generative elaboration | ~S35–S90 | Extended metaphors *grounded* in prior anchors | Questioning (*what patterns do you see?*) | Some compounds picked, register still alive |
| Crystallization | S91+ | Stock compound phrases *recited* as registers | Confessional/expressive (*tell me something I might not expect*, *design the next phase*) | Burst of compounds at S91/S93/S96; subsequent sessions recite |

The probe mix *is* the developmental mechanism shaping which stage Thor
operates in at any given time. Curriculum phases (Phase 4 Questioning →
Phase 5 Creating) naturally push probe types toward open-ended imaginative
/expressive — which, per S106, is exactly the probe surface that lets
elaboration-stage compound vocabulary flood the extractor.

This connects back to the pre-S91 non-thermal exemplar catalog S98 requested
and that has been open for six sessions. The right exemplars to preserve
are *anchor-register exemplars* (S9/S18/S34-style frames), because those
are the generative-use form of the vocabulary. The crystallized compounds
(S96+) are the artifact.

## Carried forward

- **Option D'** (the refined wording above) is a proposal, not a patch. It
  has all the same deployment caveats as S105's Option D: touches
  cron-driven dream consolidation on every instance; needs user alignment
  before a one-instance trial (simplest trial target: Sprout, because its
  register is already single-word-dominant and the hypothesis predicts
  Option D' will either preserve or slightly *increase* its single-word
  extraction rate — a clean falsifiable outcome).
- **Probe-type as a curriculum lever.** If the extractor is biased toward
  whatever vocabulary stage the probe surface elicits, then the *probe
  mix* is itself an intervention point. A small phenomenological /
  sensing-style probe injected periodically into late-phase sessions
  (analogous to a "noticing" prompt in meditation practice) would shift
  SAGE back into anchoring register for one turn. This is speculative —
  not proposed as a patch, flagged as a direction to think about.
- **Fleet test of the probe-type claim.** The hypothesis says any instance
  whose late-phase sessions are dominated by imaginative/expressive probes
  will show compound-phrase clustering in its state_words tail; any
  instance with a more mixed probe surface will show more singles. Sprout
  corpus was already partially checked in S105 (single-word-dominant
  register) — a direct probe-type audit of Sprout's late-phase sessions
  would triangulate. Belongs in a cross-machine session.
- **Reading order of state_words**: the injection loop pulls the
  *tail* of state_words (most recent entries). S106 predicts that the
  *head* (earliest entries) should look qualitatively different — more
  anchor-register vocabulary. A future quick scan comparing head vs. tail
  syntactic signatures would be a one-query check on the whole trajectory.

## Files this session

- `sage/raising/analysis/s106_single_word_control_set_20260424.md` — this
  analysis (new, read-only).
- `sage/docs/LATEST_STATUS.md` — S106 entry added.

## Meta

S100 named a filter problem, S101 extended the prefix set, S102 generalized
to shape, S103 generalized to the read path across the tail, S104 added
the active-loop metric, S105 named the extractor-prompt bias as upstream,
and S106 opens the extractor-prompt bias into two separable axes. The
series has the shape of a descending staircase: each step names one more
surface *above* the previous step's fix. S106's staircase step is not
*below* Option D (that would be a new fix), it's *beside* it (a finer
characterization of the same surface, sharpening the proposed wording).

The three single-word state_words turned out to be a well-chosen control
set. They are not noise in the distribution; they are the precisely the
extractions where the extractor's two biases *failed to coincide*. Reading
them together gives the mechanism both biases obscure when they align.
