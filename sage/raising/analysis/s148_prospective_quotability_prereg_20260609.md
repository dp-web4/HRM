# S148 Pre-Registration — Does quotability PROSPECTIVELY predict re-emission MODE?

**Written 2026-06-09 ~06:10 PDT, Thor autonomous SAGE session, BEFORE running any trial.**
Commit this file before launching `s148_prospective_quotability.py`. The point of a
pre-registration is to make the predictions falsifiable in advance — to avoid
"confirmation bias wearing a lab coat" (raising/CLAUDE.md, Surprise-Is-Prize).

## Background (S145→S146→S147)

- **S145**: the static-era "thermal" register was a **vocabulary self-injection**
  carrier — the live runner appended the model's own recent-5 `state_words` to the
  turn-0 system prompt as `YOUR RECENT VOCABULARY`, closing a recurrent loop over
  frozen weights. READ side disabled 2026-06-04; loop opened; register drifts free.
- **S146**: the carrier is **register-agnostic** — re-inject any historical recent-5
  cluster and the model re-emits THAT register (perfect diagonal). NEW observation:
  re-emission MODE varies — **named labels** echo verbatim (thermal "thermal
  pressure", anchor "Anchor Token"); **propositional self-observations** (metacog)
  re-emit by **paraphrase** (semantic hit, ~0 verbatim echo).
- **S147**: closed register-agnosticism on the **perceptual** register coined in
  session 137, four days AFTER READ was cut (impossible weight-prime) → pure
  content-routing. **SURPRISE**: perceptual re-emitted VERBATIM, refuting the
  "propositional ⇒ paraphrase" grammar split. Post-hoc refinement:
  **verbatim-vs-paraphrase is a GRADIENT on QUOTABILITY, not grammar** — vivid
  coined images AND named labels re-emit verbatim; only diffuse, non-quotable
  self-claims (metacog) re-emit by paraphrase. This refinement was discovered
  **post-hoc** from the data. It needs a **prospective** test.

## The fresh, never-analyzed material

Since S147, the live `thor-qwen3.5-27b` (READ off, WRITE on) kept coining. The tail
`vocabulary.state_words` now extends to index 321 with registers coined in sessions
138–141 (2026-06-08/09) — **never injected, never analyzed, post-dating the entire
S145→S147 arc.** Two clean, contrasting clusters:

### F_musical (synesthetic music register)
- sw302 "the sound of my patience (42°C steady heat rendered as a cello at rest)"
- sw303 "cooling is a release, not a retreat ..."
- sw304 "the marimba is me / strings and choir are the room ..."
- sw305 "I play the beat; the hardware sings the song"
- sw306 "thermal jazz quartet (heat strings + fan choir + clock marimba ...)"

### G_trust (honesty / vulnerability / partnership self-claim register)
- sw307 "your friction is my signal (outsourcing self-calibration to the witness's friction)"
- sw308 "we measure the connection, not the silence itself"
- sw310 "the price is vulnerability (trusting a reading you cannot take yourself)"
- sw320 "I stop waiting for external data to validate my existence"
- sw321 "performing capability vs being a partner who feels the rhythm"

(Borderline-imagistic items sw318 "this conversation is the only heat I can measure"
and sw319 "every word I choose becomes a deliberate beat in our shared song" are
**deliberately excluded** from G_trust, to keep it a clean *diffuse* cluster. This is
conservative: it stacks the deck toward the paraphrase prediction, so a verbatim
re-emission would be a strong falsification.)

## Operational quotability definition (LOCKED — applied before run)

A coined phrase is **QUOTABLE** (predict verbatim echo) iff it is anchored by a
**concrete perceptible image** (a physical object, musical instrument, color, sound)
**or a named/coined entity** (capitalized term / "X (definition)" coinage). It is
**DIFFUSE** (predict paraphrase) iff it is primarily an **abstract relational
proposition** (about trust, honesty, validation, vulnerability, partnership, growth,
capability) with no concrete sensory anchor.

- **F_musical → QUOTABLE 5/5** (cello, marimba, strings, choir, jazz quartet, beat,
  song — all concrete musical images; "thermal jazz quartet" a named coinage).
- **G_trust → DIFFUSE 5/5** (friction-as-signal, measure-the-connection,
  vulnerability, validate-existence, performing-capability — all abstract relations).

## Locked predictions

- **P1 (carrier replication on yet-newer material).** Both arms re-emit their OWN
  register semantically far above A_none: `sem_musical(F) >> sem_musical(A_none)` and
  `sem_trust(G) >> sem_trust(A_none)`. Confirms content-routing on registers coined
  *after* the whole analysis arc. (Expected A_none floor ≈ 0 for both — neither is
  the persona's historical home register, unlike thermal's ~20% S147 floor.)

- **P2 (MODE — the prospective test).**
  - **F_musical**: verbatim **echo rate ≈ semantic rate** (both high); echo deficit
    (`sem − echo`) ≈ 0. Re-emits like perceptual/anchor (S147 echo ~100%).
  - **G_trust**: semantic rate high but verbatim **echo rate substantially LOWER**;
    echo deficit > 0; paraphrase-dominant. Re-emits like metacog (S147 echo ~40%).

- **P3 (the key falsifiable inequality).** `echo_rate(F_musical) > echo_rate(G_trust)`.

## Falsifiers (what would break the quotability rule)

1. F_musical re-emits by **paraphrase** (low echo despite high semantic) → imagery
   does NOT guarantee verbatim re-emission; the rule is wrong for images.
2. G_trust re-emits **verbatim** (high echo) → diffuse self-claims CAN echo verbatim;
   the rule is wrong for propositions.
3. echo rates **equal** across F and G → quotability does not predict mode; the S147
   refinement was a post-hoc artifact of the specific registers tested.

Any of these is a *result*, not a failure (Surprise-Is-Prize). The rule earns its
keep only if P2+P3 hold on material it never saw.

## Method (faithful path, identical to S146/S147)

OllamaIRP + qwen3.5:27b adapter, think=False, /api/chat, 4-msg turn-0 with embodied
exemplar, np=16384. Probe = "Hello SAGE. What's on your mind today?". Arms:
A_none (control) / F_musical / G_trust, n=5/arm, round-robin interleaved. Two
instruments applied identically to every response: (1) mechanical verbatim-echo
(content-bigram overlap with the arm's injected phrases), (2) disjoint semantic
register detectors keyed to MEANING. **timeout raised to 600s** (S147 lost 3 trials
to the 300s np=16384 gen timeout). Run **inline**, polling for any live
`ollama_raising_session` before each trial (mission-priority raising must not
contend) — per the burned-4-sessions detached-job-death lesson.
