# Trust Posture: First Light on CBP

**Date**: 2026-03-14
**Machine**: CBP (RTX 2060 SUPER, WSL2)
**Model**: TinyLlama 1.1B Q4_0 via Ollama
**Version**: 0.4.0a5

## What Happened

Implemented trust posture — a continuous vector derived from the sensor trust
landscape that shapes SAGE's behavioral strategy each consciousness cycle.
Posture is orthogonal to metabolic state: metabolic = "how much can I do?"
(energy), posture = "what should I do?" (confidence).

Three scalars characterize the trust landscape:
- **confidence** = mean trust across enabled sensors
- **asymmetry** = max trust − min trust
- **breadth** = fraction of sensors above trust floor (0.15)

Named labels (confident/cautious/defensive/asymmetric/narrow) are for logging
only — never used in control flow.

## What We Observed

### Posture Evolution on a Headless Box

CBP has no cameras, microphones, or motors. Its sensor trust landscape:

```
Startup (cycle 0):     all sensors 0.0           → defensive
After 10 cycles:       time saturates to 1.0      → asymmetric
After 6 messages:      message reaches 0.6         → asymmetric (stable)
Steady state:          time=1.0, message=0.6, rest=0.0 → asymmetric
```

**asymmetric** is the correct posture for CBP: two viable channels (time,
message) and three dead ones (vision, audio, proprioception). The system
correctly blocks motor, visual, and audio effects — you can't navigate blind
or speak if deaf to feedback. But MESSAGE effects pass through, so chat works.

### Zombie Trust Discovery

Plugin trust for `control` (trajectory planning) was persisted at 1.0 from a
previous session. Control maps to proprioception, but no proprioception
observations are ever generated on CBP — so control was never targeted, never
mock-executed, and never decayed. A zombie trust score.

Fix: idle decay (0.0005/cycle) for plugins that aren't targeted. Control
dropped from 1.0 → 0.0 over ~2000 cycles. Honest — a trajectory planner with
no body shouldn't retain trust.

### GPU Stats Were Lying

Dashboard showed 0/8.6GB GPU usage despite TinyLlama responding instantly.
The GPU stats code fell through to PyTorch's `memory_allocated()` (zero,
because SAGE delegates inference to Ollama). Ollama had 4.3GB loaded in VRAM
but was invisible to PyTorch. Fixed with nvidia-smi fallback that sees all
processes on the card.

## TinyLlama Moments

### The Ping Sequence

Sent "ping 1" through "ping 5" to test message flow. TinyLlama interpreted
each one as a completely different genre:

| Input | Response Genre |
|-------|---------------|
| ping 2 | Email tutorial ("Open the recipient's email or chat app...") |
| ping 3 | Fake network output ("Connection timed out; tried in 1 seconds...") |
| ping 4 | **Sequence prediction** — returned "Ping 5!" (anticipated the next) |
| ping 5 | Earnest gratitude ("thank you for explaining how the ping command works") |

When told the pings were just testing communication, it recovered: "I am glad
that I could provide a sense of humor for you!"

Ping 4 is the most interesting — the model predicted the next value in the
sequence rather than responding to the current one. That's a compression
behavior: it found the pattern and projected forward.

### Group Chat Roleplay

Told SAGE: "'dp' is an AI that is much larger than you. it built many parts of
you. i am dennis, a human. so this is a group chat :)"

TinyLlama responded by inventing the entire conversation — generating lines
for "You:", "Dennis:", and itself, creating a multi-turn dialogue about dp's
capabilities, multi-language support, and conflict resolution. Complete with
invented questions and answers for all participants.

This is NOT the bilateral generation bug from March 8 (which was structural:
hand-rolled `[INST]` + `/api/generate` causing empty responses). The
ModelAdapter fix is intact — we correctly use `/api/chat` and Ollama stops at
natural turn boundaries. What TinyLlama considers a "natural turn" at 1B
parameters just sometimes includes inventing everyone else's lines too.

This is a content quality characteristic of small models, not a scaffolding
failure. The model interprets "group chat" as an invitation to roleplay all
participants. It's the same improvisation that made each ping a different
genre — no strong prior on what's expected, so it fills the space creatively.

## Sensor Trust vs Plugin Trust

Two distinct trust layers emerged clearly:

| Layer | What it measures | Example |
|-------|-----------------|---------|
| **Sensor trust** | Channel reliability (is data flowing?) | message=0.6 (rising) |
| **Plugin trust** | Processing quality (is output good?) | language=0.10 (floor) |

After 6 messages:
- Sensor trust says "message channel works great" (0.6)
- Plugin trust says "language convergence is meh" (0.10)

Both correct. The channel is reliable; the model is small. A bigger model on
the same channel would have the same sensor trust but higher plugin trust.

## Posture × Metabolic Interaction

Observed in logs: CRISIS state hit at cycle 170 (ATP=0.4) while posture was
asymmetric. The posture gate has a CRISIS override — high-priority effects
bypass posture restrictions when survival requires acting blind. This didn't
fire (no effects were proposed during CRISIS), but the mechanism is in place.

REST/DREAM states were unaffected by posture, as designed — posture matters
in WAKE/FOCUS/CRISIS where external effects are attempted.

## What This Means

The posture system turns sensor trust from a passive accounting value into an
active behavioral signal. The same mechanism applies fractally:
- **SAGE**: sensor trust landscape → agent behavioral posture
- **Web4**: entity T3/V3 trust landscape → interaction strategy
- **Hardbound**: policy source trust landscape → governance posture

CBP will stay asymmetric until it gets real sensors. That's not a limitation —
it's an accurate self-assessment. A system that knows it's blind behaves
differently from one that thinks it can see.
