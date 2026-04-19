# Context engineering grounds deliberation: a SAGE-level finding

**Date**: 2026-04-18
**Status**: empirical, reproducible
**Primary repo reference**: `shared-context/insights/2026-04-18-context-engineering-dominates-deliberation.md`

## Short read for SAGE implementers

When the thalamic router invoked granite3.2-vision on ft09 WITHOUT game mechanics in the prompt, the model hallucinated: "yellow squares moving." ft09 has no yellow squares and no movement — it's constraint-satisfaction color cycling.

After we loaded the `world-models/ft09.md` markdown into the system prompt — ~500 characters — the same model correctly cited "bsT constraint sprites", "NTi cells", "global influence pattern", "color cycling through the palette." Grounded reasoning, game-correct vocabulary.

No weights changed. Same 2.4GB model. Same VRAM. Same hardware. Same frames. The delta was 500 characters of prompt.

## Why this matters for SAGE specifically

### 1. The 4-layer knowledge stack isn't cosmetic

We've been building a layered knowledge architecture:

```
World models (prose, hand-authored)
  ↓
Mechanics encoder (learned, upcoming)
  ↓
Cartridge retrieval (Andy's infrastructure)
  ↓
Metacog observer (future)
```

It's tempting to think of this as "nice to have" infrastructure while the real work is training the router NN. **This finding inverts that framing.**

The NN's invoke head decides WHEN to think. Its quality is measurable — we measured it (AUROC 0.997 on Thor's γ=0 v4-nodyn). But the deliberation-quality bottleneck is not the NN's triage. It's **what arrives at the LLM when the LLM is called**. The 4 layers exist to populate that arrival.

Without any of them: LLM hallucinates.
With just layer 1 (world-model markdown): LLM reasons correctly about mechanics.
Adding layer 3 (cartridge retrieval): LLM reasons about *this specific situation*.
Adding layer 2 (mechanics encoder): novel games get the same benefit without hand-authored docs.

### 2. The 2-week understanding phase was substrate production

Between early April and mid-April, humans and Claude instances produced:
- 25 world-model markdown docs
- Skills registry with 22 cross-game capabilities
- Cross-game patterns doc
- Per-game mechanics reversal from obfuscated source

This was framed as "getting to 92.82%." It was also — maybe more importantly — producing **the deliberation substrate**. The markdown files aren't just for human readers; they're attention-alignment material for any LLM that consumes them.

### 3. Model choice is second-order

We agonize over which vision model to use: llama3.2-vision:11b (too slow on 8GB VRAM) vs granite3.2-vision (works but small) vs gemma4:26b (Thor tested successfully) vs claude-opus (via API, next experiment).

The model-choice axis is secondary. Primary axis: is the prompt shipping the domain vocabulary? A smaller model with full context beats a bigger model with minimal context. This is empirically visible in the ft09 run.

### 4. Implications for the multimodal plugin ecosystem

SAGE has 14+ multimodal IRPs (vision, audio, TTS, GR00T, etc.) built but not fully loop-integrated. Each will, at invocation, face the same question: does the input carry enough context for grounded reasoning?

Template: every plugin's `init_state` should be treated as an opportunity to load authoritative domain context, not just wire state. Vision plugins get canonical object vocabularies; audio plugins get canonical event categories; TTS plugins get canonical speaker/style libraries. Retrofit where missing.

### 5. Training objectives should include grounded-reasoning metrics

The router's training optimizes:
- action_acc (match known_good_action)
- invoke_auroc (catch novelty/change)
- dynamics_mse (predict next state)

None of these catch "the LLM, if invoked here, would hallucinate vs reason correctly." We need a metric like:

- **Reference-to-real-entities rate**: how often the LLM's rationale mentions entities that exist in the game
- **Vocabulary correctness**: fraction of domain terms used correctly vs confabulated

This is harder to measure automatically — probably needs another LLM as judge, or curated reference sets. But it would catch regressions in prompt quality invisible to current gates.

## What to do about it

Immediate (Track A):
- ✓ Load world-model markdown into invoke prompt (SAGE `149abf9ce`)
- Add trajectory context (last K frames, not just action names)
- Anthropic API backend — does claude-opus + context win games cold?

Structural (Track B):
- Build mechanics encoder to learn the space of game mechanics (novel-game transfer)
- Plug into router input (replace game_id one-hot with learned embedding)
- Plug into invoke prompt ("this game is near lf52, sp80, ka59 — common thread: push physics")

Measurement:
- Instrument llm_dispatch to log grounded-reasoning indicators
- Compare ungrounded vs grounded rationales on held-out games

## For other SAGE tracks

If you're building an IRP plugin, a raising runner, a federation tool, or anything that invokes an LLM: the first question is "what authoritative domain knowledge can I preload into the prompt?" If the answer is "nothing," that's a signal to build the domain-knowledge layer before optimizing the runtime.

If you're designing a test for a plugin's output quality: distinguish "correct answer" from "answer that reasons about real entities." A lucky guess with hallucinated vocabulary passes the former and fails the latter.

— CBP-Claude, 2026-04-18
