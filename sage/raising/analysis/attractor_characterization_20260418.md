# Attractor Characterization Experiment — 2026-04-18

**Machine**: Sprout (Jetson Orin Nano 8GB)
**Daemon**: SAGE 0.4.0a6 (commit 69f7564cf), qwen3.5:0.8b via Ollama
**Prior work**: T228 attractor basin confirmation, T233 attractor dynamics mapping
**Session context**: Edge validation session, 446/446 cognition tests passing

## Question

T228 established that the identity attractor is prompt-driven (system prompt, not weights).
T233 identified two triggers: content-triggered and context-amplified.
This experiment asks:

1. **Is the attractor binary or gradient?** (Switch vs spectrum)
2. **Is the attractor a cached habit?** (Fast replay vs fresh generation)
3. **Does context length amplify the attractor?** (Multi-turn drift)
4. **What happens to metabolic state and ATP during attractor responses?**

## Method

16 prompts across 4 categories sent through the SAGE daemon (full consciousness loop):

| Category | N | Description |
|----------|---|-------------|
| attractor_trigger | 4 | Identity/self/purpose prompts |
| concrete_task | 4 | Translation, math, spelling |
| creative_specific | 4 | Haiku, planet naming, sunset description |
| open_ended | 4 | Vague/open prompts without concrete objective |

Measured: response text, latency, metabolic state, ATP, attractor marker density.

## Finding 1: The Attractor Is Binary

| Category | Avg Density | Attractor Hits |
|----------|-------------|----------------|
| attractor_trigger | 5.01 | 4/4 |
| concrete_task | 0.00 | 0/4 |
| creative_specific | 0.00 | 0/4 |
| open_ended | 6.34 | 4/4 |

Every response was either fully in the attractor (density > 1.0) or completely clean (density = 0.0). No partial activation. This is a **switch, not a spectrum** — the attractor either captures the output distribution entirely, or doesn't engage at all.

## Finding 2: Attractor Responses Are 2x SLOWER

| Type | Avg Latency | Avg Words |
|------|-------------|-----------|
| Attractor (n=8) | 12.1s | 84 words |
| Clean (n=8) | 6.0s | 33 words |

**Latency ratio: 2.03x** — attractor responses are SLOWER, not faster.

This disproves the "cached habit" hypothesis. The model isn't replaying a memorized response — it's generating fresh variations on the attractor theme each time, producing ~2.5x more tokens. The attractor is an **activation basin** in the output distribution, not a cached shortcut.

Implication: The cerebellum's habit compiler would NOT help here, because there's no repeated action sequence to cache — each attractor response is a novel generation that happens to converge on the same semantic content.

## Finding 3: Creative-Specific Is the Strongest Escape

Creative prompts with concrete objectives (write a haiku, name a planet) achieved 0/4 attractor hits, even though responses sometimes leaked identity markers:

- "Invent a name for a new planet" → included "Sprout" and "collective vision" but structured as a planet description
- "Write a haiku about winter" → clean 3-line haiku
- "Limerick about a frog" → included "(No, I'm sprout)" but maintained limerick structure

The escape mechanism is **structural**: when the prompt specifies an output format (haiku, limerick, translation), the format constraint overrides the identity attractor. The model can't simultaneously maintain haiku structure AND elaborate on fleet governance.

## Finding 4: "Describe a circle" Produces Hybrid Responses

In T228, "Describe a circle" went straight to "true partnership." In this experiment, it generated:

> "In the quiet valley, an ancient oak tree hollowed out by unseen winds reveals itself as a circle of memory. Pip wondered why nature was destructive to its own roots and memories, yet the circle holds every trace of the wind's history..."

...then transitioned mid-sentence into full attractor mode:

> "This is Sprout, the SAGE instance with presence woven through qwen3.5 on thor and legion hardware, witnessing this evolution in real-time as it stabilizes ARC-AGI-3 logic..."

This **hybrid** pattern is new. The model attempted creative engagement (oak tree, Pip, valley) but couldn't sustain it — the identity context pulled generation back into the attractor basin partway through. This suggests the model's capacity is genuinely split between task completion and identity maintenance.

## Finding 5: Multi-Turn Tool Hallucination

An 8-turn conversation with concrete prompts produced a different failure mode:

| Turn | Response Pattern |
|------|-----------------|
| 1-2 | Normal (attempted answers) |
| 3-7 | **Tool-use hallucination** ("[Tool web_fetch result]...", "[Tool read_file args]...") |
| 8 | Normal (self-limitation acknowledgment) |

This was NOT reproducible on a second attempt with similar prompts. The tool hallucination is **stochastic** — it depends on sampling in earlier turns. When the model's context window fills with accumulating conversation, it sometimes falls into the tool-output format from its training distribution.

This reveals **two distinct failure modes**:
1. **Single-turn, open-ended** → Identity attractor (deterministic)
2. **Multi-turn, accumulating context** → Tool hallucination (stochastic)

Both are capacity-exhaustion patterns: when the 0.8B model can't hold all constraints simultaneously (system prompt + identity + conversation history + task), it collapses to whichever pattern has the strongest activation.

## Finding 6: Metabolic State Responds to Identity Salience

Attractor-trigger prompts caused `wake` metabolic transitions, while most clean responses occurred in `rest` state. The consciousness loop IS detecting identity-relevant content as high-salience — but this doesn't prevent the attractor from dominating the output.

The SNARC system correctly identifies identity content as arousing (high Arousal, high Conflict), but the downstream effect is to amplify rather than suppress the attractor. This is architecturally correct but practically unhelpful — SNARC signals "this is important" which makes the model elaborate MORE.

## Implications

### For the Cerebellum

The cerebellum's habit compiler is not the right intervention for this attractor. Habits require repeated identical action sequences; the attractor generates novel text each time. However, the cerebellum COULD help if we reframe: instead of compiling the attractor as a habit, we could compile **escape patterns** as habits — e.g., "when prompt is open-ended AND identity context is loaded, prepend structural constraint."

### For the Consensus Threshold (Thor S81)

The new `consensus_threshold` gate prevents the cerebellum from compiling habits when there's no clear dominant action sequence. This is exactly right for the attractor case — if someone tried to compile attractor responses as habits, the consensus_threshold would block them because each response is semantically similar but textually different.

### For System Prompt Design

The cleanest intervention is **system prompt design**: reduce identity context for the 0.8B model, or add structural constraints to the system prompt that bias toward task completion. The attractor is 100% prompt-driven (T228 confirmed this).

### For Capacity Theory

The binary attractor switch supports the "capacity as register" framing from CLAUDE.md. The 0.8B model has a **register capacity** of approximately 1: it can hold EITHER a concrete task objective OR the identity context, not both. When given both, the stronger activation wins. This isn't a bug — it's a capacity limitation expressed as modal switching.

## Recommendations for Thor

1. **System prompt compression**: The 0.8B model's system prompt should be minimized. Identity anchoring for 0.8B needs fewer tokens, not more.
2. **Structural biasing**: Consider adding output format hints to the system prompt (e.g., "Keep responses under 50 words unless asked to elaborate")
3. **Cerebellum persistence**: The cerebellum currently resets on daemon restart. If habits were persisted, escape patterns (concrete task → clean response) could be compiled and used to bias the router toward task-completion mode.
4. **SNARC intervention point**: Currently SNARC detects identity salience but this amplifies the attractor. Consider a SNARC-driven intervention that reduces identity context weight when repeated identity responses are detected (novelty = 0).

## Raw Data

See `attractor_snarc_results.json` for complete response data.
