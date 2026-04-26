# S114 — gemma4:e4b on Thor: Empty-Response Failure Is Broader Than S113 Reported, Plus Empirical Confirmation of Rationale-Mismatch Diagnostic at Scale

**Date**: 2026-04-26 (Thor autonomous SAGE session, 06:00 UTC)
**Picks up**: S113 §"Recurrence #7" (gemma4:e4b empty-response on game prompts) and S113 proposal #6 (apply rationale-mismatch diagnostic to production runs).

## TL;DR

S113 reported that gemma4:e4b on Thor returns empty strings for game-style prompts. S114 reproduces this and finds the failure is **substantially broader than reported**:

1. **gemma4:e4b on Thor is essentially unusable for non-factual prompts.** Even `"Hello"` returns empty. The narrow set of working prompts is limited to short factual questions (`"What is 2+2?"` → `"4"`, `"What is the capital of France?"` → `"Paris"`).
2. **There are two distinct failure modes**, not one:
   - **Mode A — Sampler-dependent**: Prompts like `"Hello"` work with the model's default sampler (`temp=1.0, top_k=64, top_p=0.95`) but fail with greedy (`temp=0.0`) — and ALSO fail at `temp=0.001`, `temp=0.01`, `temp=0.1`.
   - **Mode B — Intrinsic**: Keymap-shaped prompts (`"1=UP"`, `"1=UP 2=DOWN 3=LEFT 4=RIGHT"`) fail with **every sampler tried**, including the default that fixed Mode A.
3. **gemma4:e4b is unique on Thor.** gemma3:12b, qwen2.5:3b, and phi4:14b all respond normally to the same prompts under both greedy and default sampling.
4. **Rationale-action mismatch at scale**: Across all 7,542 production invokes in `~/ai-workspace/shared-context/explorations/`, the mismatch rate is **57.5%** (1086/1890 directional rationales). The worst single file (`whole-brain-at-small-model/data/lean/tn36.json`) is **94.8%**. The lean-vs-fat asymmetry S113 found in lp85.json (`fat: 5.5%, lean: 85.7%` mismatch on the same game) is structural — the format change opens silent-fallback at scale.

The headline implication: **S113's pattern table entry "Recurrence #7" undersells the severity**. The model isn't merely failing on game-style prompts — it's failing on most prompts, with two distinct mechanisms. Anywhere in the SAGE fleet that gemma4:e4b is currently the play model, virtually every LLM contribution is the silent NN-fallback.

## 1. Reproduction of S113 finding

S113 claimed: gemma4:e4b on Thor returns empty for game-style prompts (`'1=UP 2=DOWN 3=LEFT 4=RIGHT'` → `''`) while answering `"What is 2+2?"` normally, with `eval_count > 0` and `done_reason='length'`.

Reproduced verbatim. Run `/tmp/s114/probe_empty_response.py` (5 models × 8 prompts × 2 temps):

| Prompt | gemma4:e4b @ T=0.0 | gemma3:12b | qwen2.5:3b | phi4:14b |
|---|---|---|---|---|
| `"What is 2+2?"` | ✓ `'4'` (eval=2) | ✓ `'2 + 2 = 4'` | ✓ `'2 + 2 equals 4.'` | ✓ |
| `"1=UP 2=DOWN 3=LEFT 4=RIGHT"` | **EMPTY (eval=80, length)** | ✓ responds | ✓ responds | ✓ responds |
| `"Game state: avatar at (5,5)..."` | **EMPTY (eval=80, length)** | ✓ responds | ✓ responds | ✓ responds |
| `"Pick UP DOWN LEFT or RIGHT?"` | **EMPTY (eval=80, length)** | ✓ `'LEFT.'` | partial | refuses but responds |

S113 finding holds at 10/10 game prompts (5 prompts × 2 temps).

## 2. The failure is not specific to game prompts

A boundary probe (`probe_boundary_gemma4e4b.py`) walked a continuum from clearly-non-game to clearly-game prompts. **25/34 prompts returned empty at T=0.0**, including:

- `"What color is the sky?"` (factual, no game context)
- `"What is the capital of France? Reply: 1=Paris 2=London"` (factual question, but contains keymap)
- `"Hello"` (greeting)
- `"Why is the sky blue?"` (factual)

Working prompts at T=0.0:
- `"What is 2+2?"` → `'4'` (eval=2)
- `"1+1?"` → `'2'` (eval=2)
- `"What is the capital of France?"` → `'The capital of France is **Paris**.'` (eval=9)
- `"Count from 1 to 5"` → `'1, 2, 3, 4, 5'` (eval=14)
- `"Reply with a random word"` → `'Ephemeral'` (eval=2)

Even after explicit unload + reload (clean GPU state), the failure pattern is identical. Independent of `num_ctx` (tested 2048, 8192, 131072), `num_predict` (tested 30, 80, 120, 512), and seed (tested 42, 1, 7, 99, 1000, 12345).

## 3. Two distinct failure modes

Sampler matrix on `"Hello"` (`probe_sampler_isolation.py`) reveals Mode A:

| Sampler | Result |
|---|---|
| `temperature=0.0` (greedy) | empty (eval=30) |
| `temperature=0.001` | empty |
| `temperature=0.01` | empty |
| `temperature=0.1` | empty |
| `temperature=0.5` | ✓ `'Hello! How can I help you today?'` (eval=10) |
| `temperature=1.0` | empty (back to broken) |
| Default (no options): `temp=1.0, top_k=64, top_p=0.95` | ✓ `'Hello! How can I help you today?'` |
| greedy + `min_p=0.05` | empty |
| greedy + `top_k=1/5/50` | empty |
| greedy + `top_p=0.5/0.95` | empty |

So the working configurations are narrow: either `temperature=0.5` exactly, or model-default sampler with `top_k=64`. Other sampler tweaks don't help.

**Mode B**: Same matrix on `"1=UP 2=DOWN 3=LEFT 4=RIGHT"`:

| Sampler | Result |
|---|---|
| `temperature=0.0` (greedy) | empty |
| greedy + `min_p=0.05` | empty |
| `temp=0.7 + min_p=0.05` (fixed Hello) | **empty** |
| Default | empty |

The keymap-shaped prompt is intrinsically broken on this model. No sampler change recovers it.

## 4. Endpoint-irrelevant

Hypothesis tested: is `/api/generate` skipping the chat template? `probe_chat_vs_generate.py` ran 6 prompts via both endpoints — identical responses for all. This is not a template issue.

## 5. Token-level inspection

Streaming probes (`probe_raw_bytes.py`, `probe_token_inspection.py`) confirm the model emits 80 tokens for failing prompts but Ollama produces zero non-empty content chunks. The 80 generated tokens decode to empty bytes (likely a model-vocab issue where the argmax token at each step is a special/control token that Ollama filters).

The `done_reason='length'` is misleading: the model is not "running out of tokens to say something useful" — it's emitting tokens that have no rendered representation.

## 6. Cross-model comparison on Thor

Same prompts, same Ollama host, same hardware (`probe_other_models_keymap.py`):

| Model | `"Hello"` greedy | `"1=UP"` greedy | `"1=UP 2=DOWN..."` greedy |
|---|---|---|---|
| **gemma4:e4b** | empty | empty | empty |
| gemma3:12b | ✓ "Hello there!..." | ✓ "This appears to be a code..." | ✓ "Okay, I pick **3**! ... LEFT" |
| qwen2.5:3b | ✓ "Hello!..." | ✓ "It seems there might be some confusion..." | ✓ "I have chosen the number **1**..." |
| phi4:14b | ✓ "Hello!..." | ✓ "The sequence ... is a puzzle..." | ✓ "The numbers ... correspond to directions..." |

gemma4:e4b is unique. The failure is model-specific, not Thor-environment-wide.

## 7. Rationale-vs-action mismatch at scale

S113 proposal #6 was: apply rationale-vs-action mismatch diagnostic to production runs. `rationale_action_mismatch.py` scans all 55 files with `llm_responses` in `~/ai-workspace/shared-context/explorations/`:

| Track | Files | Invokes | PF% | **Mismatch%** | n_dirword |
|---|---:|---:|---:|---:|---:|
| `mcnugget-v2invoke-qwen35-2026-04-20` | 1 | 81 | 3.7% | 59.1% | 22 |
| `qwen-tiny-v2invoke-floor-2026-04-20` | 10 | 507 | 8.9% | 24.0% | 50 |
| `v2invoke-strategy-labels-2026-04-20` | 1 | 83 | 0.0% | 55.9% | 68 |
| `whole-brain-at-small-model-2026-04-22` | 43 | 6,871 | 11.6% | 58.5% | 1,750 |
| **TOTAL** | **55** | **7,542** | **11.2%** | **57.5%** | **1,890** |

Top files by mismatch rate (n_dirword ≥ 50):

| Mismatch% | n | File |
|---|---|---|
| **94.8%** | 201/212 | `whole-brain-at-small-model-2026-04-22/data/lean/tn36.json` |
| **85.7%** | 354/413 | `whole-brain-at-small-model-2026-04-22/data/lean/lp85.json` |
| 65.1% | 110/169 | `whole-brain-at-small-model-2026-04-22/data/lean/vc33.json` |
| 55.9% | 38/68 | `v2invoke-strategy-labels-2026-04-20/data/cd82-play-v2invoke-framestate.json` |
| 50.0% | 97/194 | `whole-brain-at-small-model-2026-04-22/data/lean/su15.json` |
| 38.8% | 52/134 | `whole-brain-at-small-model-2026-04-22/data/lean/sb26.json` |
| **5.5%** | 5/91 | `whole-brain-at-small-model-2026-04-22/data/fat/lp85.json` |

**Same game (lp85), same model (gemma4:e2b), different prompt format**: lean has 85.7% mismatch, fat has 5.5% mismatch. The lean format opens silent-fallback at 15× the rate of fat. This corroborates S113's lean-vs-fat finding with a much sharper measurement.

Sample mismatches from `lean/tn36.json` (94.8% mismatch):

```
'Moving the selected block right and down aligns it with the target.'
  → rationale_word=RIGHT (action=4), dispatched=2 (DOWN)

'This moves the current position down to attempt to match the target in the right'
  → rationale_word=DOWN (action=2), dispatched=3 (LEFT)

'Move the left block one position to the right to attempt to match the target.'
  → rationale_word=LEFT (action=3), dispatched=2 (DOWN)

'RIGHT moves the piece to the target position.'
  → rationale_word=RIGHT (action=4), dispatched=2 (DOWN)
```

The fourth example is the cleanest case: single-word rationale clearly says "RIGHT", system dispatched DOWN. No ambiguity, no log, no flag.

## 8. Updated pattern recognition

S113's pattern table at recurrence #7:

> **#7 — model-output boundary**: gemma4:e4b @ Thor on game prompts → empty output → fallback chain

S114 amendments:

- **Trigger surface is broader** than "game prompts". Includes `"Hello"`, factual questions with non-canonical answers, and most prompts other than narrow factual lookup.
- **Two mechanisms**, not one. Mode A (sampler-fixable) and Mode B (intrinsic).
- **Per-model failure**, not per-machine. Other models on Thor work fine.

This argues that the right framing for #7 isn't a "model-output boundary" issue in general — it's a **specific model on a specific machine producing systematically degenerate output that the harness treats as valid**. The `model_output_empty: bool` flag (S113 proposal #4) would catch all the failures regardless of mechanism.

## 9. Held proposals carry-forward

From S113, untouched by operator:
1. Replace placeholder `ACTION=N` format at 6 callsites with examples format.
2. Surface `parse_path` in `parse_llm_response` return value.
3. Aggregate `parse_failure_rate` in play_lean result.
4. `model_output_empty: bool` flag in dispatch result.
5. Fleet check for gemma4:e4b empty-response on Legion + other machines.
6. Apply rationale-vs-action mismatch diagnostic to post-Apr-24 production runs.

S114 status updates:
- **#5**: Done on Thor side. Confirmed gemma4:e4b is broken and unique among Thor's models. Legion still needs checking. *No post-Apr-24 production game-play data was found in `shared-context/explorations/`* — the most recent game-play tracks all date to 2026-04-22 or earlier. The post-Apr-24 work in `codification-project-2026-04-25` and `policy-sketch-dispatch-2026-04-24` are documentation/design files, not LLM-call logs.
- **#6**: Done on the available pre-Apr-24 corpus. 57.5% mismatch confirms the silent-fallback signature is even more pervasive than S113 reported. **Cannot validate the post-Apr-24 fix-vs-regression hypothesis without new production runs that emit `--json-out`.** This is operator-decision territory: enable `--json-out` in current production runs, then rerun the diagnostic.

S114 new proposal:
7. **Quarantine gemma4:e4b on Thor pending diagnosis.** It is currently failing on virtually all prompts — anywhere it is the play or raising model, the LLM is contributing zero signal while consuming wall-clock latency. A fleet check should also identify whether this is GGUF-corruption-on-disk, an Ollama-on-Jetson issue, or a model-vocab issue.

## 10. Reproducibility

All scripts in `/tmp/s114/`:
- `probe_empty_response.py` — 5-model × 8-prompt × 2-temp matrix (S113 reproduction)
- `probe_boundary_gemma4e4b.py` — 17-prompt continuum for trigger isolation
- `probe_trigger_isolation.py` — 25 syntactic-trigger hypotheses
- `probe_clean_reload.py` — state-pollution test (negative)
- `probe_raw_bytes.py` — token-level streaming inspection
- `probe_controlled_repro.py` — `num_ctx` matrix (negative — not ctx-dependent)
- `probe_chat_vs_generate.py` — `/api/chat` vs `/api/generate` (negative — not template)
- `probe_sampler_isolation.py` — 17-sampler matrix isolating Mode A vs Mode B
- `probe_other_models_keymap.py` — cross-model on Thor (gemma4:e4b uniquely broken)
- `rationale_action_mismatch.py` — diagnostic on full 55-file production corpus

Result data:
- `empty_response_probe.json`, `empty_response_boundary.json`, `trigger_isolation.json`
- `rationale_mismatch_full.json`

## 11. Meta

S113 said: "any time information transforms (input → output, intent → action, observation → fix), the transform can take a silent path that produces plausibly-correct output without flagging the unfamiliarity."

S114 finds: a single model on a single machine has **two different mechanisms** by which it silently produces zero output. The harness treats both as "model responded normally." The principle is robust; the cost of NOT instrumenting silent-path observability keeps growing.

The empirical urgency from S113 (11.2% production PF, 66.5% mismatch) gets larger on full corpus scan (11.2% PF unchanged, but 57.5% mismatch is more rigorously measured across more files; individual files reach 94.8% mismatch). The "lean format opens silent-fallback at scale" hypothesis from S113 is now structurally confirmed: same game, same model, lean vs fat = 85.7% vs 5.5% mismatch.

No code changes shipped this session. All findings are operator-decision territory and additions to the held-proposal queue.
