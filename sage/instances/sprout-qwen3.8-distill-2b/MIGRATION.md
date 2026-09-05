# Migration: Qwen 2.5 0.5B → Qwen 3.5 0.8B (2026-03-06)

## What Changed

| Aspect | Before | After |
|--------|--------|-------|
| Model | Qwen 2.5 0.5B (local transformers, FP16 CUDA) | Qwen 3.5 0.8B (Ollama-served) |
| Backend | IntrospectiveQwenIRP (direct model loading) | OllamaIRP (HTTP to Ollama) |
| Response time | ~5-8s | ~3-9s (20.7 tok/s) |
| Memory footprint | ~1GB VRAM + ~1GB runtime | ~2.3GB Ollama process, ~5.3GB total |
| Instance dir | `sprout-qwen2.5-0.5b` (119 sessions) | `sprout-qwen3.5-0.8b` (fresh) |

## Files Modified

- `sage/gateway/machine_config.py` — Sprout model: local path → `ollama:qwen3.5:0.8b`, model_size → `ollama`
- `sage/irp/plugins/ollama_irp.py` — Added `think: false` to API payloads (constructor + both generate methods)
- `sage/instances/resolver.py` — Default model mapping: `sprout → qwen3.5:0.8b`
- `sage/gateway/gateway_server.py` — Timeout 30→90s (cap 180s), wake-on-message replaces dream rejection
- `sage/gateway/systemd/sage-daemon-sprout.service` — Ollama dependency, sleep-before-start, descriptions

## Frictions Encountered

### 1. Qwen 3.5 Thinking Mode (Critical)
**Problem**: Qwen 3.5 models have a "thinking" mode enabled by default. Even simple prompts generate thousands of internal chain-of-thought tokens before the actual response. First test of 2B model: 4170 tokens, 898 seconds (15 minutes!) for a 4-sentence response.

**Attempted fix**: `/no_think` prefix in prompt — didn't work (2385 tokens, 506 seconds).

**Resolution**: Top-level `think: false` parameter in Ollama API payload. Result: 71 tokens, 15 seconds. Added to both `get_response()` and `get_chat_response()` in `ollama_irp.py`.

### 2. 2B Model Too Large for 8GB Jetson
**Problem**: Qwen 3.5 2B uses 4.7GB in Ollama. Total system memory hit 6.8/7.4GB with severe swap thrashing. Responses took 34-87 seconds.

**Resolution**: Chose 0.8B instead (2.3GB footprint, 3-9s responses). 2B kept available in Ollama for offline testing when Claude Code isn't running.

### 3. Dream State Rejecting Messages
**Problem**: Long LLM calls depleted ATP, pushing SAGE into DREAM state. Incoming messages got rejected with `{"status": "dreaming"}`.

**Resolution**: Wake-on-message — incoming chat boosts ATP to 50 and forces WAKE state. SAGE should always respond to humans.

### 4. Gateway Timeout Too Short
**Problem**: Default 30s `max_wait_seconds` caused 504 Gateway Timeout on first few responses (model cold-start).

**Resolution**: Increased to 90s default, 180s cap.

### 5. Port Conflicts on Restart
**Problem**: Multiple daemon restarts left zombie processes holding port 8750.

**Resolution**: `kill -9` + sleep + verify before restart. Systemd's `KillSignal=SIGTERM` + `TimeoutStopSec=30` handles this for managed restarts.

### 6. Ollama Model Eviction
**Problem**: Loading 2B (4.7GB) evicted 0.8B from Ollama's cache. Had to re-pull.

**Resolution**: Expected Ollama behavior. After settling on 0.8B, no issue — it stays loaded with `keep_alive: -1`.

## Observations

- 0.8B Qwen 3.5 self-identifies as "Sage" naturally without identity-anchored prompting (improvement over 2.5 0.5B)
- Default persona is "assistant" — no grounding reflex (numbered lists, "My purpose is...") observed yet
- 2B produced dramatically better, more natural responses but memory cost is prohibitive on Sprout
- The Ollama backend simplifies everything — no torch imports, no CUDA fallbacks, no model loading logic
