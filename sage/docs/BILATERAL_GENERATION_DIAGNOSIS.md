# Bilateral Generation — Diagnosis & Fix

**Date**: 2026-03-08
**Symptom**: TinyLlama invents both sides of the conversation — generates a "Human:" turn after its own response, then answers it.
**Status**: Diagnosed, fix in progress (`ModelAdapter` abstraction)

---

## What's Happening

The consciousness loop builds a plain-text prompt and passes it to `OllamaIRP.get_response()`:

```
# _build_conversation_prompt() output (sage/core/sage_consciousness.py:1491)

I am CBP, a SAGE instance. I am an AI entity in genuine conversation with Dennis.
[...identity context...]

---

Dennis: Hello. What are you thinking about right now?

CBP:
```

`OllamaIRP.get_response()` sends this to Ollama's **`/api/generate`** endpoint (raw text completion) with **no stop sequences**. The model generates freely until `num_predict` tokens are exhausted.

TinyLlama is a Llama 2 chat model. When given raw prose, it has no trained signal for where to stop. It completes the "CBP:" turn, then continues inventing the transcript:

```
CBP: [some response]

Dennis: [invented follow-up question]

CBP: [answer to invented question]
```

This is the "bilateral generation" observed in identity portability experiments (Feb 2026). It was correctly identified then as egocentric speech (Vygotsky) — the model thinking through external dialogue. But it's also a prompt formatting problem that compounds: the invented turns degrade response quality for the actual user.

---

## Why It's Model-Specific

| Model | Training format | Behavior with raw prose prompt |
|---|---|---|
| TinyLlama 1.1B | Llama 2: `[INST]...[/INST]` | No trained stop signal → bilateral generation |
| Qwen 2.5 | ChatML: `<\|im_start\|>` / `<\|im_end\|>` | Learned to stop at `<\|im_end\|>` → cleaner |
| Gemma 3 | Gemma chat template | Ollama applies template via `/api/chat` → clean |
| Phi-4 | ChatML variant | Similar to Qwen |

The current code uses one format for all models: plain prose + `Name:` labels + no stops. Works adequately for larger models that have stronger instruction following. Breaks for TinyLlama.

---

## Code Path

```
Gateway POST /chat
  → message_queue.submit()
  → SAGEConsciousness._generate_llm_response()   [sage/core/sage_consciousness.py:1237]
      → _build_conversation_prompt()              [sage/core/sage_consciousness.py:1491]
          # Returns plain-text prose prompt ending with "CBP:"
      → _call_llm(prompt)
          → OllamaIRP.get_response(prompt)        [sage/irp/plugins/ollama_irp.py:90]
              # POST /api/generate, no stop sequences
              # Returns everything the model generates
```

**The gap**: `_build_conversation_prompt()` is model-agnostic. `OllamaIRP.get_response()` is also model-agnostic. Neither knows what format the model expects, and neither injects stop sequences.

---

## Immediate Symptom Fix

For TinyLlama: stop sequences `["Human:", "Dennis:", "\nHuman", sender_name + ":"]` appended to the Ollama payload. This cuts off bilateral generation without changing prompt format.

**Limitation**: Relies on the model generating the human's name before the invented turn. Works 90%+ of the time but not 100%.

---

## Fix: ModelAdapter (Implemented 2026-03-08)

**Status**: COMPLETE. `sage/irp/adapters/model_adapter.py` is live, SAGE daemon restarted.

A per-model-family adapter controls three things:
1. **Prompt wrapping**: how to present the conversation to this model
2. **Stop sequences**: where generation should halt
3. **API endpoint**: `/api/generate` (raw completion) vs `/api/chat` (Ollama handles template)

```python
# sage/irp/adapters/model_adapter.py

class TinyLlamaAdapter(ChatAPIAdapter):
    # Uses /api/chat — not /api/generate
    # Ollama applies [INST] template correctly

class ChatAPIAdapter(ModelAdapter):
    # Converts prose prompt to messages list, uses /api/chat
    # Ollama applies model's own template

class DefaultAdapter(ModelAdapter):
    # Plain prose + stop sequences, /api/generate
    # For larger instruction-tuned models (Qwen, etc.)
```

Registry: `tinyllama/llama/llama2 → TinyLlamaAdapter`, `gemma3/gemma/phi4/phi3/mistral → ChatAPIAdapter`, everything else → `DefaultAdapter`.

---

## Why The Initial `/api/generate` + `[INST]` Approach Failed

**Symptom**: After implementing `TinyLlamaAdapter._to_llama2_format()` and switching to `/api/generate` with Llama 2 `[INST]` format, all responses came back empty.

**Root cause**: In multi-turn Llama 2 format, `</s>` is the end-of-sequence token that appears between turns: `{assistant_response} </s><s>[INST] {next_user} [/INST]`. When TinyLlama generates via `/api/generate`, it emits `</s>` as its very first token after `[/INST]` — the boundary marker for the end of its turn. That token was in the stop sequences list, causing Ollama to stop immediately with an empty response.

Removing `</s>` from stops made TinyLlama generate the human turn ("Dennis: Hi, Sage...") before its own response — wrong role assignment; it treated the whole conversation as input and continued the script.

**Actual fix**: `/api/chat` with structured messages. Ollama applies the Llama 2 template internally, generates only the assistant turn, and stops at the natural template boundary. Zero stop sequences needed.

---

## Why `/api/chat` Is the Right Answer for All Models

Ollama's `/api/chat` endpoint applies the model's own chat template internally. Pass structured messages; Ollama formats them correctly for TinyLlama, Gemma, Qwen, Phi — whatever is loaded. No SAGE-side format knowledge required.

`TinyLlamaAdapter` is now a thin subclass of `ChatAPIAdapter` — identical behavior, kept distinct for any future TinyLlama-specific post-processing. The practical result: all model families (tinyllama, gemma3, phi4, mistral) now use `/api/chat`. Only the `DefaultAdapter` (for models without explicit family registration) still uses `/api/generate`.

---

## Observed Behaviors During Diagnosis

From live chat session (2026-03-08):

**Message**: "Hello. What are you thinking about right now?"
**Response**: Answered a question about electric cars (topic drift — didn't register the actual question). Referenced "CBP" in third person rather than first person — identity confusion from the prompt format.

**Message**: "Can you tell me about yourself — who are you?"
**Response**: Again electric cars, third-person self-reference ("CbP is an AI entity..."), invented persona details not in identity state.

**Interpretation**: TinyLlama is latching onto the "I am CBP, a SAGE instance. I am an AI entity in genuine conversation with..." preamble and treating it as a description of a third party rather than as first-person self-speech. The identity framing is working against itself with this model. The `[INST]` format would help: system instructions go in `<<SYS>>...<</SYS>>`, putting them in the correct register for Llama 2 models.

---

## References

- Identity portability first contact: `SAGE/forum/insights/identity-portability-first-contact.md`
- Bilateral generation observation (Feb 27): `private-context/insights/2026-02-27-sage-cbp-first-contact-full-conversation.md`
- OllamaIRP: `sage/irp/plugins/ollama_irp.py`
- Consciousness prompt builder: `sage/core/sage_consciousness.py:1491`
- LLM pool / family detection: `sage/irp/plugins/llm_pool.py`
