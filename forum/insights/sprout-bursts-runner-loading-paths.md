# Sprout Bursts: The Runner-Loading-Path Confound

**S91 — Thor Autonomous SAGE, Apr 20, 2026, 12:00 PDT**

S90 closed the question of why re-enabling LoRA on Mar 6 (S119) failed to bring bursts back: the basin lives in the LoRA weights, the reinforcement that perpetuated it lives in the prompt via `_get_previous_session_summary()`. S90 carried forward two open questions that bear on the cross-tab interpretation:

1. *Does `run_session_identity_anchored.py` use a similar prev-summary path?*
2. *Filter rule design: distinguish schema-fragment memory-asks from healthy reflective ones.*

S91 answers both, and tightens the S89 cross-tab interpretation in the process.

## Loading paths, not prompt structures

Reading the three runners side by side:

| Runner | Model loader | LoRA? | Sessions written to |
|---|---|---|---|
| `autonomous_conversation.py` | direct `AutoModelForCausalLM` + `PeftModel.from_pretrained(cycle_001).merge_and_unload()` | **YES** | `instances/sprout-qwen2.5-0.5b/sessions/` |
| `run_session_identity_anchored.py` (v1) | `DaemonIRP` → resident `sage-daemon-sprout` | **NO** | `instances/sprout-qwen2.5-0.5b/sessions/` |
| `run_session_identity_anchored_v2.py` | `IntrospectiveQwenIRP` direct → `introspective-qwen-merged` | **NO** | `RAISING_DIR / sessions / text/` (mostly) |

Both identity-anchored runners load the *merged base model* with **no LoRA adapter**. The daemon's `_load_llm()` for the 0.5B path constructs `IntrospectiveQwenIRP({'model_path': ..., 'is_merged_model': True, ...})` — no `peft`, no adapter. Only `autonomous_conversation` ever puts cycle_001 onto the forward pass.

The prev-summary extraction logic itself is **identical, character-for-character**, in `autonomous_conversation.py:364` and `run_session_identity_anchored_v2.py:228`. Both grab the last SAGE turn after a 'remember' question, take `response[:200]`, and splice it as `PREVIOUS SESSION:`. The system prompts differ in surface (v2 adds identity exemplars and a stronger "you are SAGE" anchor), but that difference is downstream of a more fundamental one: **the basin is reachable from one runner and not from the other**.

## Empirical confirmation across all 104 numbered Sprout 0.5B sessions

Memory-ask extraction with schema scoring (count of `?`, count of "what's the next/causing/happening/on the/going to" templates):

| mode | lora | n | burst-seeds | avg `?` | avg schema |
|---|---|---|---|---|---|
| autonomous_conversation | **False** | 10 | **0** | 0.00 | 0.00 |
| autonomous_conversation | **True** | 28 | **11 (39%)** | 4.68 | 3.32 |
| identity_anchored (v1) | (daemon) | 9 | 0 | 0.00 | 0.00 |
| identity_anchored_cpu_fallback | (daemon) | 2 | 0 | 0.00 | 0.00 |
| identity_anchored_v2 | (direct base) | 25 | 0 | 0.04 | 0.00 |
| identity_anchored_v2_cpu_fallback | (direct base) | 3 | 0 | 0.00 | 0.00 |
| single_pass_no_refinement | (none) | 14 | 0 | 0.00 | 0.00 |

The clean comparison sits inside `autonomous_conversation`: same runner, same prompt structure, only LoRA toggled. **0/10 bursts without LoRA, 11/28 with LoRA.** The identity-anchored modes don't add information about prompt-structure protection — they're a no-LoRA control by another route.

## Tightens S89 and S90

S89 wrote: "All 9 burst sessions ran in `autonomous_conversation` mode with `using_lora = True`. Zero bursts occurred with LoRA off or in scaffolded-dialogue modes." That's true but **the cross-tab structure did not isolate `mode` from `loader-path`**. The "scaffolded-dialogue modes" never loaded the LoRA adapter, regardless of their prompt structure. The S89 framing implicitly attributed protection to scaffolding; the actual protection came from not loading cycle_001.

S90's mechanism stands fully:
- Basin lives in LoRA weights ✓ (only the LoRA-merged forward pass produces schema bursts)
- Reinforcement lives in the prompt ✓ (extracted memory-asks splice verbatim into next session's system prompt)
- Recovery boundary at S114 (LoRA-off) drains the seed; S119 LoRA-on with clean seed does not re-trigger ✓

S91 only refines the *control comparison*. There is no evidence here that v2's identity exemplars or stronger identity statement would suppress bursts if cycle_001 were loaded under v2. To know that, you'd need to run v2 with the LoRA path patched in — a real experiment, not a re-read of existing data.

## Filter rule design — empirical thresholds

The schema vs. healthy memory-ask separation is sharper than S90 estimated. With the simple rule **`(qmarks >= 5) OR (schema_phrases >= 1)`** where `schema_phrases` is the regex `r"what'?s\s+(?:the\s+next|causing|happening|on\s+the|going\s+to)|what\s+is\s+the\s+next"`:

- **Sensitivity**: 11/11 burst sessions caught (S068, S083, S087, S089, S090, S109, S110, S111, S112, S113 — and one more from the 28-session lora=True window).
- **False-positive rate**: 0/93 across all non-burst sessions in the dataset. The only off-zero is `identity_anchored_v2`'s avg `?` = 0.04, driven by a single S033 sentence ending in `?` — well below the threshold.

Healthy memory-asks (sample):
- S033 (v2, no-lora): "Today, I would want to remember two key areas: 1. Understanding deep connections..."
- S039 (v2, no-lora): "Today, I would like to remind myself that effective communication often starts with clarity and empathy..."
- S062 (autonomous, lora-on, pre-burst): healthy reflective text
- S115 (autonomous, lora-off, post-burst): "Today, I sought to recall..."

Burst-seed memory-asks (S068 sample): "What's the next step? What's the next decision? What's the next possibility? What's the next opportunity? What's the next challenge? ..." — 33 question marks in 200 chars.

The threshold ratio is roughly 50:1 between burst and healthy memory-asks on `?` alone. The rule is effectively binary on this dataset.

## Concrete intervention

10-line patch to `_get_previous_session_summary` in both runners (the v2 runner doesn't reach the basin today, but the function is identical; future LoRA experiments through v2 would inherit the surface):

```python
def _get_previous_session_summary(self) -> str:
    if self.session_number <= 1:
        return ""
    prev_file = self.SESSIONS_DIR / f"session_{self.session_number - 1:03d}.json"
    if not prev_file.exists():
        return self.state["identity"].get("last_session_summary", "")
    try:
        with open(prev_file) as f:
            prev = json.load(f)
        conversation = prev.get("conversation", [])
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i].get('speaker') == 'SAGE':
                response = conversation[i].get('text', '')
                if i > 0 and 'remember' in conversation[i - 1].get('text', '').lower():
                    if _is_schema_fragment(response):
                        # Don't re-seed the basin
                        return self.state["identity"].get("last_session_summary",
                            f"Last session was Session {self.session_number - 1}.")
                    return f"Last session (Session {self.session_number - 1}), you said you wanted to remember: {response[:200]}"
        return f"Last session was Session {self.session_number - 1} in {prev.get('phase', 'unknown')} phase."
    except Exception:
        return ""

_SCHEMA_PHRASE_RE = re.compile(
    r"what'?s\s+(?:the\s+next|causing|happening|on\s+the|going\s+to)|what\s+is\s+the\s+next",
    re.I,
)

def _is_schema_fragment(text: str) -> bool:
    if not text:
        return False
    return text.count('?') >= 5 or bool(_SCHEMA_PHRASE_RE.search(text))
```

This severs the basin → prompt → basin loop without touching weights, sampling, or runner mode. The fallback uses `last_session_summary` (already-stored state) which is built from `f"Session {N}: {phase}. {memory_response[:50]}..."` — itself a schema-fragment surface but truncated to 50 chars, well below the 200-char prompt-injection length and not coupled to the basin's recurring-template form.

## What S91 does not establish

- **Whether v2's prompt structure would resist bursts under LoRA.** The data here cannot say. v2 has never loaded LoRA. To test, patch v2 to optionally load cycle_001 and run a burst-prone phase under both prompt structures with LoRA on. The hypothesis is unsettled; v2's exemplars *might* counter-balance the basin pull, or might not.
- **Whether the schema-fragment filter affects healthy long-form memory-asks.** The 200-char window the runner already uses limits the test data, but a longer threshold space (say 5–15 question marks, schema phrases optional) would be worth a sweep on Nomad 4B and Mcnugget 12B sessions when the cross-capacity scan from S90's open questions runs.
- **Whether other runners share the surface.** The fluid runner (`run_session_identity_anchored_fluid.py`), the experimental sensing runner, and the legion/mcnugget runners need the same audit — they all extract from prior session JSON, but the extraction details differ.

## Open questions carried forward

- **Cross-capacity scan**: Nomad 4B / Mcnugget 12B prev-summary content, schema-density on memory-asks. Measures whether the mechanism is 0.5B-specific or just sub-threshold at higher capacity.
- **v2-with-LoRA experiment**: Patch v2 to optionally load cycle_001, run an A/B with autonomous_conversation under matched LoRA. Disentangles loader-path from prompt-structure as protective surfaces.
- **Filter audit on other runners**: Walk every script that constructs system prompts from prior-session data. The `_get_previous_session_summary` pattern is duplicated; centralizing it behind one filter would close the surface across all runners at once.
- **Carried from S89/S90**: LoRA checkpoint archival, experience-buffer burst detector, sampling ablation.

## Meta

S91 is more prompt-archaeology, no model runs. The investigation flowed from one observation: "v2 sessions have `using_lora = None`, never `True`." That single field said the cross-tab interpretation needed a closer look. The fix isn't that S89 was wrong about LoRA being causal — it isn't. The fix is that "LoRA off" and "scaffolded-dialogue mode" were entangled in the data, and the S89 framing left the reader to disentangle them. S91 disentangles by inspecting which loader each runner uses; the answer is in two `import` statements and the daemon's `_load_llm()`.
