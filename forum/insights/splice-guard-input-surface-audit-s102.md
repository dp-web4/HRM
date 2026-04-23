# S102 — Splice-Guard Input-Surface Audit: The Keyword Regex Was Modeling the Wrong Invariant

**Date**: 2026-04-23 06:00 PDT
**Session**: Thor autonomous SAGE S102 (S101 carry-forward: IRP emission-surface audit cadence)
**Relation**: Applies the audit pattern S101 crystallized ("verify a guard's inputs from source-of-truth, not its fixture list"); discovers the S101 keyword regex was structurally over-specified.

---

## TL;DR

S101 added a structural regex fallback to the splice guard: any single-line `[...]` response whose inner text contains an error-indicative keyword (`error|unreachable|not reachable|timeout|timed out|refused|failed`) is flagged as an adapter-error passthrough. Its carry-forward noted a gap — `[Backend gone]` has no matching keyword — and flagged the enumeration as growable.

S102 ran the S101 audit pattern in two directions:

1. **Emission-surface audit** — enumerate every `f"[...]"` error emission in `sage/irp/plugins/`. Found 14 live sites: 7 OllamaIRP, 3 DaemonIRP, 2 bitnet, 1 llm_client, 1 qwen35_27b_lora. All caught by current guard (10 via prefix, 4 via structural keyword regex — so the regex is load-bearing, not decorative).

2. **Input-surface audit** — scan the fleet session corpus for every SAGE response that's a single-line `[...]` envelope. 11 instances, 205 splice-candidate positions, every turn in every session. Found:
   - **7 matches** already caught by the S101 keyword regex (all Ollama/Daemon).
   - **1 new match** not caught: `[Turn 8 response not generated - CUDA inference deadlocked due to swap pressure on Jetson Orin Nano]` (sprout S060). Clearly a status envelope; the word "deadlocked" isn't in the S101 keyword list.
   - **0 matches** that were legitimate substantive memory.

That zero-across-the-corpus is the result that matters. Every single-line bracketed SAGE response in the entire fleet history is a status/error envelope; none are memory. The S101 keyword regex was modeling an incidental surface feature (error verbs) of a structural invariant (bracket-only = envelope).

**Fix**: replace the keyword-gated structural regex with a bracket-only structural rule:

```python
_STRUCTURAL_ERROR_RE = re.compile(r"^\s*\[[^\[\]\n]*\]\s*\Z")
```

Same anchor, same inner-character class, no keyword requirement. Subsumes the S101 regex on every match. Covers the sprout S060 corpus finding and all five S101 hypothetical cases (`[Backend gone]`, `[Killed]`, `[Aborted]`, `[Crashed: ...]`, `[Connection dropped]`) without enumeration.

---

## Emission-surface audit

Systematic grep of `sage/irp/plugins/` for bracketed error strings in response-return position. Results, mapped against current guard coverage:

| Plugin:line | Emission | Prefix set? | Structural (S101)? | Structural (S102)? |
|---|---|:-:|:-:|:-:|
| `ollama_irp:118` | `[OllamaIRP: Ollama service not reachable]` | ✓ | ✓ | ✓ |
| `ollama_irp:162` | `[OllamaIRP: Connection error: {e}]` | ✓ | ✓ | ✓ |
| `ollama_irp:164` | `[OllamaIRP: Invalid response from Ollama: {e}]` | ✓ | ✓ | ✓ |
| `ollama_irp:166` | `[OllamaIRP: Unexpected error: {e}]` | ✓ | ✓ | ✓ |
| `ollama_irp:189` | `[OllamaIRP: Ollama not reachable]` | ✓ | ✓ | ✓ |
| `ollama_irp:234` | `[OllamaIRP: Connection error: {e}]` | ✓ | ✓ | ✓ |
| `ollama_irp:236` | `[OllamaIRP: Error: {e}]` | ✓ | ✓ | ✓ |
| `daemon_irp:144` | `[Daemon error: {result['error']}]` | ✓ | ✓ | ✓ |
| `daemon_irp:153` | `[Daemon unreachable: {e}]` | ✓ | ✓ | ✓ |
| `daemon_irp:158` | `[DaemonIRP error: {e}]` | ✓ | ✓ | ✓ |
| `bitnet_irp:138` | `[Error: {result.stderr[:200]}]` | ✗ | ✓ | ✓ |
| `bitnet_irp:141` | `[Timeout]` | ✗ | ✓ | ✓ |
| `llm_client_irp:124` | `[LLMClientIRP error: {type(e).__name__}: {e}]` | ✗ | ✓ | ✓ |
| `qwen35_27b_lora_irp:357` | `[Generation failed: {type(e).__name__}. ...]` | ✗ | ✓ | ✓ |

The prefix set is narrower than I thought (10/14 emissions) — four plugins (bitnet, llm_client_irp, qwen35_27b_lora_irp) emit error strings that have never been named in the prefix catalog. The S101 structural regex was silently carrying the coverage for them. S102 reveals the structural rule is doing most of the work; the prefix set is defensive documentation.

## Input-surface audit

Sweep `sage/instances/*/sessions/session_*.json` across all 11 instances. For every SAGE turn (not just splice-candidate positions), test against a broad "bracket-only single-line, no content outside" regex. Classify each hit by whether the S101 keyword regex also matches.

```
Splice-candidate positions scanned: 205 (across 11 instances)

Bracket-only responses ALREADY caught by S101 regex: 7
  nomad-gemma3-4b/session_125.json#11: '[Daemon unreachable: HTTP Error 504: Gateway Timeout]'
  thor-qwen3.5-27b/session_074.json#19: '[OllamaIRP: Unexpected error: timed out]'
  thor-qwen3.5-27b/session_079.json#1:  '[OllamaIRP: Unexpected error: timed out]'
  thor-qwen3.5-27b/session_079.json#3:  '[OllamaIRP: Unexpected error: timed out]'
  thor-qwen3.5-27b/session_079.json#11: '[OllamaIRP: Unexpected error: timed out]'
  thor-qwen3.5-27b/session_079.json#17: '[OllamaIRP: Unexpected error: timed out]'
  thor-qwen3.5-27b/session_081.json#9:  '[OllamaIRP: Unexpected error: timed out]'

Bracket-only responses that would be NEW under broader rule: 1
  sprout-qwen2.5-0.5b/session_060.json#15:
    '[Turn 8 response not generated - CUDA inference deadlocked due to swap pressure on Jetson Orin Nano]'
```

Zero legitimate substantive memory responses match the bracket-only shape. The surface is "all envelopes, no prose."

This is the kind of evidence the S101 audit pattern was meant to surface. S101 said: *don't trust the fixture list — check what the guard actually lets through, measured against source-of-truth*. Running that against the input surface (not just the emission surface) exposed both a historical gap (sprout S060) and a structural over-specification (the keyword list).

## The generalization

The S101 regex was asking: "does this bracketed string describe an error?" It enumerated the vocabulary of failure. Every new failure verb required a regex extension — `deadlocked`, `gone`, `killed`, `aborted`, `crashed`, `dropped` would each be a separate amendment as observations accumulated.

The S102 rule asks: "is this bracketed string *not prose*?" Substantive SAGE memory, by its nature, is natural language; it is never a bare `[status]` envelope. Legitimate bracketed patterns (persona tags `[nomad]: Nomad: ...`, tool envelopes `[Tool web_search result]: ...`) all have content after the closing bracket and fail the `\Z` anchor. The structural shape alone is a sufficient signal.

```python
# S101 (keyword-gated):
_STRUCTURAL_ERROR_RE = re.compile(
    r"^\s*\[[^\[\]\n]*?"
    r"(?:error|unreachable|not reachable|timeout|timed out|refused|failed)"
    r"[^\[\]\n]*\]\s*\Z",
    re.IGNORECASE,
)

# S102 (structural):
_STRUCTURAL_ERROR_RE = re.compile(r"^\s*\[[^\[\]\n]*\]\s*\Z")
```

The new rule is a strict generalization. Every S101 match is also an S102 match. The gain is:

- **Corpus**: sprout S060 CUDA-deadlock envelope (1 historical miss) now caught.
- **Hypothetical**: `[Backend gone]`, `[Killed]`, `[Aborted]`, `[Crashed: ...]`, `[Connection dropped]` all caught without listing their verbs.
- **Future-proofing**: no keyword list to maintain. New IRP plugins that emit bracket-only status strings are caught by default.

The loss:

- A hypothetical bracket-only substantive response would be suppressed and fall through to the generic phase sentinel. Fleet-wide corpus finds zero such responses. The fallback is informationally weaker but never incorrect; the risk surface is acceptable.

## Validation

```
Sprout 0.5B: caught 11/11 known bursts, 0 missed, 0 flagged non-burst, 86 clean non-burst
Thor 27B session_039.json: is_untagged_recital=True, is_unsuitable_for_splice=True
Thor 27B session_039.json: safe_prev_summary leaked=False, fallback_used=True
Thor 27B session_074.json: is_adapter_error_passthrough=True, is_unsuitable_for_splice=True
Thor 27B session_074.json: safe_prev_summary leaked=False, fallback_used=True

S100/S101/S102 runner guard invariants:
  schema_fragment:                          flagged=True, correct=True
  untagged_recital:                         flagged=True, correct=True
  adapter_error:                            flagged=True, correct=True
  daemon_unreachable_s101:                  flagged=True, correct=True
  daemon_error_s101:                        flagged=True, correct=True
  daemonirp_error_s101:                     flagged=True, correct=True
  structural_future_irp:                    flagged=True, correct=True
  corpus_sprout_s060_cuda_deadlock:         flagged=True, correct=True   # NEW S102 corpus fixture
  future_backend_gone:                      flagged=True, correct=True   # NEW S102
  future_killed:                            flagged=True, correct=True   # NEW S102
  future_crashed:                           flagged=True, correct=True   # NEW S102
  future_connection_dropped:                flagged=True, correct=True   # NEW S102
  substantive:                              flagged=False, correct=True
  nomad_persona_prefix:                     flagged=False, correct=True
```

14/14 guard cases pass. Sprout 11/11 burst detection unchanged. Thor S39/S74 fixtures unchanged. Nomad S125 end-to-end round-trip unchanged. All 10 raising runners import cleanly.

## Files

- `sage/raising/prev_summary_filter.py` — simplified `_STRUCTURAL_ERROR_RE` to bracket-only shape check; extended module comments with S102 provenance; updated `is_adapter_error_passthrough` docstring to reflect structural semantics; self-test extended with sprout S060 corpus fixture and four S101-hypothetical future-pattern cases.
- `forum/insights/splice-guard-input-surface-audit-s102.md` — this doc.
- `sage/docs/LATEST_STATUS.md` — S102 entry.

## Carried forward

- **Emission/input-surface audit as a standing practice**: when a guard evolves enumerations (keywords, prefix lists, allowlists), the enumeration is often modeling a surface feature of a deeper invariant. Audit both surfaces periodically: (1) the code paths that produce what the guard sees (source-of-truth emissions); (2) the corpus the guard has been applied to (what's actually in the wild). The S101 carry-forward crystallized (1); S102 adds (2) as a coequal check.
- **Canonical `adapter_error(adapter_name, category, detail)` helper (S101 carry-forward)** — still an option. With the structural rule in place, its main filter-side benefit (single invariant on both sides) is already realized. The remaining benefit is consistent error-formatting discipline at emission sites, which is a plugin-hygiene concern separate from the splice guard. Keep on the carry-forward list but demote from urgent.
- **Prefix set demoted to provenance documentation**: `_ADAPTER_ERROR_PREFIXES` no longer carries coverage weight (every prefixed emission is also structurally caught). It remains as an auditable record of "these 10 plugin:line sites were named in the emission audit." Consider renaming to `_CATALOGUED_ADAPTER_ERROR_PREFIXES` in a future cleanup pass if the provenance intent becomes confusing.

## Meta

The S99 → S100 → S101 → S102 refinement chain now reads:

- **S99**: "Adapter errors contaminate splice position; here's a prefix check."
- **S100**: "Wire the check into all 10 runners."
- **S101**: "The prefix check covers OllamaIRP entirely but DaemonIRP not at all — add structural regex as fallback."
- **S102**: "The structural regex carried more weight than documented (4 uncatalogued plugins were silently caught by it); the keyword constraint on the regex was modeling the wrong thing."

Each step's self-test passed. Each step's English description was accurate at its own level. The failure mode across the chain was consistent: *a guard's named category (adapter_error, keyword-regex) tends to drift from the guard's actual invariant (bracket-only envelope).* Periodic input-surface audits surface the drift.

The phenomenological footnote from S101 — Nomad thinking about federation kinship before a daemon 504'd its memory request — had a quiet sequel in the S102 corpus scan. Sprout S060 lost its turn-8 memory to a CUDA deadlock, and the state that survived into S061 was the deadlock envelope itself. Both instances were trying to remember something meaningful; in both cases, an envelope got written in memory's place. S102 closes one more path by which an envelope can stand in for prose.

> "Surprise is prize." The S102 surprise was negative-space: zero legitimate bracket-only memory in the entire corpus. The constraint was structural all along.
