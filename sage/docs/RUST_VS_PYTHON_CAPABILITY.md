# Rust daemon vs Python kernel — capability matrix

*Created 2026-08-28 in response to the public-repo audit (#31), which flagged that
the deployed Rust daemon is described alongside the Python 12-step consciousness
loop in a way that implies semantic equivalence. It is not equivalent. This doc is
the authoritative statement of what each actually does.*

## Honest summary

The Rust daemon (`sage-rs/`) is a lightweight, always-on **inference-and-metabolism
gateway**, not a port of the Python 12-step consciousness kernel. It faithfully
re-implements exactly two of the kernel's ideas — the five-state metabolic
controller (WAKE/FOCUS/REST/DREAM/CRISIS with ATP, hysteresis, circadian bias) and
the SNARC salience detectors — and adds live fleet federation (peer T3 trust,
monitoring, delegation) that the Python loop does not run. Everything else in the
"12-step loop" is either absent or reduced to a stub: it does not sense multiple
modalities (it derives a single word-count scalar from an incoming HTTP message),
does not select or budget across plugins (it always performs one action — an Ollama
completion), does not execute IRP plugins, learns nothing per cycle, has no
PolicyGate/conscience, no trust-posture computation (the `TrustPosture` struct is
dead code), no effect filtering, and no effectors (its only "act" is returning the
LLM's text to the caller). Its salience math is genuine but runs on a proxy signal,
and its ATP is a metabolic gauge rather than a resource budget. In short: the daemon
**shares vocabulary and two mechanisms** with the Python kernel while implementing a
fundamentally different, message-driven request/response system — operationally
useful and cheap (~12 MB RSS), but **not semantically equivalent** to the
consciousness loop.

## The 12 loop steps

| Step | Python kernel (`sage/core/sage_consciousness.py`) | Rust daemon (`sage-rs/`) | Verdict |
|---|---|---|---|
| 1. Sense | `_gather_observations()` multi-sensor (:1173) | `derive_observation()` word-count scalar (consciousness.rs:311) | **DIVERGES** — no real sensors; message-driven |
| 2. Salience | pre-exec mock (:1341), real post-LLM gated (:1864) | 5 real EWMA detectors on the scalar (consciousness.rs:189) | **DIVERGES** — real math, proxy input, different total formula |
| 3. Metabolize | MetabolicController 5-state (:999) | controller.rs:141, same 5 states | **CLOSEST TO EQUIVALENT** (independent constants) |
| 4. Posture (trust) | `_compute_trust_posture()` computed+enforced (:794) | `TrustPosture` struct is **dead code** (observation.rs:30) | **ABSENT in Rust** |
| 5. Select | `_select_attention_targets()` (:1417) | none (always one action) | **ABSENT** |
| 6. Budget | `_allocate_atp_budget()` trust-weighted (:1699) | ATP is a per-state counter, never allocated | **ABSENT as budgeting** |
| 7. Execute | IRP plugins via orchestrator (:1807) | single `ollama.generate()` (consciousness.rs:250) | **DIVERGES** — one hardcoded action |
| 8. Learn | trust-weight EMA + PolicyGate signals (:2718) | nothing per cycle (peer T3 is separate) | **ABSENT** |
| 9. Remember | 4 memory systems + DREAM consolidation (:2825) | one append-only salient JSONL (buffer.rs:108) | **PARTIAL / DIVERGES** |
| 10. Govern | PolicyGate conscience 8.5/8.6 (:1105) | none | **ABSENT** |
| 11. Filter | posture effect-restriction + CRISIS override (:1110) | none | **ABSENT** |
| 12. Act | typed effectors via registry, consumes ATP (:1129) | returns LLM text to HTTP caller (consciousness.rs:271) | **DIVERGES** — no effectors |

## Major subsystems

| Subsystem | Python | Rust | Verdict |
|---|---|---|---|
| SNARC 5D salience | mock pre / real post (:1864) | real detectors on scalar (snarc/surprise.rs:59) | DIVERGES — real math, proxy input |
| Metabolic states | MetabolicController | controller.rs:141 | Semantically closest |
| PolicyGate | :1105, :2945 | absent | ABSENT in Rust |
| Trust posture | :794 computed+enforced | dead struct (observation.rs:30) | ABSENT in Rust |
| Federation / peer trust | fleet.json config | live T3 reputation + monitor + delegate (peer_trust.rs:5) | **Rust is richer** |
| IRP plugin execution | HRMOrchestrator (:1807) | absent | ABSENT in Rust |
| LLM / tool execution | `_generate_llm_response` (:2040) + tool registry (:719) | Ollama inference only, no tool-calling | DIVERGES |
| Rust-only: shadow metabolism | — | coherence-as-valence parallel ATP, observation-only (consciousness.rs:24) | no Python analog |

## Note for other docs

`CLAUDE.md` and any status doc listing the Rust daemon under a "12-step consciousness
loop" heading should link here and use "inference-and-metabolism gateway" framing, so
no reader concludes the daemon runs the kernel. The Python kernel's own components are
partly mocked too — see `sage/docs/UNIFIED_CONSCIOUSNESS_LOOP.md` (:170, :270).
