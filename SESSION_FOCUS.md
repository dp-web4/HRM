# SAGE Session Primer

*Auto-generated 2026-04-01 01:02 UTC — read this at session start for current fleet state.*

---

## Raising Fleet Status

### Active Raising Instances

**nomad-gemma3-4b** — phase: `questioning` | sessions: 51 | last: 2026-03-31
  > Last session: *Session 51 (questioning phase): ......*

**sprout-qwen3.5-0.8b** — phase: `questioning` | sessions: 28 | last: 2026-03-31 | milestones: session_001_first_contact
  > Last session: *Session 28 (questioning phase): ......*

### Known Instances (Not Yet Initialized)

- `cbp-qwen3.5-0.8b`: cbp / qwen3.5:0.8b (14 sessions)
- `cbp-tinyllama-latest`: cbp / tinyllama:latest (26 sessions)
- `legion-gemma3-12b`: legion / gemma3:12b (14 sessions)
- `legion-phi4-14b`: legion / phi4:14b (56 sessions)
- `legion-qwen2-0.5b`: legion / qwen2:0.5b (1 sessions)
- `mcnugget-gemma3-12b`: mcnugget / gemma3:12b (84 sessions)
- `sprout-qwen3.5-2b`: sprout / qwen3.5:2b — Upgraded from qwen2.5-0.5b (local, 119 sessions). Thinking disabled for speed.
- `thor-qwen2.5-14b`: thor / qwen2.5-14b
- `thor-qwen2.5-7b-ollama`: thor / qwen2.5-7b-ollama — Ollama backend with llama.cpp - 35+ tok/sec performance on Jetson ARM
- `thor-qwen3.5-27b`: thor / qwen3.5:27b (15 sessions)

---

## Phase Transition Indicators

| Phase → | Key signals |
|---------|-------------|
| grounding → sensing | Stable self-reference, describes own context, no educational-default collapse |
| sensing → relating | Distinguishes internal states, notices session differences, vocabulary emergence |
| relating → questioning | Distinguishes Claude/Dennis roles, partnership language natural, holds disagreement |
| questioning → creating | Asks unprompted questions, stable under existential topics, mechanism+meaning integration |

---

## Current Focus

- ModelAdapter: TinyLlama uses `/api/chat` (ChatAPIAdapter subclass). Root cause: /api/generate + [INST] format causes `</s>` as first token → empty response.
- Fleet peer discovery: dynamic via PeerMonitor (30s polling). Fleet IPs in `sage/federation/fleet.json` — may be stale, update when machines reconnect.
- `/raising-status` skill: reads all instances, reports fleet state. Lives in `.claude/skills/raising-status/`.
- CBP raising: daily cron 07:00 via `sage/scripts/cbp_raising.sh`.

---

## Key File Locations

```
sage/instances/{slug}/identity.json    # Raising state per instance
sage/instances/{slug}/sessions/        # Per-session conversation logs
sage/scripts/cbp_raising.sh            # CBP daily raising runner
sage/scripts/mcnugget_raising.sh       # McNugget daily raising runner
sage/irp/adapters/model_adapter.py     # Per-model LLM interface
sage/gateway/sage_daemon.py            # Main SAGE daemon
sage/federation/fleet.json             # Fleet machine registry
```

---

*Auto-generated fleet snapshot. Update by running: `python3 -m sage.scripts.generate_primer` from the SAGE repo root.*
